"""XTTS-v2 zero-shot voice-clone TTS via coqui-tts.

Heavy imports (`torch`, `TTS`) are inside `_load_model` to keep the daemon
import-light when XTTS isn't actually used (eg. user opts into ElevenLabs).
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from loguru import logger

from ...config import XTTSConfig
from ...types import Lang
from ..duration_guard import DurationBaseline
from .base import TTSProvider

_LANG_CODE = {"zh": "zh-cn", "en": "en"}

# XTTS-v2 renders at 24 kHz mono; `inference_stream` yields float chunks we
# emit as little-endian 16-bit PCM, so ffplay needs these decode hints.
_SAMPLE_RATE = 24000
_STREAM_INPUT_ARGS = (
    # Skip input probing/buffering — the format is fully specified below,
    # so ffplay can start decoding on the first bytes.
    "-probesize", "32",
    "-analyzeduration", "0",
    "-fflags", "nobuffer",
    # NB: ffplay (unlike ffmpeg) has no `-ac`; mono must be spelled
    # `-ch_layout mono` (FFmpeg ≥ 5.1). `-ac 1` makes ffplay exit at spawn.
    "-f", "s16le", "-ar", str(_SAMPLE_RATE), "-ch_layout", "mono",
)

# Emotion → (speed multiplier, temperature delta). XTTS has no native
# emotion conditioning; these nudge the two knobs that audibly shift
# delivery — pace and sampling variance. Values are deliberately subtle:
# the Bettany voice reads as unhinged past ~±0.1 temperature.
_EMOTION_PROSODY: dict[str, tuple[float, float]] = {
    # session_start greeting — normal pace, a touch more melodic variance.
    "warm": (1.0, 0.03),
    "neutral": (1.0, 0.0),
    # idle nudge — slightly slower and flatter, an aside rather than an alert.
    "gentle": (0.95, -0.03),
    # tool_failure — measured pace, minimal variance: bad news read straight.
    "grave": (0.92, -0.05),
    # task_complete — brisker and brighter, the closest we get to celebratory.
    "pleased": (1.05, 0.08),
    # dry wit lands on a slight slow-down; a small variance bump keeps the
    # intonation from going fully deadpan.
    "sardonic": (0.96, 0.03),
}


def _prosody_for(emotion: str | None) -> tuple[float, float]:
    """(speed multiplier, temperature delta) for `emotion`.

    Unknown or absent emotions are a no-op (1.0, 0.0) — prosody shaping is
    best-effort garnish, never a reason to fail synthesis.
    """
    if emotion is None:
        return (1.0, 0.0)
    return _EMOTION_PROSODY.get(emotion, (1.0, 0.0))


# XTTS's GPT accepts only ~250 chars per generation (its tokenizer's
# char_limits); anything beyond is silently dropped — the tail of a long
# announcement just vanishes. coqui's own enable_text_splitting needs spaCy
# (which we don't ship), so we split at sentence boundaries ourselves and
# generate piece by piece. 240 leaves headroom under the 250 limit.
_GPT_CHAR_LIMIT = 240
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;。!?;])\s+")

# Cold speaker amps take ~200 ms to wake from silence and eat whatever plays
# during that window — synthesized speech starts at full energy within ~1 ms,
# so the first phoneme was getting swallowed. Lead every utterance with a
# quarter second of silent PCM: the wake-up burns padding, not speech.
_PREROLL_SECONDS = 0.25

# XTTS reads an em/en dash as a dead stop — measured (2026-07-09, 3-run
# probe per case) at 0.42-1.2s for a single dash versus 0.36-0.56s for a
# comma, with run-to-run variance users hear as random dragging. The Jarvis
# house style leans on "clause — clause" in nearly every line, so every
# notification carried that lottery. The written form keeps its dashes;
# the audio gets commas. ASCII hyphens (file names, CLI flags) untouched.
_DASH_RUN = re.compile(r"\s*[—–]+\s*")


def _cap_mps_allocator(ratio: float) -> None:
    """Bound PyTorch's MPS caching allocator (see XTTSConfig.mps_memory_ratio).

    PyTorch reads both watermark ratios when it builds the allocator, and
    rejects a low watermark above the high one — so they are only ever set as
    a pair, low at half of high. `setdefault`, not assignment: an operator who
    has pinned either ratio in the environment (or the launchd plist) has
    already made this decision and must win.
    """
    if ratio <= 0:  # explicit opt-out — keep PyTorch's own default
        return
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(ratio))
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(ratio / 2))


def _normalize_pauses(text: str, lang: str) -> str:
    """Rewrite dash pauses as comma pauses in the text handed to the GPT."""
    sep = "，" if lang == "zh" else ", "
    out, n = _DASH_RUN.subn(sep, text)
    if n:
        # An utterance-final dash (LLM trailing off) must not become a
        # dangling comma — drop it and let the sentence just end.
        out = out.rstrip(" ,，")
    return out


def _split_for_gpt(text: str, limit: int = _GPT_CHAR_LIMIT) -> list[str]:
    """Split `text` into GPT-sized pieces at sentence boundaries.

    Sentences are packed greedily up to `limit`; a single overlong sentence
    is hard-split (degraded prosody beats a vanished tail).
    """
    pieces: list[str] = []
    current = ""
    for sent in _SENTENCE_BOUNDARY.split(text.strip()):
        if not sent:
            continue
        candidate = f"{current} {sent}" if current else sent
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(sent) > limit:
            pieces.append(sent[:limit])
            sent = sent[limit:]
        current = sent
    if current:
        pieces.append(current)
    return pieces


def _to_int16_pcm(wav: Any) -> Any:
    """Float waveform → int16 samples.

    Clip rather than peak-normalise: peak-normalising each utterance
    independently makes loudness jump between lines. XTTS output is already
    ~[-1, 1]; occasional overshoot is clamped here.
    """
    import numpy as np  # type: ignore

    arr = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767).astype(np.int16)


class XTTSProvider(TTSProvider):
    name = "xtts"
    # The GPT decoder is autoregressive — chunked streaming lets ffplay start
    # playing the first ~half-second while the rest is still decoding, cutting
    # perceived latency by ~60-70% versus waiting for the whole utterance.
    supports_streaming = True
    stream_input_args = _STREAM_INPUT_ARGS
    # Same byte stream, described for the in-process sounddevice sink; the
    # ffplay flags above remain the fallback decode spec.
    stream_pcm = (_SAMPLE_RATE, 1)

    def __init__(self, cfg: XTTSConfig) -> None:
        self.cfg = cfg
        # Must happen before torch's MPS allocator initialises, which is why
        # it sits in __init__ (daemon construction) and not in _load_model:
        # the ratios are read once, when the allocator is first built.
        if cfg.device == "mps":
            _cap_mps_allocator(cfg.mps_memory_ratio)
        self._model: Any | None = None
        self._latents: dict[str, Any] | None = None
        # (gpt_cond_latent, speaker_embedding) per language for the streaming
        # path — loaded from the .pth embedding (en) or computed once from the
        # ref wav (zh / no embedding) and kept resident.
        self._cond: dict[str, tuple[Any, Any]] = {}
        # Serializes the lazy loads: prewarm at daemon start can race the
        # first real event, and without this both worker threads would load
        # the multi-GB model. RLock because _conditioning_for may call
        # _load_model while holding it.
        self._warm_lock = threading.RLock()
        # Babble-tail guard: per-text clean-duration baseline, judged in
        # _generate_piece. See XTTSConfig.duration_ratio_threshold.
        self._baseline = DurationBaseline(
            Path(cfg.duration_baseline_path).expanduser()
        )

    def _generate_piece(
        self,
        infer: Callable[[str], Any],
        piece: str,
        lang: Lang,
        stop: threading.Event | None = None,
    ) -> Any:
        """Run `infer(piece)`, again while the take runs past the text's
        clean duration; return the accepted float32 waveform.

        XTTS's GPT sometimes misses its stop token after a very short input
        and pads the line with a second or two of gibberish — heard as a
        "wa wa wa" tail. Nothing about such a take is knowable up front,
        but it IS longer than a clean one, so this is the CosyVoice
        double-take loop (cosyvoice.synthesize) applied per GPT piece:
        judge by duration against the text's baseline, regenerate on a
        verdict, ship the shortest take if every attempt is flagged. Rates
        and thresholds are in XTTSConfig.

        Synchronous — runs on the caller's worker thread. `stop` (streaming)
        ends the retries when playback has been cancelled; whatever is
        returned then is never enqueued anyway.
        """
        import numpy as np  # type: ignore

        fallback_cps = (
            self.cfg.fallback_cps_zh if lang == "zh" else self.cfg.fallback_cps
        )
        attempts = max(1, self.cfg.max_synth_attempts)
        best: Any = None
        best_duration = 0.0
        attempt = 0
        for attempt in range(1, attempts + 1):
            wav = np.asarray(infer(piece)["wav"], dtype=np.float32)
            duration = len(wav) / _SAMPLE_RATE
            verdict = self._baseline.check(
                piece, duration,
                ratio_threshold=self.cfg.duration_ratio_threshold,
                fallback_cps=fallback_cps,
            )
            if duration > 0 and (best is None or duration < best_duration):
                best, best_duration = wav, duration
            if not verdict.is_repeat:
                # Clean takes update the rolling-window baseline.
                self._baseline.record(piece, duration)
                if attempt > 1:
                    logger.info(
                        "xtts retry succeeded on attempt {}: {:.1f}s "
                        "(ratio {:.2f}) text={!r}",
                        attempt, duration, verdict.ratio, piece,
                    )
                return wav
            logger.warning(
                "xtts babble tail suspected: {:.1f}s audio for {} chars "
                "(expected {:.1f}s, ratio {:.2f}), attempt {}/{}; "
                "regenerating text={!r}",
                duration, len(piece), verdict.expected, verdict.ratio,
                attempt, attempts, piece,
            )
            if stop is not None and stop.is_set():
                break
        if best is None:
            return np.zeros(0, dtype=np.float32)
        # Every attempt flagged (or cancelled mid-retry): ship the shortest.
        # Escape valve as in CosyVoice — learn its duration only if its pace
        # is plausibly clean, so a baseline that drifted low can recover
        # instead of flagging this text forever; a real babble take (too
        # slow) is left unlearned.
        self._baseline.record(piece, best_duration, min_cps=fallback_cps * 0.75)
        logger.warning(
            "xtts shipping shortest take ({:.1f}s) after {} attempts; text={!r}",
            best_duration, attempt, piece,
        )
        return best

    def _ref_audio_for(self, lang: Lang) -> Path:
        path = self.cfg.ref_audio_zh if lang == "zh" else self.cfg.ref_audio_en
        return Path(path)

    def _embedding_path(self, lang: Lang) -> Path | None:
        """Configured speaker-embedding `.pth` for `lang`, or None.

        The bundled Jarvis (Bettany) embedding only sounds right in English —
        speaking Chinese through it is muddy — so Chinese always falls back to
        the ref-audio clone path regardless of `speaker_embedding`.
        """
        if lang == "zh" or not self.cfg.speaker_embedding:
            return None
        path = Path(self.cfg.speaker_embedding)
        return path if path.is_file() else None

    def _load_model(self) -> Any:
        """Lazy-load the XTTS-v2 model. Called at most once per provider lifetime."""
        if self._model is not None:
            return self._model
        with self._warm_lock:
            if self._model is not None:
                return self._model
            logger.info("Loading XTTS-v2 model from {} on {}", self.cfg.model_dir, self.cfg.device)
            from TTS.api import TTS  # type: ignore

            self._model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(self.cfg.device)
            return self._model

    def _load_latents(self) -> dict[str, Any]:
        """Lazy-load the pre-extracted speaker embedding onto the model device."""
        if self._latents is not None:
            return self._latents
        with self._warm_lock:
            if self._latents is not None:
                return self._latents
            import torch  # type: ignore

            path = self.cfg.speaker_embedding
            logger.info("Loading XTTS speaker embedding from {}", path)
            # The .pth is a plain dict of tensors (not weights-only safe), hence
            # weights_only=False. Move both latents onto the synthesis device so
            # inference doesn't trip over a CPU/MPS tensor mismatch.
            raw = torch.load(path, map_location=self.cfg.device, weights_only=False)
            self._latents = {
                "gpt_cond_latent": raw["gpt_cond_latent"].to(self.cfg.device),
                "speaker_embedding": raw["speaker_embedding"].to(self.cfg.device),
            }
            return self._latents

    def _speed_for(self, text: str) -> float:
        return (
            self.cfg.speed_short
            if len(text) < self.cfg.short_threshold_chars
            else self.cfg.speed_long
        )

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        # XTTS has no notion of an EL-style voice_id (clones from ref audio /
        # embedding); the override is ignored. Future: could resolve voice_id
        # to a named embedding under voices/<voice_id>.pth for multi-voice.
        _ = voice_id
        text = _normalize_pauses(text, lang)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Emotion shapes prosody, not timbre: nudge speed and sampling
        # temperature (clamped — below ~0.3 XTTS turns robotic, above ~0.85
        # it starts repeating words).
        speed_mult, temp_delta = _prosody_for(emotion)
        speed = self._speed_for(text) * speed_mult
        temperature = min(max(self.cfg.temperature + temp_delta, 0.3), 0.85)
        language = _LANG_CODE.get(lang, "en")
        embedding = self._embedding_path(lang)

        if embedding is None:
            ref = self._ref_audio_for(lang)
            if not ref.is_file():
                raise FileNotFoundError(f"reference audio missing: {ref}")

            def _run_ref() -> None:
                model = self._load_model()
                model.tts_to_file(
                    text=text,
                    speaker_wav=str(ref),
                    language=language,
                    file_path=str(out_path),
                    temperature=temperature,
                    speed=speed,
                )

            await asyncio.to_thread(_run_ref)
            return out_path

        def _run_embedding() -> None:
            import numpy as np  # type: ignore
            from scipy.io import wavfile  # type: ignore

            tts = self._load_model()
            latents = self._load_latents()
            sr = int(getattr(tts.synthesizer, "output_sample_rate", _SAMPLE_RATE))
            # Amp-wake pre-roll + per-piece generation (GPT char limit) —
            # see _PREROLL_SECONDS / _split_for_gpt for the two whys.
            parts = [np.zeros(int(sr * _PREROLL_SECONDS), dtype=np.float32)]
            for piece in _split_for_gpt(text):
                parts.append(self._generate_piece(
                    lambda p: tts.synthesizer.tts_model.inference(
                        text=p,
                        language=language,
                        gpt_cond_latent=latents["gpt_cond_latent"],
                        speaker_embedding=latents["speaker_embedding"],
                        temperature=temperature,
                        speed=speed,
                    ),
                    piece, lang,
                ))
            wavfile.write(str(out_path), sr, _to_int16_pcm(np.concatenate(parts)))

        await asyncio.to_thread(_run_embedding)
        return out_path

    def _conditioning_for(self, lang: Lang) -> tuple[Any, Any]:
        """Resolve (gpt_cond_latent, speaker_embedding) for `lang`, cached.

        English uses the pre-extracted .pth embedding (fast, timbre-stable);
        Chinese / no-embedding computes the latents once from the ref wav via
        the model's own encoder. Runs synchronously — callers invoke it from a
        worker thread.
        """
        cached = self._cond.get(lang)
        if cached is not None:
            return cached
        with self._warm_lock:
            cached = self._cond.get(lang)
            if cached is not None:
                return cached
            if self._embedding_path(lang) is not None:
                # Same tensors the batch path uses — one load, one resident copy.
                latents = self._load_latents()
                cond = (latents["gpt_cond_latent"], latents["speaker_embedding"])
            else:
                ref = self._ref_audio_for(lang)
                if not ref.is_file():
                    raise FileNotFoundError(f"reference audio missing: {ref}")
                model = self._load_model()
                gpt_cond_latent, speaker_embedding = (
                    model.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=str(ref),
                    )
                )
                cond = (gpt_cond_latent, speaker_embedding)
            self._cond[lang] = cond
            return cond

    async def stream(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield 16-bit PCM chunks as the GPT decoder produces them.

        The blocking `inference_stream` generator runs in a worker thread and
        feeds chunks back to the event loop through a queue, so playback can
        begin on the first chunk while the rest is still decoding.
        """
        _ = voice_id  # XTTS clones from ref/embedding; no swappable voice id.
        text = _normalize_pauses(text, lang)
        language = _LANG_CODE.get(lang, "en")
        # Same prosody shaping as the batch path — see synthesize().
        speed_mult, temp_delta = _prosody_for(emotion)
        speed = self._speed_for(text) * speed_mult
        temperature = min(max(self.cfg.temperature + temp_delta, 0.3), 0.85)

        loop = asyncio.get_running_loop()
        # None is the end-of-stream marker (PCM payloads are always bytes).
        queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        # Set when the consumer goes away (playback cancelled) so the worker
        # thread stops decoding instead of finishing the whole utterance.
        stop = threading.Event()

        def _produce() -> None:
            try:
                model = self._load_model()
                gpt_cond_latent, speaker_embedding = self._conditioning_for(lang)
                # Amp-wake pre-roll (see _PREROLL_SECONDS): silent int16 PCM,
                # emitted before decoding so the audio device opens and wakes
                # while the GPT is still working on the first chunk.
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    bytes(2 * int(_SAMPLE_RATE * _PREROLL_SECONDS)),
                )
                # Per-piece generation: the GPT silently truncates past its
                # char limit — see _split_for_gpt.
                # Piece-buffered delivery: measured IN-DAEMON decode runs at
                # RTF 1.1–1.7 (slower than realtime; the 0.76 standalone
                # benchmark does not hold under daemon conditions), so
                # chunk-by-chunk streaming inevitably starves any real-time
                # sink — heard as word-by-word, half-word playback. Instead
                # each piece is decoded COMPLETELY and enqueued as one blob:
                # within a piece playback is gapless by construction, and the
                # next piece decodes while the current one plays. The cost is
                # a possible short pause at piece boundaries — a natural
                # sentence break, not a mid-word chop.
                #
                # And because the piece is buffered whole anyway, this uses
                # `inference` rather than `inference_stream`: nothing consumed
                # the chunks incrementally, so the streaming decoder bought
                # nothing while costing roughly double the GPU memory —
                # measured 2026-08-25, an 8-10 GB plateau versus 4.3 GB, on a
                # machine whose unified memory the allocator never gives back
                # (see XTTSConfig.mps_memory_ratio). Head-to-head on identical
                # text, both orderings, three runs each: batch was equal or
                # faster at every length (RTF medians 0.55/0.68 vs 0.70/0.82
                # short, 1.09/0.93 vs 1.22/1.24 long), the spread between runs
                # of one path being wider than the gap between paths.
                pieces = _split_for_gpt(text)
                for i, piece in enumerate(pieces):
                    if stop.is_set():
                        break
                    t0 = time.perf_counter()
                    # Cancellation is now only checked between pieces (and
                    # between babble-guard retries): a batch decode has no
                    # yield point. Nothing is audible either way — a
                    # cancelled piece was never enqueued — so the only cost
                    # is finishing a decode whose audio is then dropped.
                    wav = self._generate_piece(
                        lambda p: model.synthesizer.tts_model.inference(
                            text=p,
                            language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=temperature,
                            speed=speed,
                        ),
                        piece, lang, stop=stop,
                    )
                    # inference() returns host-side samples (the file path
                    # relies on that too); the guard hands back float32.
                    blob = bytearray(_to_int16_pcm(wav).tobytes())
                    if blob and not stop.is_set():
                        loop.call_soon_threadsafe(queue.put_nowait, bytes(blob))
                    wall = time.perf_counter() - t0
                    audio_s = len(blob) / 2 / _SAMPLE_RATE
                    logger.debug(
                        "xtts stream piece {}/{}: {:.1f}s audio in {:.1f}s wall"
                        " (rtf {:.2f})",
                        i + 1, len(pieces), audio_s, wall,
                        wall / audio_s if audio_s else float("inf"),
                    )
            except Exception as exc:  # noqa: BLE001 — surfaced to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        fut = loop.run_in_executor(None, _produce)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            await fut

    async def prewarm(self) -> None:
        """Load the model and English conditioning latents off-thread so the
        first real notification doesn't eat the multi-second cold start — for
        XTTS on MPS this load is the single biggest one-off cost."""

        def _warm() -> None:
            self._load_model()
            self._conditioning_for("en")

        await asyncio.to_thread(_warm)

    async def healthcheck(self) -> bool:
        # English is served by the embedding when present; zh always needs its
        # ref wav. Stay healthy as long as English can be produced.
        en_ok = self._embedding_path("en") is not None or self._ref_audio_for("en").is_file()
        return en_ok
