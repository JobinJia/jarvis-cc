"""TOML-backed config with safe defaults. Layered: file > defaults."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def resolve_api_key(cfg: object) -> str | None:
    """Resolve a provider's API key. An inline ``api_key`` set in config.toml
    wins; otherwise fall back to the environment variable named by
    ``api_key_env``. This lets keys live in the TOML alongside the rest of the
    config instead of the launchd plist's XML — empty/unset inline falls
    through to the env var, so existing env-based setups keep working."""
    return getattr(cfg, "api_key", "") or os.getenv(cfg.api_key_env)


@dataclass
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 5.0


@dataclass
class AnthropicConfig:
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    model: str = "claude-haiku-4-5-20251001"
    timeout_seconds: float = 5.0


@dataclass
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 5.0


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: float = 10.0
    # How long Ollama keeps the model resident after a request. Notifications
    # are bursty and often >5 min apart; at the server default (5m) the 8B
    # model gets evicted between events, so the next one eats a multi-second
    # cold reload. Keeping it warm for 30m trades a little RAM for a steady
    # sub-second first token. Set "-1" to pin it permanently, "0" to unload
    # immediately after each call.
    keep_alive: str = "30m"
    # Context window to request per call. Ollama 0.33's server default is
    # 32768, and the KV cache for that alone is ~4-5 GB on an 8B model (or
    # ~3.7 GB even on qwen3:0.6b) — memory the daemon then fights XTTS for.
    # A Jarvis prompt is ~800 tokens (system + few-shots + avoid list) plus
    # a 200-token reply, so 2048 leaves headroom while keeping the model's
    # footprint close to its weights.
    num_ctx: int = 2048


@dataclass
class ZhipuConfig:
    """Zhipu AI (智谱) GLM, OpenAI-compatible chat API — a free cloud fallback
    for phrasing when the local Ollama is down (real-name verification
    required). Defaults to ``glm-4-flash``: the always-free workhorse, reliable
    for one-line phrasing. The stronger ``glm-4.7-flash`` is also free but its
    free tier is frequently rate-limited (HTTP 429 code 1305). Note the
    endpoint is ``.../paas/v4/chat/completions`` — NO ``/v1`` segment, which is
    a common 404 trap when reusing OpenAI clients."""
    api_key_env: str = "ZHIPU_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    model: str = "glm-4-flash"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    timeout_seconds: float = 5.0


@dataclass
class SiliconFlowConfig:
    """SiliconFlow (硅基流动), OpenAI-compatible chat API — a second free cloud
    fallback alongside Zhipu (different provider, different rate-limit pool, so
    one being throttled doesn't take both down). ``Qwen/Qwen2.5-7B-Instruct`` is
    always-free, plenty for one-line phrasing. China-direct, no proxy. Standard
    OpenAI endpoint at ``{base_url}/v1/chat/completions``."""
    api_key_env: str = "SILICONFLOW_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = "https://api.siliconflow.cn"
    timeout_seconds: float = 5.0


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    # Single fallback (back-compat). For a multi-level chain set `fallbacks`
    # instead — when non-empty it takes precedence over `fallback`.
    fallback: str = "ollama"
    fallbacks: list[str] = field(default_factory=list)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    zhipu: ZhipuConfig = field(default_factory=ZhipuConfig)
    siliconflow: SiliconFlowConfig = field(default_factory=SiliconFlowConfig)


@dataclass
class XTTSConfig:
    model_dir: str = "~/.jarvis/models/xtts-v2"
    ref_audio_zh: str = "~/.jarvis/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis/voices/jarvis_en.wav"
    # Pre-extracted speaker embedding (.pth holding `gpt_cond_latent` +
    # `speaker_embedding`). When set and present, the provider clones from
    # this cached latent via `inference()` instead of re-encoding a ref wav
    # on every call — faster and timbre-stable. The bundled Jarvis (Paul
    # Bettany) embedding sounds noticeably better than our ref-wav clone, so
    # it is the default fixed voice. English only — the Bettany timbre sounds
    # muddy speaking Chinese, so the zh path always uses ref_audio_zh instead.
    # Empty string falls back to the ref_audio_{zh,en} clone path above.
    speaker_embedding: str = "~/.jarvis/voices/jarvis_speaker.pth"
    device: str = "mps"
    # Ceiling on PyTorch's MPS caching allocator, as a fraction of the GPU's
    # recommended working-set size (PyTorch's own default is 1.7). Only read
    # when device is "mps".
    #
    # Left alone, that allocator grows to whatever RAM happens to be free and
    # never returns the pages: measured 2026-08-25, ONE 240-char synthesis
    # took the process to 23.5 GB of unified — i.e. real — memory on an idle
    # 32 GB machine, and the live daemon sat at 13 GB after a day, which is
    # what drove the box 30 GB into swap and dragged synthesis from RTF 1.2 to
    # a median of 4.7. `torch.mps.empty_cache()` does NOT fix this: it hands
    # blocks back to PyTorch's pool (driver_allocated drops to 0) while the
    # pages stay charged to the process, so only the cap actually bounds it.
    #
    # The allocator grows to whatever this allows, so the cap — not the decode
    # path — sets the daemon's resident size. What the path decides is how
    # tight the cap can be before synthesis starts failing: the old
    # inference_stream path OOMed on every utterance at 0.2 and fell through
    # to the `say` voice, while the batch path used since 2026-08-25 holds
    # 0.2 across repeated worst-case (240-char) lines. Raise it if "MPS
    # backend out of memory" shows up in the log; 0 disables the cap.
    mps_memory_ratio: float = 0.2
    # XTTS GPT decoder sampling temperature. Library default 0.75 is too
    # high for our use case — short Jarvis-toned commands suffer audible
    # word repetition / pacing drift across takes. 0.5 cuts variance to
    # the point where successive synths of the same line are consistent.
    temperature: float = 0.5
    # XTTS's GPT learned an asymmetry from training data: short utterances
    # are delivered slowly (with pauses for emphasis), long utterances flow
    # quickly. A single `speed` multiplier therefore over-speeds long text
    # while still feeling sluggish on short status lines. We split the knob:
    # texts under `short_threshold_chars` get `speed_short` (faster),
    # longer ones get `speed_long` (closer to 1.0). 1.30 felt rushed in the
    # Bettany voice on short status lines; 1.15 reads a touch slower (~9%)
    # while staying snappy.
    speed_short: float = 1.15
    speed_long: float = 1.00
    short_threshold_chars: int = 60
    # Babble-tail guard — the per-text duration baseline CosyVoice uses for
    # double-takes (see tts/duration_guard.py), applied to each GPT piece.
    # XTTS's GPT sometimes misses its stop token after a very short input
    # and pads the line with a second or two of gibberish. Measured
    # 2026-09-08 over the daemon log: 4.9% of 12-22-char sentences ran past
    # 2x their expected length (14 of 285) versus 0-1% of 35-60-char ones —
    # and since the two-short-sentence phrasing (2026-09-01) most lines are
    # short, so ~1 announcement in 10 carried a tail. A take running past
    # baseline x duration_ratio_threshold is regenerated, up to
    # max_synth_attempts in total; if every attempt is flagged the shortest
    # ships. A retry on a short piece costs ~1s wall.
    #
    # fallback_cps is the pre-baseline estimate: clean XTTS English measured
    # median 13.8 chars/s (p10 9.5), so 12 with the 1.5x threshold flags
    # anything under 8 chars/s — where the babble takes sat (5.6-7.7) and
    # clean takes did not. Chinese runs ~3x slower per char, as in CosyVoice.
    # The baseline file is separate from CosyVoice's: different model,
    # different pace.
    duration_ratio_threshold: float = 1.5
    fallback_cps: float = 12.0
    fallback_cps_zh: float = 4.5
    max_synth_attempts: int = 3
    duration_baseline_path: str = "~/.jarvis/cache/xtts_duration_baseline.json"
    # `stream_chunk_size` lived here until 2026-08-25, tuned to 20 on MPS.
    # It is gone rather than deprecated: the provider no longer calls
    # `inference_stream` at all (see xtts._produce — each piece was already
    # buffered whole, so the streaming decoder bought nothing and cost double
    # the GPU memory), and a knob that silently drives nothing is worse than
    # no knob. Unknown keys are ignored by _merge, so configs still setting it
    # keep loading.


@dataclass
class ElevenLabsConfig:
    api_key_env: str = "ELEVENLABS_API_KEY"
    api_key: str = ""  # inline key (config.toml); falls back to api_key_env
    voice_id: str = ""
    model: str = "eleven_turbo_v2_5"


@dataclass
class CosyVoiceConfig:
    """CosyVoice 3 (FunAudioLLM) via the Apache-2.0 Rust+Candle binding
    `cosyvoice3.rs`. Apple Silicon Metal acceleration; no PyTorch needed.

    Quality outperforms XTTS-v2 on speaker similarity in our A/B; the
    permissive license also clears the OSS path that XTTS's CPML blocks.
    """
    model_dir: str = "~/.jarvis/models/cosyvoice3-0.5b-candle"
    ref_audio_zh: str = "~/.jarvis/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis/voices/jarvis_en.wav"
    # Transcript of each ref audio. When provided, the provider routes
    # through inference_zero_shot (which uses the transcript to ground the
    # LLM and prevent the double-take loop that cross_lingual mode falls
    # into on short utterances). Leave blank to use cross_lingual instead.
    ref_text_zh: str = ""
    ref_text_en: str = ""
    # CFM sampling steps for the flow decoder. Library default is 10;
    # we drop to 5 because the flow-decoder cost scales linearly with
    # steps and the quality delta on Bettany short lines is sub-perceptible
    # in A/B testing. Bump back to 10 if you ever hear artifacts.
    n_timesteps: int = 5
    # Double-take detection (see tts/duration_guard.py). After each synth we
    # compare its duration against the text's clean baseline (median of a
    # rolling window of recent clean takes); a synth running past baseline x
    # duration_ratio_threshold is a repeat and gets retried. `fallback_cps`
    # drives the chars/cps estimate used before a per-text baseline exists.
    # (An SSM self-similarity approach was tried and discarded — on real audio
    # it couldn't separate a true double-take from the ordinary self-similarity
    # of normal long speech.) `save_synth_samples` dumps each synth + metadata
    # to sample_dir for offline analysis.
    #
    # Threshold is 1.5 against the MEDIAN baseline: clean takes vary ±30-40%, so
    # a normal take lands within ~1.3x of the median while a full double-take is
    # ~2x — 1.5 sits cleanly between. (An earlier 1.35 against the per-text
    # MINIMUM mis-flagged ~30% of clean takes, because the min was the fastest
    # fluke, not the typical length.) Trade-off: a partial double-take near 1.4x
    # may pass — duration alone cannot separate it from clean jitter.
    duration_ratio_threshold: float = 1.5
    fallback_cps: float = 12.0
    # Chinese runs ~4-5 chars/second — a third of English's chars-per-second
    # rate. With the en value a clean 15-char zh take (≈3.8s vs "expected"
    # 1.25s) mis-flags as a double-take and burns every retry attempt
    # (measured 2026-07-10: 4x20s futile resynths per idle line).
    fallback_cps_zh: float = 4.5
    max_synth_attempts: int = 4
    save_synth_samples: bool = False
    sample_dir: str = "~/.jarvis/cache/samples"
    duration_baseline_path: str = "~/.jarvis/cache/duration_baseline.json"


@dataclass
class PiperConfig:
    """Piper TTS (rhasspy) via the MIT `piper-tts` wheel: ONNX, CPU-only, no
    PyTorch. Unlike the CosyVoice zero-shot clone, Piper renders from a FIXED
    single-speaker model, so the speaker/accent is baked into the weights and
    cannot drift (the CosyVoice 0.5B clone intermittently drifted to an Indian
    accent) and there is no flow-decoder double-take to guard against. Warm
    per-utterance RTF is ~0.03 (≈30x realtime) since the model stays resident.

    Voices are `<name>.onnx` (+ `.onnx.json`) under `data_dir`; fetch with
    `python -m piper.download_voices <name> --data-dir <data_dir>`.
    """
    data_dir: str = "~/.jarvis/models/piper"
    # British male butler voice — the default Jarvis identity is English.
    # Swap to a JARVIS-tuned voice (e.g. jgkawell/jarvis on HF) for closer timbre.
    voice_en: str = "en_GB-alan-medium"
    voice_zh: str = "zh_CN-huayan-medium"


@dataclass
class TTSConfig:
    provider: str = "xtts"
    fallback: str = "say"
    # Optional Chinese-primary override (e.g. "piper" for a native-Mandarin
    # fixed voice). The XTTS Bettany clone is English-born — cross-lingual
    # cloning reads Chinese with a foreign accent — so zh routes to a native
    # speaker while en keeps the clone. Empty = no override.
    provider_zh: str = ""
    # Route raw-PCM streaming through the in-process sounddevice sink instead
    # of an ffplay subprocess. On by default since 2026-08-25: the jitter
    # buffer this switch was waiting for exists (PCMPlayer prebuffers, then
    # pads underruns with clean silence), and the ffplay alternative turned
    # out to be the worse of the two — it paces off a clock that keeps running
    # while its stdin is dry, so a slower-than-realtime decoder loses most of
    # each utterance (a 5s stall cut a 5s line to 1.0s under measurement).
    # Turning this off no longer routes raw PCM to ffplay; it disables
    # streaming for those providers entirely, in favour of synth+afplay.
    pcm_playback: bool = True
    # Run the heavy providers (see tts.factory.HEAVY_PROVIDERS) in a child
    # process the daemon can replace. They leak native memory per utterance —
    # ~40 MB for XTTS, linear, no plateau, inside torch/coqui rather than our
    # code — and native memory only returns when a process ends. Recycling the
    # DAEMON would reclaim it too, but at the cost of dropping the socket, the
    # event queue and the warmed retrieval index; recycling a child costs a
    # model reload and nothing else. Off puts the model back in-process, which
    # is simpler to debug and fine for a short-lived daemon.
    worker_process: bool = True
    # Syntheses a worker serves before the daemon replaces it at its next idle
    # moment. 100 lines is roughly 4 GB of leak — about a day at the observed
    # rate of 76 utterances in 6.5 hours. Lower trades more model reloads for a
    # smaller ceiling; 0 disables recycling and lets the child leak forever.
    worker_max_syntheses: int = 100
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    cosyvoice: CosyVoiceConfig = field(default_factory=CosyVoiceConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)


@dataclass
class PrivacyConfig:
    cloud_redaction: bool = True


@dataclass
class SessionBriefingConfig:
    """Iron-Man-style opening briefing on new CC/Codex sessions:
    greeting + local time + weather, English voice, no LLM round-trip.
    """
    enabled: bool = True
    # City queried against the weather sources. Empty = derive from
    # timezone tail (`Asia/Shanghai` → "Shanghai"). Override to pin a
    # location when the timezone is a continent root or you're abroad
    # on a VPN.
    city: str = ""
    # How long a single weather lookup is reused across briefings — keeps
    # us off the weather APIs if you open ten sessions in two minutes.
    weather_ttl_seconds: int = 600
    # When a fresh fetch fails (network blip, VPN switch), a previously
    # fetched snapshot no older than this is spoken instead of dropping
    # to a time-only line. Slightly stale weather beats none.
    weather_stale_max_age_seconds: int = 7200
    # Floor between briefings. 0 = every session_start speaks. Bump up if
    # you open many tabs and find the chorus tiresome.
    min_interval_seconds: int = 0
    # HTTP timeout for the weather fetch. Briefing falls back to a
    # time-only line if this elapses — we never block the worker on it.
    weather_timeout_seconds: float = 3.0


@dataclass
class BehaviorConfig:
    dedup_window_seconds: int = 10
    queue_max_size: int = 5
    # User-facing language switch for what Jarvis SPEAKS. Accepted values:
    # "en" (English, default — matches the user's chosen British voice
    # identity), "zh" (Chinese), or "auto" (pick per-event from content).
    # Honored by both the LLM phrase path and the hook AskUserQuestion path.
    voice_language: str = "en"
    events: list[str] = field(
        default_factory=lambda: [
            "permission_prompt",
            "idle_prompt",
            "elicitation_dialog",
            "ask_user_question",
            "session_start",
            "tool_failure",
            # `task_complete` (CC Stop) fires after every assistant turn, so
            # it stays opt-in: not in the default allowlist. Add it to
            # `[behavior].events` in config.toml to hear "All done, sir."
            # Tier 1 lifecycle events — on by default.
            "context_compacting",
            "rate_limited",
            "subagent_spawned",
            "max_turns_reached",
            # Tier 2 lifecycle events stay opt-in: add to `[behavior].events`
            # in config.toml to enable (api_error, session_end,
            # context_compacted, context_overflow).
        ]
    )
    # DEPRECATED: kept so old config.toml files don't error on load. Not read
    # at runtime; replaced by phrase_target_chars + phrase_hard_cap below.
    phrase_max_chars: int = 30
    phrase_target_chars: int = 70
    phrase_hard_cap: int = 120
    # When True (default), the hook sends a cancel signal on UserPromptSubmit /
    # PostToolUse so the daemon stops any in-flight audio for that session.
    cancel_on_user_action: bool = True
    # When True (default), hook events fired from INSIDE a subagent's work
    # (its tool calls, failures, prompts — payloads carrying both `agent_id`
    # and `agent_type`) are dropped before reaching the daemon, so only
    # main-session activity speaks. The SubagentStart lifecycle notice is
    # exempt — it stays governed by the `events` allowlist above.
    mute_subagent_events: bool = True
    # Event types immune to that cancel: they keep playing (and stay queued)
    # when the session moves on. A permission/idle prompt goes stale the
    # moment the user acts, but a failure notice stays true — and with a
    # slower-than-realtime local TTS the announcement often starts only
    # after the session's next tool call, whose PostToolUse cancel would
    # otherwise cut it off one word in ("Sir, —").
    cancel_exempt_events: list[str] = field(
        default_factory=lambda: ["tool_failure"]
    )
    # Drop LLM-phrased events older than this (seconds) at dequeue time — a
    # backlog burst otherwise speaks stale notifications the user already
    # acted on. 0 disables. Pre-baked text (`say --text`) and session_start
    # briefings are exempt: those should speak whenever they surface.
    stale_event_max_age_seconds: float = 60.0
    # How much wit Jarvis allows himself, 0-3. Selects both the tone clause
    # AND the few-shot example set in the phrase prompt (the examples are
    # what actually move a small model), plus the briefing tone.
    #   0 — deadpan formal butler (no jokes)
    #   1 — hint of dry wit (default for first-time users)
    #   2 — MCU Jarvis: dry banter, witty asides
    #   3 — Tony-mode: openly sardonic, never sycophantic
    # Out-of-range values are clamped on load by `load_config`.
    # Adjust live with `jarvis tone <0-3>` — no daemon restart needed.
    humor_level: int = 1
    # How Jarvis addresses the user, per output language. Substituted into
    # the few-shot examples as well as the system prompt, so small models
    # actually honor it.
    address_en: str = "Sir"
    address_zh: str = "先生"
    # Streaming pipeline: overlap LLM token generation, sentence chunking,
    # and TTS synthesis so playback of the first sentence starts before the
    # LLM finishes. Reduces perceived latency from 2-6s to ~1s. Opt-in
    # because it requires providers that support streaming and changes the
    # playback cadence (multiple short audio clips instead of one long one).
    streaming_pipeline: bool = False
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    session_briefing: SessionBriefingConfig = field(default_factory=SessionBriefingConfig)


@dataclass
class SkillsConfig:
    """RAG-over-skills: hide the long tail of skills from the CC/Codex startup
    prompt and surface the right one per-turn via embedding retrieval, injected
    by the UserPromptSubmit hook. Opt-in (`enabled=false`) so existing TTS-only
    users pull none of the embedding stack.

    Needs the `skills` extra (fastembed). The daemon degrades to a no-op if the
    extra is absent, so the hook simply injects nothing.
    """
    enabled: bool = False
    # Cross-lingual model; see skills/embedder.py for why this default.
    model_name: str = "jinaai/jina-embeddings-v2-base-zh"
    # Persistent model cache — fastembed otherwise drops it in a temp dir that
    # the OS can purge, forcing a slow re-download.
    cache_dir: str = "~/.jarvis/skills/models"
    # Where catalog.json + vectors.npy live.
    index_dir: str = "~/.jarvis/skills"
    top_k: int = 5
    # Hybrid-score tiers (cosine + lexical boost): >= high injects the skill
    # body; >= med offers a menu. Tuned for jina-v2-base-zh on Chinese prompts,
    # where correct matches land ~0.28-0.86, clear mis-ranks <=0.27, pure noise
    # <0.15. high=0.42 body-injects confident hits; med=0.28 still surfaces a
    # menu for short queries (e.g. "做个落地页" tops out ~0.29) while staying
    # ~3x above the noise floor.
    high_threshold: float = 0.42
    med_threshold: float = 0.28
    max_skills: int = 2
    max_body_chars: int = 6000
    total_char_budget: int = 9000
    # Hook-side socket round-trip budget. The prompt must never stall on us, so
    # a miss/timeout injects nothing.
    query_timeout_ms: int = 400


@dataclass
class McpConfig:
    """MCP intent routing: match user prompts against a registry of known MCP
    servers and inject connection instructions for the best match.  Shares the
    embedding model with SkillsConfig.  Opt-in (``enabled=false``)."""

    enabled: bool = False
    registry_path: str = "~/.jarvis/mcp/registry.json"
    index_dir: str = "~/.jarvis/mcp"
    top_k: int = 5
    high_threshold: float = 0.35
    med_threshold: float = 0.22


@dataclass
class WebhookConfig:
    """Optional remote push of the spoken line to a webhook (Bark / ntfy /
    Slack / Discord / generic POST), so a phone/IM surfaces notifications when
    the user is away. Opt-in (``enabled=false``) — no behavior change out of
    the box. The daemon fires this fail-soft and non-blocking: a webhook error
    never touches local audio. See notify/webhook.py for the payload shape.
    """
    enabled: bool = False
    url: str = ""
    # Payload shape. "generic" (default) keeps today's flat JSON object —
    # back-compat for Slack/ntfy/anything. "bark" emits Bark's native
    # title/body/group/level fields so iOS (and the mirrored Apple Watch
    # notification) renders a proper title + per-project grouping instead of
    # a raw JSON blob. See notify/webhook.py for both shapes.
    format: str = "generic"
    # Static headers sent on every request (e.g. ntfy's `Title`, a content
    # type, etc). Auth tokens belong in `auth_env`, not here.
    headers: dict[str, str] = field(default_factory=dict)
    # Auth header injected from an env var so the token stays out of
    # config.toml: `auth_header` is the header name (e.g. "Authorization"),
    # `auth_env` the env var holding its value. Unset env var = header omitted.
    auth_header: str = ""
    auth_env: str = ""
    # Optional allowlist of notification types to push. Empty = push all that
    # reach the webhook (which are already filtered by behavior.events first).
    events: list[str] = field(default_factory=list)
    timeout_seconds: float = 5.0


@dataclass
class RemoteConfig:
    """ntfy-based actionable approvals: outbound pushes carry Approve/Deny
    buttons whose taps POST a decision to a second secret topic that the
    daemon subscribes to — so the phone/watch never needs inbound network
    access to the Mac. Opt-in (``enabled=false``). Both topic names act as
    bearer secrets: anyone who knows them can read pushes / inject decisions,
    so use long random strings (or self-host ntfy). See notify/remote.py.
    """
    enabled: bool = False
    ntfy_base: str = "https://ntfy.sh"
    topic_notify: str = ""    # secret topic the phone subscribes to
    topic_reply: str = ""     # secret topic action buttons publish decisions to
    # Which events get actionable pushes (need a decision from the user).
    events: list[str] = field(default_factory=lambda: [
        "permission_prompt", "ask_user_question", "elicitation_dialog",
    ])
    # Optional shell command run on every decision, with JARVIS_SESSION_ID /
    # JARVIS_DECISION / JARVIS_CWD in the env — the pluggable bridge toward
    # actually unblocking a session (e.g. wiring send-input.sh). Empty = skip.
    on_decision_cmd: str = ""
    timeout_seconds: float = 5.0


@dataclass
class PathsConfig:
    socket: str = "~/.jarvis/jarvis.sock"
    log: str = "~/.jarvis/daemon.log"
    missed_log: str = "~/.jarvis/missed.log"
    # Where `jarvis mute` / `events off <type> <duration>` park the
    # current silence, and where the daemon reads it back per event. State,
    # not preference — see mute.py for why it is not a [behavior] field.
    mute_state: str = "~/.jarvis/mute.json"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)


def expanduser(p: str) -> str:
    return str(Path(os.path.expanduser(p)))


def _merge(dst, src: dict) -> None:
    """Shallow-recursive merge: dict src into dataclass dst, mutating dst."""
    for key, val in src.items():
        if not hasattr(dst, key):
            continue
        cur = getattr(dst, key)
        if isinstance(val, dict) and hasattr(cur, "__dataclass_fields__"):
            _merge(cur, val)
        else:
            setattr(dst, key, val)


def load_config(path: str | Path) -> Config:
    """Load TOML config from `path`, fall back to defaults if missing/empty.

    Path-like values under `[paths]` and `[tts.xtts]` get `~` expanded.
    """
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        _merge(cfg, data)

    cfg.paths.socket = expanduser(cfg.paths.socket)
    cfg.paths.log = expanduser(cfg.paths.log)
    cfg.paths.missed_log = expanduser(cfg.paths.missed_log)
    cfg.paths.mute_state = expanduser(cfg.paths.mute_state)
    cfg.tts.xtts.model_dir = expanduser(cfg.tts.xtts.model_dir)
    cfg.tts.xtts.ref_audio_zh = expanduser(cfg.tts.xtts.ref_audio_zh)
    cfg.tts.xtts.ref_audio_en = expanduser(cfg.tts.xtts.ref_audio_en)
    if cfg.tts.xtts.speaker_embedding:
        cfg.tts.xtts.speaker_embedding = expanduser(cfg.tts.xtts.speaker_embedding)
    cfg.tts.cosyvoice.model_dir = expanduser(cfg.tts.cosyvoice.model_dir)
    cfg.tts.cosyvoice.ref_audio_zh = expanduser(cfg.tts.cosyvoice.ref_audio_zh)
    cfg.tts.cosyvoice.ref_audio_en = expanduser(cfg.tts.cosyvoice.ref_audio_en)
    cfg.skills.cache_dir = expanduser(cfg.skills.cache_dir)
    cfg.skills.index_dir = expanduser(cfg.skills.index_dir)
    cfg.mcp.registry_path = expanduser(cfg.mcp.registry_path)
    cfg.mcp.index_dir = expanduser(cfg.mcp.index_dir)
    # Clamp humor_level rather than rejecting — a typo in config shouldn't
    # leave the daemon refusing to start.
    cfg.behavior.humor_level = max(0, min(3, int(cfg.behavior.humor_level)))
    return cfg


DEFAULT_CONFIG_PATH = expanduser("~/.jarvis/config.toml")
