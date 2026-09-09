"""Daemon entrypoint: assembles the full pipeline and runs forever."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from loguru import logger

from ..briefing import WeatherCache, compose_briefing, detect_city
from ..config import DEFAULT_CONFIG_PATH, Config, load_config
from ..mute import describe as describe_mute
from ..mute import load as load_mute
from ..mute import muted_reason
from ..notify import remote as remote_notify
from ..notify import webhook as webhook_notify
from ..phrase.language import detect_for
from ..phrase.redact import speakable
from ..phrase.providers.anthropic import AnthropicProvider
from ..phrase.providers.base import PhraseProvider
from ..phrase.providers.deepseek import DeepSeekProvider
from ..phrase.providers.ollama import OllamaProvider
from ..phrase.providers.openai import OpenAIProvider
from ..phrase.providers.siliconflow import SiliconFlowProvider
from ..phrase.providers.zhipu import ZhipuProvider
from ..phrase.router import PhraseRouter
from ..player import (
    Cancellable,
    PCMPlayer,
    StreamPlayer,
    on_pcm_wedge,
    open_pcm_sink,
    pcm_sink_unusable,
    play,
    play_stream,
)
from ..tts.engine import TTSEngine
from ..tts.factory import HEAVY_PROVIDERS, make_provider
from ..tts.providers.base import TTSProvider
from ..tts.providers.say import SayProvider
from ..tts.worker_client import WorkerProvider
from ..types import Event, Lang, emotion_for
from .dedup import DedupWindow
from .health import HealthServer
from .listener import serve_unix_socket
from .queue import BoundedEventQueue


#: How long this process must have been up before a PortAudio wedge is allowed
#: to restart it (see Daemon._self_heal_if_wedged). Any real wedge so far took
#: hours of healthy playback to develop; anything faster is more likely a
#: device that cannot be opened at all, and restarting into that is a loop.
_SELF_HEAL_MIN_UPTIME_SECONDS = 600.0

#: How long the self-heal exit waits for detached webhook/ntfy pushes to
#: finish before killing the process. Long enough for a normal ntfy POST,
#: short enough that a hung one cannot hold the recovery hostage.
_SELF_HEAL_PUSH_DRAIN_SECONDS = 5.0


def _make_phrase_provider(name: str, cfg: Config) -> PhraseProvider | None:
    factories = {
        "deepseek": lambda: DeepSeekProvider(cfg.llm.deepseek),
        "anthropic": lambda: AnthropicProvider(cfg.llm.anthropic),
        "openai": lambda: OpenAIProvider(cfg.llm.openai),
        "ollama": lambda: OllamaProvider(cfg.llm.ollama),
        "zhipu": lambda: ZhipuProvider(cfg.llm.zhipu),
        "siliconflow": lambda: SiliconFlowProvider(cfg.llm.siliconflow),
    }
    factory = factories.get(name)
    if factory is None:
        if name:  # empty string means "no fallback configured", not a typo
            logger.warning("Unknown phrase provider {!r}; skipping in chain", name)
        return None
    return factory()


def _make_tts_provider(
    name: str, cfg: Config, config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TTSProvider | None:
    """Build a provider, putting the memory-hungry ones behind a worker child.

    The wrapper satisfies the same TTSProvider interface, so every caller —
    sink selection, streaming, the synth+afplay fallback — is unaware of which
    process the model actually lives in. See tts.worker for why the heavy ones
    are isolated at all.
    """
    if cfg.tts.worker_process and name in HEAVY_PROVIDERS:
        return WorkerProvider(
            name, config_path, max_syntheses=cfg.tts.worker_max_syntheses,
        )
    return make_provider(name, cfg)


class Daemon:
    def __init__(
        self, cfg: Config, health_port: int = 9527,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.cfg = cfg
        # Where `reload_behavior` re-reads from — the same file `cfg` was
        # loaded from at startup.
        self._config_path = config_path
        # Mute state lives outside config.toml (see mute.py) and is re-read
        # per event: the CLI can mute a running daemon with no round-trip,
        # and a mute set while the daemon was down still holds when it comes
        # back up.
        self._mute_path = cfg.paths.mute_state
        self.queue = BoundedEventQueue(maxsize=cfg.behavior.queue_max_size)
        self.dedup = DedupWindow(window_seconds=cfg.behavior.dedup_window_seconds)
        # Build the fallback chain: prefer the multi-level `fallbacks` list,
        # else wrap the singular `fallback` (back-compat). Unconstructable
        # providers (missing dep/key) are dropped here; a key-less one that
        # slips through just raises at call time and the router skips to the
        # next link.
        fb_names = cfg.llm.fallbacks or (
            [cfg.llm.fallback] if cfg.llm.fallback else []
        )
        fb_providers = [
            p for n in fb_names
            if (p := _make_phrase_provider(n, cfg)) is not None
        ]
        self.router = PhraseRouter(
            primary=_make_phrase_provider(cfg.llm.provider, cfg),
            fallbacks=fb_providers,
            cfg=cfg,
            on_primary_fallback=self._announce_phrase_fallback,
        )
        primary_tts = (
            _make_tts_provider(cfg.tts.provider, cfg, config_path)
            or SayProvider()
        )
        fallback_tts = _make_tts_provider(cfg.tts.fallback, cfg, config_path)
        overrides: dict[str, TTSProvider] = {}
        if cfg.tts.provider_zh and cfg.tts.provider_zh != cfg.tts.provider:
            zh_tts = _make_tts_provider(cfg.tts.provider_zh, cfg, config_path)
            if zh_tts is not None:
                overrides["zh"] = zh_tts
        self.tts = TTSEngine(
            primary=primary_tts, fallback=fallback_tts, overrides=overrides,
        )
        self.health = HealthServer(
            host="127.0.0.1",
            port=health_port,
            state_getter=self._snapshot,
        )
        self._last_text: str | None = None
        # PortAudio self-heal state: the reason recorded by the wedge callback,
        # and whether the worker has already ruled on it. See
        # _self_heal_if_wedged.
        self._started_at = time.monotonic()
        self._wedge_reason: str | None = None
        self._self_heal_checked = False
        on_pcm_wedge(self._note_pcm_wedge)
        # Keep strong refs to in-flight webhook tasks so they aren't garbage
        # collected mid-flight (asyncio only holds weak refs to tasks).
        self._webhook_tasks: set[asyncio.Task] = set()
        # The in-flight playback handle: an afplay/ffplay Process or a
        # PCMPlayer — anything cancel_session can kill() (see Cancellable).
        self._current_proc: Cancellable | None = None
        self._current_session_id: str | None = None
        # False while the in-flight event's type is cancel-exempt (see
        # behavior.cancel_exempt_events) — cancel_session then leaves the
        # playback alone.
        self._current_cancellable: bool = True
        self._cancelled_sessions: set[str] = set()
        # Phrase prefetch: while event N plays, the worker phrases the event
        # at the queue head so N+1 skips its LLM round-trip. Keyed by event
        # IDENTITY (the queue hands back the same object), so a cached entry
        # for a dropped/cancelled event simply never matches. No locking:
        # everything here runs on the one event loop.
        self._prefetched: tuple[Event, str, Lang] | None = None
        # Whether the event currently being dispatched got its text from the
        # prefetch cache — set by _worker right before dispatch, read by
        # _process_one for the TIMING log. An instance attr (not a kwarg)
        # so _process_one's signature — and every test that patches it —
        # stays unchanged.
        self._dispatch_prefetched: bool = False
        # Throttle Jarvis's voice alert when the local phrase provider
        # quietly slips onto the cloud fallback. 5 min between announcements
        # is enough for the user to notice without spamming during an
        # extended ollama outage.
        self._last_phrase_fallback_alert_at: float = 0.0
        self._phrase_fallback_alert_throttle_s: float = 300.0
        # Session briefing: shared weather cache + per-machine throttle
        # (user-configurable; default 0 = announce every session_start).
        self._weather_cache = WeatherCache(
            ttl_seconds=cfg.behavior.session_briefing.weather_ttl_seconds,
            stale_max_age_seconds=(
                cfg.behavior.session_briefing.weather_stale_max_age_seconds
            ),
        )
        self._last_briefing_at: float = 0.0
        # Shared embedding model for RAG retrieval (skills + MCP intent
        # routing).  Created once; both services share the instance to avoid
        # loading the ~640 MB ONNX model twice.
        self.skills: object | None = None
        self.mcp: object | None = None
        _embedder = None
        if cfg.skills.enabled or cfg.mcp.enabled:
            try:
                from ..retrieval.embedder import Embedder

                _embedder = Embedder(
                    cfg.skills.model_name, cache_dir=cfg.skills.cache_dir,
                )
            except ImportError as exc:
                logger.warning(
                    "retrieval stack unavailable ({}); "
                    "install jarvis[skills]", exc,
                )
        if cfg.skills.enabled and _embedder is not None:
            try:
                from ..skills.service import SkillService

                self.skills = SkillService(cfg.skills, _embedder)
                logger.info("skills: retrieval enabled")
            except ImportError as exc:
                logger.warning(
                    "skills deps missing ({}); install jarvis[skills]", exc,
                )
        if cfg.mcp.enabled and _embedder is not None:
            try:
                from ..mcp.service import McpService

                self.mcp = McpService(cfg.mcp, _embedder)
                logger.info("mcp: intent routing enabled")
            except ImportError as exc:
                logger.warning("mcp deps missing ({})", exc)

    def _snapshot(self) -> dict:
        return {
            "queue_size": self.queue.size,
            "queue_capacity": self.queue.maxsize,
            "dropped": self.queue.dropped_count,
            "last_text": self._last_text,
            # Empty unless something is muted. A silent Jarvis is otherwise
            # indistinguishable from a broken one, and this repo has spent
            # days on exactly that confusion — `status` must say so.
            "muted": describe_mute(load_mute(self._mute_path)),
        }

    async def _announce_phrase_fallback(self, primary_name: str) -> None:
        """Wired into PhraseRouter; fires when the primary LLM (typically
        local ollama) fails and the cloud fallback takes over. Throttle to
        one alert per `_phrase_fallback_alert_throttle_s`, then enqueue a
        pre-baked Jarvis line so the user audibly notices."""
        now = time.monotonic()
        if now - self._last_phrase_fallback_alert_at < self._phrase_fallback_alert_throttle_s:
            return
        self._last_phrase_fallback_alert_at = now
        text = (
            f"Sir, the local language model {primary_name} appears unreachable. "
            "I am falling back to the cloud."
        )
        logger.warning(
            "Phrase primary ({}) unreachable — alerting via voice", primary_name,
        )
        alert = Event(
            notification_type="idle_prompt",
            tool_name=None, tool_input={},
            text=text, lang="en",
        )
        await self.queue.put_or_drop(alert)

    async def cancel_session(self, session_id: str) -> None:
        """Cancel any in-flight audio for `session_id` and drop its queued
        events. Types in behavior.cancel_exempt_events (failure notices by
        default) are immune throughout — they stay queued, and an in-flight
        one plays out; the workers' own cancelled-checks skip them too."""
        exempt = self.cfg.behavior.cancel_exempt_events
        self._cancelled_sessions.add(session_id)
        # No need to touch `_prefetched` here: dropping the queued event means
        # the worker's identity check never matches it, so the cached phrase
        # is discarded on the next dequeue.
        self.queue.drop_matching(
            lambda e: e.session_id == session_id
            and e.notification_type not in exempt
        )
        proc = self._current_proc
        if (
            proc is not None
            and self._current_session_id == session_id
            and self._current_cancellable
        ):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _on_event(self, event: Event) -> None:
        dkey = event.dedup_key()
        # Mute outranks every other gate, including pre-baked `say --text`:
        # the user asked for silence, so silence is what they get.
        reason = muted_reason(load_mute(self._mute_path), event.notification_type)
        if reason:
            logger.debug("DROP {} key={}", reason, dkey)
            return
        if event.notification_type not in self.cfg.behavior.events:
            logger.debug("DROP filtered-by-events-allowlist key={}", dkey)
            return
        # session_start has its own gate (enabled flag + per-machine floor)
        # in addition to the generic dedup. Drop early so we don't waste a
        # queue slot when the user opens five tabs in a minute.
        if event.notification_type == "session_start":
            sb = self.cfg.behavior.session_briefing
            if not sb.enabled:
                logger.debug("DROP session_briefing.enabled=false key={}", dkey)
                return
            now = time.monotonic()
            if sb.min_interval_seconds > 0 and \
                    now - self._last_briefing_at < sb.min_interval_seconds:
                logger.debug(
                    "DROP session_start throttled key={} last={:.0f}s ago",
                    dkey, now - self._last_briefing_at,
                )
                return
            self._last_briefing_at = now
        if self.dedup.is_duplicate(event):
            logger.debug("DROP dedup-hit key={}", dkey)
            return
        logger.debug("QUEUE key={}", dkey)
        await self.queue.put_or_drop(event)

    async def _verify(self, text: str, candidates: list, *, noun: str):
        """LLM-classify intent candidates (skills or MCP). Returns a
        ``VerifyResult`` with status confirmed/none/unclear. On any error the
        status is ``unclear`` — we ask the user rather than fabricate a confirm
        we couldn't actually make."""
        from ..retrieval.verifier import UNCLEAR, VerifyResult, verify_candidates

        try:
            return await verify_candidates(
                text, candidates, self.cfg.llm.ollama, noun=noun,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("{}: verification skipped ({})", noun, exc)
            return VerifyResult(UNCLEAR)

    def _reload_behavior(self) -> dict:
        """Re-read config.toml and swap the [behavior] section in place.

        The Config object is shared by reference with the phrase router (it
        reads cfg.behavior.* per call), so replacing `.behavior` takes effect
        on the very next event — no TTS reload, no daemon restart. Only
        [behavior] is refreshed; LLM/TTS/provider changes still need a
        restart, which is why this doesn't swap the whole Config."""
        try:
            fresh = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001 — a bad config must not kill the daemon
            logger.warning("reload_behavior failed: {}", exc)
            return {"ok": False, "error": str(exc)}
        self.cfg.behavior = fresh.behavior
        # The weather cache is a separate object holding copies of its two
        # knobs, so it needs an explicit push — without it those two would be
        # the odd ones out in an otherwise live-reloadable [behavior] section.
        briefing = fresh.behavior.session_briefing
        self._weather_cache.ttl = briefing.weather_ttl_seconds
        self._weather_cache.stale_max_age = briefing.weather_stale_max_age_seconds
        # Queue/dedup sizes are constructor-fixed; note them so a user who
        # edits those fields learns why nothing changed.
        logger.info(
            "behavior reloaded: humor_level={} address_en={!r} "
            "(queue/dedup sizes still need a restart)",
            fresh.behavior.humor_level, fresh.behavior.address_en,
        )
        return {"ok": True, "humor_level": fresh.behavior.humor_level}

    def _apply_mute(self) -> dict:
        """Act on a mute the CLI just wrote.

        The state file alone is enough for events that have not arrived yet —
        `_on_event` re-reads it every time. This exists for the line already
        in the air: muting is usually a reaction to hearing something, so a
        fresh global mute cuts the current utterance and empties the queue
        instead of letting the backlog finish first."""
        active = describe_mute(load_mute(self._mute_path))
        if "all" not in active:
            logger.info("mute state reloaded: {}", active or "nothing muted")
            return {"ok": True, "muted": active}
        dropped = self.queue.drop_matching(lambda _e: True)
        proc = self._current_proc
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        logger.info(
            "muted {} — cut the line in flight, dropped {} queued",
            active["all"], dropped,
        )
        return {"ok": True, "muted": active, "dropped": dropped}

    async def _on_query(self, payload: dict) -> dict:
        """Request/response handler for the hook's skill_query / skill_refresh,
        the CLI's reload_behavior, and MCP intent matching with LLM
        verification. Runs CPU-bound retrieval off the event loop. Never
        raises."""
        if payload.get("command") == "reload_behavior":
            return self._reload_behavior()
        if payload.get("command") == "reload_mute":
            return self._apply_mute()
        if payload.get("command") == "skill_refresh":
            if self.skills is not None:
                await asyncio.to_thread(self.skills.refresh)
            if self.mcp is not None:
                await asyncio.to_thread(self.mcp.refresh)
            return {"ok": True}

        text = payload.get("text") or ""
        sid = payload.get("session_id")

        # Skills and MCP are independent pipelines (separate gate -> verify ->
        # build). Run them concurrently so a prompt that hits both gates costs
        # one LLM round-trip of latency, not two.
        (skill_context, skill_mode, skill_matches), mcp_context = await asyncio.gather(
            self._skill_context(text, sid),
            self._mcp_context(text),
        )

        parts = [p for p in (skill_context, mcp_context) if p]
        merged = "\n\n---\n\n".join(parts) if parts else None

        return {
            "context": merged,
            "mode": skill_mode,
            "matches": skill_matches,
        }

    async def _skill_context(
        self, text: str, sid: str | None
    ) -> tuple[str | None, str, list]:
        """Skills pipeline: gate -> LLM-classify -> build. Confirmed intent
        body/menu-injects; "unclear" injects a clarify note (never a command);
        "none" injects nothing. Returns (context, mode, telemetry matches)."""
        from ..retrieval.verifier import CONFIRMED, UNCLEAR

        if self.skills is None:
            return None, "none", []
        gate = await asyncio.to_thread(self.skills.gate, text)
        matches = gate.get("matches", [])
        candidates = gate.get("candidates", [])
        if not candidates:
            return None, "none", matches
        vr = await self._verify(text, candidates, noun="skill")
        if vr.status == CONFIRMED and vr.matches:
            built = await asyncio.to_thread(
                self.skills.build, vr.matches, session_id=sid, query=text,
            )
        elif vr.status == UNCLEAR:
            built = await asyncio.to_thread(self.skills.clarify, candidates)
        else:
            return None, "none", matches
        return built.get("context"), built.get("mode", "none"), matches

    async def _mcp_context(self, text: str) -> str | None:
        """MCP pipeline: gate -> LLM-classify -> connect/clarify injection.
        Mirrors _skill_context but returns just the context string."""
        from ..mcp.service import _build_clarify, _build_injection
        from ..retrieval.verifier import CONFIRMED, UNCLEAR

        if self.mcp is None:
            return None
        result = await asyncio.to_thread(self.mcp.query, text)
        candidates = result.get("candidates", [])
        if not candidates:
            return None
        vr = await self._verify(text, candidates, noun="tool server")
        if vr.status == CONFIRMED and vr.matches:
            return _build_injection(
                vr.matches,
                high_threshold=self.cfg.mcp.high_threshold,
                med_threshold=self.cfg.mcp.med_threshold,
                query=text,
            )
        if vr.status == UNCLEAR:
            return _build_clarify(candidates)
        return None

    def _wants_streaming(self, event: Event) -> bool:
        """Route to the streaming pipeline when the config flag is set AND the
        event is an LLM-phrased one (not pre-baked text, not session_start
        which has its own compose path)."""
        return (
            self.cfg.behavior.streaming_pipeline
            and event.text is None
            and event.notification_type != "session_start"
        )

    def _is_stale(self, event: Event) -> bool:
        """A backlog burst (long playback blocking the queue) can dequeue a
        notification long after the user already acted on it — speaking it
        then is noise. Pre-baked text and session_start briefings are exempt
        (those should speak whenever they surface), as are events without a
        received_at stamp (0.0 — synthetic/test events carry no timestamp)."""
        max_age = self.cfg.behavior.stale_event_max_age_seconds
        if (
            max_age <= 0
            or event.text is not None
            or event.notification_type == "session_start"
            or not event.received_at
        ):
            return False
        return time.time() - event.received_at > max_age

    async def _prefetch_next(self) -> None:
        """Phrase the event at the queue head while the current one plays.

        Zero-overhead when there's nothing to do: an empty queue makes peek()
        return None and we exit immediately, so the common single-event case
        is untouched. Pre-baked text and session_start have no LLM round-trip
        to hide (the briefing composes inline), and a stale head is about to
        be dropped at dequeue anyway — skip all of those. Best-effort: any
        failure just means the worker phrases the event itself, as before.
        """
        try:
            nxt = self.queue.peek()
            if (
                nxt is None
                or nxt.text is not None
                or nxt.notification_type == "session_start"
                or self._is_stale(nxt)
            ):
                return
            lang = (
                detect_for(nxt.cwd)
                if self.cfg.behavior.voice_language == "auto"
                else self.cfg.behavior.voice_language  # type: ignore[assignment]
            )
            emotion = emotion_for(nxt.notification_type)
            text = await self.router.phrase(nxt, lang=lang, emotion=emotion)
            self._prefetched = (nxt, text, lang)
        except Exception as exc:  # noqa: BLE001
            logger.debug("prefetch failed ({})", exc)

    async def _worker(self) -> None:
        while True:
            event = await self.queue.get()
            # Stale check on the ORIGINAL event, before any prefetch
            # consumption — a prefetched phrase for a stale event must not
            # resurrect it via the pre-baked exemption in _is_stale.
            if self._is_stale(event):
                logger.debug(
                    "DROP stale age={:.0f}s key={}",
                    time.time() - event.received_at, event.dedup_key(),
                )
                continue
            # Consume the prefetch cache. Identity check: the queue hands
            # back the same object it stored, so a hit means THIS event was
            # phrased during the previous playback. A miss means the cached
            # event was dropped (cancel_session) or otherwise superseded —
            # the cache is garbage either way, so always clear it.
            prefetched = (
                self._prefetched is not None and self._prefetched[0] is event
            )
            if prefetched:
                _, text, lang = self._prefetched
                event = replace(event, text=text, lang=lang)
            self._prefetched = None
            # Kick off phrasing for the event now at the queue head so its
            # LLM round-trip overlaps this event's playback. Routing
            # consequence: a prefetched event carries `text`, so
            # _wants_streaming returns False and it flows through the batch
            # path's pre-baked branch — exactly what we want under backlog
            # (full text ready before playback starts beats intra-event
            # streaming).
            prefetch_task = asyncio.create_task(self._prefetch_next())
            self._dispatch_prefetched = prefetched
            if self._wants_streaming(event):
                await self._process_one_streaming(event)
            else:
                await self._process_one(event)
            # Usually already done — playback outlasts phrasing. Awaiting
            # keeps the task lifecycle clean (no orphans, errors surface).
            await prefetch_task
            # Idle point: the queue is drained and nothing is playing, so
            # this is where housekeeping that costs a pause belongs.
            await self.tts.recycle_worn_workers()
            await self._self_heal_if_wedged()

    async def _process_one(self, event: Event) -> None:
        sid = event.session_id
        started = time.monotonic()
        # Cancel-exempt events ignore every cancelled-check below: a cancel
        # that lands while one is phrasing/synthesizing/playing must not
        # silence it (behavior.cancel_exempt_events).
        cancellable = (
            event.notification_type not in self.cfg.behavior.cancel_exempt_events
        )
        # Capture-and-reset the worker's prefetch flag so a direct call
        # (tests, future callers) never inherits a stale True.
        prefetched = self._dispatch_prefetched
        self._dispatch_prefetched = False
        if sid and sid in self._cancelled_sessions:
            # Stale cancel signal preceded the event; clear and play normally.
            self._cancelled_sessions.discard(sid)

        def _register(proc: Cancellable) -> None:
            self._current_proc = proc
            self._current_session_id = sid
            self._current_cancellable = cancellable

        # Derive the emotion for this event once; it flows into both the
        # phrase prompt (shaping the LLM's written text) and the TTS
        # provider (ElevenLabs voice_settings presets, etc.).
        emotion = emotion_for(event.notification_type)

        try:
            phrase_started = time.monotonic()
            if event.text is not None:
                # Caller pre-baked the phrase; skip the LLM entirely.
                text = event.text
                lang = event.lang or "en"
            elif event.notification_type == "session_start":
                # Compose Jarvis briefing inline. We hand the primary
                # phrase provider (Ollama by default) in so the line is
                # freshly phrased each session — falls back to a
                # rotating template if the LLM is unreachable. Weather
                # fetch is best-effort with a TTL cache.
                text, lang = await compose_briefing(
                    self.cfg.behavior.session_briefing,
                    cache=self._weather_cache,
                    llm=self.router.primary,
                    humor_level=self.cfg.behavior.humor_level,
                    address=self.cfg.behavior.address_en,
                )
            else:
                lang = (
                    detect_for(event.cwd)
                    if self.cfg.behavior.voice_language == "auto"
                    else self.cfg.behavior.voice_language  # type: ignore[assignment]
                )
                text = await self.router.phrase(
                    event, lang=lang, emotion=emotion,
                )
            # Second speakability gate (the first scrubs the LLM's input):
            # pre-baked text never saw the router, and the LLM can echo an
            # ID from its own context — never hand TTS a UUID to spell out.
            text = speakable(text)
            # Stage timing (DEBUG): phrase covers the LLM/briefing cost above
            # (~0 for pre-baked/prefetched text); total spans through playback.
            phrase_ms = (time.monotonic() - phrase_started) * 1000

            def _log_timing() -> None:
                logger.debug(
                    "TIMING type={} phrase_ms={:.0f} total_ms={:.0f} "
                    "prefetched={}",
                    event.notification_type, phrase_ms,
                    (time.monotonic() - started) * 1000, prefetched,
                )

            self._last_text = text
            # Remote push (opt-in). Fire-and-forget as a detached task so the
            # network call runs CONCURRENTLY with audio playback below and can
            # never delay or block TTS. `text` is the final spoken line — for
            # LLM-phrased events it was built from an already-redacted summary
            # (notify.webhook itself is fail-soft and never raises).
            if self.cfg.webhook.enabled:
                self._fire_webhook(event, text)
            # Actionable remote push (ntfy Approve/Deny buttons) rides the
            # same detached-task pattern: concurrent with playback, fail-soft.
            if (
                self.cfg.remote.enabled
                and event.notification_type in self.cfg.remote.events
            ):
                self._fire_remote(event, text)
            # Phrasing/briefing above can take seconds (LLM round-trip). If the
            # user already acted on this session in that window, the line is
            # stale — drop it before any play proc starts rather than speak a
            # step too late. No proc is registered yet, so the cancel that just
            # landed had nothing to kill; this guard is what honours it.
            if cancellable and sid and sid in self._cancelled_sessions:
                logger.debug("DROP cancelled-before-play sid={}", sid)
                return
            # Whole line, not a prefix: this is the only record of what was
            # said, and an audio complaint ("it babbled at the end") can only
            # be matched to its text if the end is here too.
            logger.debug(
                "PLAY type={} sid={} emotion={} text={!r}",
                event.notification_type, sid, emotion, text,
            )
            if await self._try_stream(
                text, lang, event.voice_id,
                on_spawn=_register, session_id=sid,
                emotion=emotion, cancellable=cancellable,
            ):
                _log_timing()
                return
            # Skip synth fallback if a cancel arrived between stream attempt
            # and now — otherwise the same line gets re-synthesized + replayed.
            if cancellable and sid and sid in self._cancelled_sessions:
                return
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            await self.tts.synthesize(
                text, lang, out_path,
                voice_id=event.voice_id, emotion=emotion,
            )
            # synthesize() is the long pole for non-streaming providers
            # (CosyVoice ~seconds) and registers no play proc, so a cancel that
            # lands mid-synth kills nothing. Re-check before playing so the
            # finished file is discarded instead of surfacing as stale audio.
            if cancellable and sid and sid in self._cancelled_sessions:
                logger.debug("DROP cancelled-after-synth sid={}", sid)
                try:
                    out_path.unlink()
                except OSError:
                    pass
                return
            try:
                await play(out_path, on_spawn=_register)
            finally:
                try:
                    out_path.unlink()
                except OSError:
                    pass
            _log_timing()
        except Exception as exc:
            if cancellable and sid and sid in self._cancelled_sessions:
                logger.debug("playback cancelled for session {}", sid)
            else:
                logger.exception("worker failed: {}", exc)
        finally:
            self._current_proc = None
            self._current_session_id = None
            self._current_cancellable = True
            if sid:
                self._cancelled_sessions.discard(sid)

    async def _process_one_streaming(self, event: Event) -> None:
        """Streaming pipeline: overlap LLM → sentence chunking → TTS → play.

        For each sentence the LLM produces, we synthesize and play it
        immediately — so the first sentence is audible while the LLM is still
        generating the rest. All streamed sentences feed ONE audio sink
        (in-process PCM or ffplay — see _spawn_stream_sink; spawned lazily on
        the first sentence) so consecutive sentences play gaplessly instead
        of paying a device-open/process-spawn pause between each.
        """
        sid = event.session_id
        started = time.monotonic()
        # Mirror _process_one: cancel-exempt events ride out every
        # cancelled-check in this pipeline.
        cancellable = (
            event.notification_type not in self.cfg.behavior.cancel_exempt_events
        )
        # Stage timing (DEBUG): first successful feed is our proxy for first
        # audible audio — the whole point of this pipeline is shrinking it.
        first_feed_at: float | None = None
        if sid and sid in self._cancelled_sessions:
            self._cancelled_sessions.discard(sid)

        def _register(proc: Cancellable) -> None:
            self._current_proc = proc
            self._current_session_id = sid
            self._current_cancellable = cancellable

        # Mirror _process_one: derive the emotion once so it shapes both the
        # phrase prompt (the LLM's written tone) and every per-sentence TTS
        # call — streamed lines must not sound flatter than batch ones.
        emotion = emotion_for(event.notification_type)

        session: PCMPlayer | StreamPlayer | None = None
        try:
            lang = (
                detect_for(event.cwd)
                if self.cfg.behavior.voice_language == "auto"
                else self.cfg.behavior.voice_language  # type: ignore[assignment]
            )
            spoken_parts: list[str] = []
            async for sentence in self.router.phrase_stream(
                event, lang=lang, emotion=emotion,
            ):
                if cancellable and sid and sid in self._cancelled_sessions:
                    logger.debug("DROP cancelled-mid-stream sid={}", sid)
                    break
                # Second speakability gate — see _process_one.
                sentence = speakable(sentence)
                spoken_parts.append(sentence)
                logger.debug(
                    "STREAM-PLAY type={} sid={} emotion={} chunk={!r}",
                    event.notification_type, sid, emotion, sentence,
                )
                # Try streaming TTS first (e.g. XTTS/ElevenLabs), feeding the
                # shared ffplay pipe; fall back to file-based synth+play for
                # this sentence on failure. Lang-routed: a zh override
                # provider (tts.provider_zh) may not stream — those sentences
                # take the file-synth path below.
                primary = self.tts.primary_for(lang)
                if primary.supports_streaming and not self._streams_into_ffplay(
                    primary,
                ):
                    try:
                        if session is None:
                            session = await self._spawn_stream_sink(
                                primary, on_spawn=_register,
                            )
                        await session.feed(primary.stream(
                            sentence, lang,
                            voice_id=event.voice_id, emotion=emotion,
                        ))
                        if first_feed_at is None:
                            first_feed_at = time.monotonic()
                        continue
                    except Exception as exc:
                        if cancellable and sid and sid in self._cancelled_sessions:
                            # The worker killed ffplay to cancel playback —
                            # the broken pipe is the kill, not a TTS failure.
                            # Stop speaking rather than re-synthesize the
                            # line we just silenced.
                            logger.debug("stream cancelled for session {}", sid)
                            break
                        logger.warning(
                            "Streaming TTS failed, falling back to synth: {}",
                            exc,
                        )
                        # The shared pipe is in an unknown state; kill it.
                        # Later sentences may retry streaming — a fresh
                        # session gets spawned on the next attempt.
                        if session is not None:
                            await session.abort()
                            session = None
                if cancellable and sid and sid in self._cancelled_sessions:
                    break
                # Reaching the file path with a live sink still open means the
                # streaming decision flipped mid-utterance (the PCM sink went
                # away while sentence N-1 was in flight, so _streams_into_ffplay
                # now says no). Play out what that sink already holds before
                # starting afplay: otherwise the two overlap into a double
                # voice, and _register below would rebind _current_proc to the
                # afplay handle, leaving the sink uncancellable.
                if session is not None:
                    try:
                        await session.close()
                    except Exception as exc:  # noqa: BLE001 — cleanup path
                        logger.debug("stream session close failed: {}", exc)
                    session = None
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = Path(tmp.name)
                await self.tts.synthesize(
                    sentence, lang, out_path,
                    voice_id=event.voice_id, emotion=emotion,
                )
                if cancellable and sid and sid in self._cancelled_sessions:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
                    break
                try:
                    await play(out_path, on_spawn=_register)
                finally:
                    try:
                        out_path.unlink()
                    except OSError:
                        pass

            # Utterance complete (or cancelled): close the sink so it drains
            # its remaining buffer (ffplay: stdin EOF; PCM: StopStream). A
            # close error after a cancel is just the kill we issued
            # concluding — not a fault.
            if session is not None:
                try:
                    await session.close()
                except Exception as exc:
                    if cancellable and sid and sid in self._cancelled_sessions:
                        logger.debug(
                            "stream session ended by cancel for session {}",
                            sid,
                        )
                    else:
                        logger.warning(
                            "stream session close failed: {}", exc,
                        )
                finally:
                    session = None

            full_text = " ".join(spoken_parts) if spoken_parts else ""
            self._last_text = full_text
            # first_feed_ms=-1 means no sentence ever streamed (every one
            # took the file-synth fallback, or the LLM produced nothing).
            logger.debug(
                "TIMING-STREAM type={} first_feed_ms={:.0f} total_ms={:.0f} "
                "sentences={}",
                event.notification_type,
                (first_feed_at - started) * 1000
                if first_feed_at is not None else -1.0,
                (time.monotonic() - started) * 1000,
                len(spoken_parts),
            )
            if full_text and self.cfg.webhook.enabled:
                self._fire_webhook(event, full_text)
            if (
                full_text
                and self.cfg.remote.enabled
                and event.notification_type in self.cfg.remote.events
            ):
                self._fire_remote(event, full_text)
        except Exception as exc:
            if cancellable and sid and sid in self._cancelled_sessions:
                logger.debug("streaming playback cancelled for session {}", sid)
            else:
                logger.exception("streaming worker failed: {}", exc)
        finally:
            # Belt and braces: if an exception bailed us out with the shared
            # ffplay still live, kill it — never leave an orphan process
            # holding the pipe (abort() reaps and never raises).
            if session is not None:
                await session.abort()
            self._current_proc = None
            self._current_session_id = None
            self._current_cancellable = True
            if sid:
                self._cancelled_sessions.discard(sid)

    def _fire_webhook(self, event: Event, text: str) -> None:
        """Schedule the webhook POST as a detached task and track it so it
        isn't GC'd before completion. notify() is fail-soft (never raises),
        so the task needs no error handling of its own."""
        task = asyncio.ensure_future(
            webhook_notify.notify(self.cfg.webhook, event, text)
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    def _fire_remote(self, event: Event, text: str) -> None:
        """Schedule the ntfy actionable push as a detached task, sharing the
        webhook task-tracking set (same lifecycle need: keep a strong ref so
        the task isn't GC'd mid-flight). push_actionable is fail-soft."""
        task = asyncio.ensure_future(
            remote_notify.push_actionable(self.cfg.remote, event, text)
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    async def _remote_listener(self) -> None:
        """Lifetime task: subscribe to the ntfy reply topic and route each
        Approve/Deny decision into _on_remote_decision. listen_replies
        reconnects forever and only exits on cancellation (daemon shutdown)."""
        await remote_notify.listen_replies(
            self.cfg.remote, self._on_remote_decision,
        )

    async def _on_remote_decision(self, decision: str, sid: str) -> None:
        """A decision arrived from the phone/watch: acknowledge audibly and
        (optionally) hand it to the configured bridge command. Fail-soft —
        a broken bridge must never take the listener down."""
        logger.info("remote decision: {} sid={}", decision, sid)
        text = (
            "Sir, remote approval received."
            if decision == "approve"
            else "Understood, sir — request denied remotely."
        )
        # Pre-baked spoken ack (same pattern as _announce_phrase_fallback):
        # no LLM round-trip, session_id=None so a pending cancel for the
        # decided session can't silence the acknowledgement itself.
        ack = Event(
            notification_type="idle_prompt",
            tool_name=None, tool_input={},
            text=text, lang="en",
        )
        await self.queue.put_or_drop(ack)
        cmd = self.cfg.remote.on_decision_cmd
        if not cmd:
            return
        env = dict(os.environ)
        env.update({
            "JARVIS_SESSION_ID": sid,
            "JARVIS_DECISION": decision,
            # Reserved for a future push→decision correlation that carries
            # the project dir; bridges can rely on the var existing today.
            "JARVIS_CWD": "",
        })
        try:
            # Fire-and-forget: we never await completion or read output —
            # the bridge owns its own logging/failure story.
            await asyncio.create_subprocess_shell(
                cmd,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001 — never raise into listener
            logger.warning(
                "remote: on_decision_cmd spawn failed ({}): {}",
                type(exc).__name__, exc,
            )

    async def _spawn_stream_sink(
        self,
        primary: TTSProvider,
        *,
        on_spawn: Callable[[Cancellable], None] | None = None,
    ) -> PCMPlayer | StreamPlayer:
        """The one sink-selection point for both streaming call sites: raw-PCM
        providers (stream_pcm set) get the in-process sounddevice sink — which
        itself falls back to ffplay when sounddevice is unavailable — while
        container-format providers (MP3 from ElevenLabs) keep ffplay, the only
        decoder we have for those bytes.

        The PCM sink is additionally gated behind `tts.pcm_playback` (on by
        default since 2026-08-25) — see TTSConfig. Callers must check
        `_streams_into_ffplay` first: a raw-PCM provider that would land on
        ffplay must not stream at all. `open_pcm_sink` RAISES rather than
        degrading to ffplay, and both callers turn that into the same
        synth+afplay fallback, so a device that fails to open never
        silently lands us back on the pipe that truncates.
        """
        if primary.stream_pcm is not None and self.cfg.tts.pcm_playback:
            rate, channels = primary.stream_pcm
            return await open_pcm_sink(
                rate=rate, channels=channels, on_spawn=on_spawn,
            )
        return await StreamPlayer.spawn(
            input_args=primary.stream_input_args, on_spawn=on_spawn,
        )

    def _note_pcm_wedge(self, reason: str) -> None:
        """`player.on_pcm_wedge` handler. Fires on the event loop, in the
        middle of the utterance that hit the wedge, so it only records —
        the worker decides what to do once the queue is idle."""
        if self._wedge_reason is None:
            self._wedge_reason = reason

    async def _self_heal_if_wedged(self) -> None:
        """Restart the process when PortAudio has wedged and the queue is idle.

        Nothing in-process can undo a wedge: the leaked release thread holds
        CoreAudio's HAL mutex for as long as the process lives. Before
        2026-08-25 the only signal was a single WARNING line, so the daemon
        just stayed degraded until someone noticed by ear — 75 minutes, that
        time. launchd's KeepAlive respawns us, and the new process gets a
        clean PortAudio.

        Called last in the worker loop, with the queue drained and nothing
        playing, so no utterance is ever cut off mid-word. The uptime floor is
        the loop guard: a machine whose audio device wedges on every open
        would otherwise respawn forever. Below the floor we stay up and
        degraded, which since the same date means synth+afplay — slower to
        first sound, but complete and intelligible.
        """
        if self._wedge_reason is None or self._self_heal_checked:
            return
        if self.queue.size > 0:
            return  # still backed up; decide at the next idle moment
        self._self_heal_checked = True
        uptime = time.monotonic() - self._started_at
        if uptime < _SELF_HEAL_MIN_UPTIME_SECONDS:
            logger.warning(
                "pcm sink: wedged {:.0f}s into this process ({}) — below the "
                "{:.0f}s self-heal floor, staying up on synth+afplay",
                uptime, self._wedge_reason, _SELF_HEAL_MIN_UPTIME_SECONDS,
            )
            return
        logger.warning(
            "pcm sink: wedged ({}) — exiting to recover PortAudio; launchd "
            "KeepAlive respawns us (up {:.0f}s, queue idle)",
            self._wedge_reason, uptime,
        )
        # An empty queue does not mean nothing is in flight: webhook and ntfy
        # pushes are detached tasks that deliberately outlive playback, and the
        # actionable ones carry the Approve/Deny buttons. os._exit takes no
        # atexit hooks and cancels nothing, so a POST still on the wire would
        # simply vanish. Bounded — a hung push must not block the recovery it
        # is riding along with.
        if self._webhook_tasks:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*self._webhook_tasks, return_exceptions=True),
                    _SELF_HEAL_PUSH_DRAIN_SECONDS,
                )
        sys.stderr.flush()
        os._exit(1)

    def _streams_into_ffplay(self, primary: TTSProvider) -> bool:
        """Would this provider's stream end up in ffplay's stdin pipe?

        True means DON'T stream it. ffplay paces playback off a clock that
        keeps running while its pipe is dry, so a producer that stalls loses
        the audio it hands over afterwards — the stall is charged against the
        utterance and ffplay exits early, rc=0, stderr empty (measured
        2026-08-25; see player.pcm_sink_unusable for the numbers). Every
        local PCM provider stalls that way, because each piece is decoded in
        full before a single byte is handed over (xtts.stream) at rtf ~1.2.
        The jitter-buffered PCM sink is immune — it pads with silence and
        never advances a clock — so streaming is fine whenever it's available.

        Container-format providers (ElevenLabs MP3) are exempt: ffplay is the
        only decoder we have for those bytes, and they arrive from the network
        in a steady trickle rather than one blob per sentence.
        """
        if primary.stream_pcm is None:
            return False
        return pcm_sink_unusable() or not self.cfg.tts.pcm_playback

    async def _try_stream(
        self, text: str, lang, voice_id: str | None,
        *, on_spawn=None, session_id: str | None = None,
        emotion: str | None = None, cancellable: bool = True,
    ) -> bool:
        """If the primary TTS supports streaming, pipe chunks straight to the
        audio sink so playback begins before synthesis completes. Returns True
        on success (or on cancel — see below); False on any other failure so
        the caller can fall back to the file-based synth+afplay path.

        When the worker kill()s the sink to cancel playback, the feed raises
        (broken pipe for ffplay, PortAudioError for the PCM sink). That is
        NOT a TTS failure — re-synthesizing and falling back to afplay would
        replay the line we just killed. Detect this by checking
        `_cancelled_sessions` and return True so the caller treats it as
        "playback already concluded".
        """
        primary = self.tts.primary_for(lang)
        if not primary.supports_streaming:
            return False
        if self._streams_into_ffplay(primary):
            # Streaming would truncate the line — take the synth+afplay path.
            logger.debug(
                "SKIP-STREAM {} would stream into ffplay; using synth+afplay",
                primary.name,
            )
            return False
        try:
            chunks = primary.stream(text, lang, voice_id=voice_id, emotion=emotion)
            if primary.stream_pcm is not None:
                sink = await self._spawn_stream_sink(primary, on_spawn=on_spawn)
                try:
                    await sink.feed(chunks)
                except BaseException:
                    # Conclude the sink without masking the real cause
                    # (abort never raises), then let the error decide
                    # cancel-vs-failure below.
                    await sink.abort()
                    raise
                await sink.close()
            else:
                await play_stream(
                    chunks,
                    on_spawn=on_spawn,
                    input_args=primary.stream_input_args,
                )
            return True
        except Exception as exc:
            # A cancel-exempt event's sink is never kill()ed, so for it an
            # exception here is always a genuine failure — even with a
            # (ignored) cancel flag pending on the session.
            if cancellable and session_id and session_id in self._cancelled_sessions:
                logger.debug("stream cancelled for session {}", session_id)
                return True
            logger.warning("Streaming TTS failed, falling back to synth: {}", exc)
            return False

    async def _prewarm_skills(self) -> None:
        """Load the model + index AND run one throwaway query at daemon start so
        the user's first prompt doesn't eat the ~1s cold path. The model load is
        cheap; the real cost is onnxruntime's first inference, which a cached
        index never triggers — so we embed once here. Best-effort: any failure
        just defers the cost to the first real query."""
        if self.skills is not None:
            try:
                ok = await asyncio.to_thread(self.skills.ensure_ready)
                if ok:
                    await asyncio.to_thread(self.skills.query, "warmup")
                logger.info("skills: prewarm {}", "ready" if ok else "unavailable")
            except Exception as exc:  # noqa: BLE001
                logger.warning("skills: prewarm failed ({})", exc)
        if self.mcp is not None:
            try:
                ok = await asyncio.to_thread(self.mcp.ensure_ready)
                if ok:
                    await asyncio.to_thread(self.mcp.query, "warmup")
                logger.info("mcp: prewarm {}", "ready" if ok else "unavailable")
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp: prewarm failed ({})", exc)

    def _tts_prewarm_plan(
        self,
    ) -> tuple[list[TTSProvider], list[tuple[TTSProvider, str]]]:
        """Split the configured TTS providers into (warm now, load lazily).

        Why not just warm every provider we can see: a language override the
        current config can never route to still pays its full model load at
        every daemon start and then sits resident forever. Measured on the
        live daemon 2026-08-25 with behavior.voice_language = "en" and
        tts.provider_zh = "cosyvoice": CosyVoice3 loaded 4.7GB of weights at
        02:28:33 and ran ZERO syntheses in the following 13 hours, while the
        box sat at 97% swap and synthesis RTF degraded from ~1.2 to a median
        of 4.7 as the weights we DO use got paged out.

        Deferral is safe: every provider's model load is lazy at synthesis
        time (CosyVoiceProvider._load_model is called from _synth_once, XTTS's
        from synthesize/stream), so a deferred provider still speaks — its
        first call just pays the load prewarm would have hidden.

        Reachability follows the real lang branches in _process_one /
        _process_one_streaming, not the config alone:
        - the primary is ALWAYS warmed: pre-baked events fall back to "en"
          (`event.lang or "en"`) and the session_start briefing always
          composes in English, so it is reachable whatever the setting says;
        - a pinned voice_language ("en"/"zh") is used verbatim on the LLM
          path, so only that language's override can be selected;
        - "auto" resolves per event via detect_for(cwd), so every override is
          genuinely reachable and all of them are warmed.

        An override for a language the setting never picks is reachable only
        through an explicitly pre-baked event (`jarvis say --lang zh`) —
        rare enough to pay the cold start on.
        """
        voice_lang = self.cfg.behavior.voice_language
        warm: dict[TTSProvider, None] = {self.tts.primary: None}
        reasons: dict[TTSProvider, str] = {}
        for lang, provider in self.tts.overrides.items():
            if voice_lang == "auto" or lang == voice_lang:
                warm[provider] = None
            else:
                reasons.setdefault(
                    provider,
                    f"behavior.voice_language={voice_lang!r} never routes to "
                    f"{lang!r}; loads on first use",
                )
        deferred = [(p, why) for p, why in reasons.items() if p not in warm]
        return list(warm), deferred

    async def _prewarm_tts(self) -> None:
        """Let the providers this config can actually select warm their one-off
        state at daemon start so the user's first notification doesn't eat the
        cold path. Each provider knows what (if anything) it needs to warm —
        XTTS loads the model and conditioning latents; API/subprocess providers
        are no-ops. Providers the config can never route to are left to load
        lazily on their first real synthesis (see _tts_prewarm_plan).
        Best-effort: any failure just defers the cost to the first real event."""
        warm, deferred = self._tts_prewarm_plan()
        for provider, why in deferred:
            logger.info("tts: prewarm deferred ({}) — {}", provider.name, why)
        for provider in warm:
            try:
                await provider.prewarm()
                logger.info("tts: prewarm ready ({})", provider.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tts: prewarm failed ({})", exc)

    async def _prewarm_briefing(self) -> None:
        """Warm the weather cache at daemon start so the first session_start
        briefing doesn't stall on the weather API. Best-effort: any failure
        just defers the fetch to the first real briefing."""
        sb = self.cfg.behavior.session_briefing
        if not sb.enabled:
            return
        try:
            city = sb.city or detect_city()
            await self._weather_cache.get(city, sb.weather_timeout_seconds)
            logger.info("briefing: weather prewarmed ({})", city)
        except Exception as exc:  # noqa: BLE001
            logger.warning("briefing: weather prewarm failed ({})", exc)

    async def run(self) -> None:
        await self.health.start()
        coros = [
            self._worker(),
            self._prewarm_skills(),
            self._prewarm_tts(),
            self._prewarm_briefing(),
        ]
        # Reply-topic subscription only makes sense with a topic to listen
        # on; a half-configured [remote] simply gets no listener (pushes
        # without buttons still work via topic_notify alone).
        if self.cfg.remote.enabled and self.cfg.remote.topic_reply:
            coros.append(self._remote_listener())
        tasks = [asyncio.create_task(coro) for coro in coros]
        try:
            await serve_unix_socket(
                Path(self.cfg.paths.socket),
                self._on_event,
                on_cancel=self.cancel_session,
                # Always wired: handles reload_behavior even when skills/MCP
                # are disabled (the skill paths no-op on None services).
                on_query=self._on_query,
            )
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            # Worker children outlive their parent's event loop unless told
            # otherwise — an orphaned one keeps a multi-GB model resident with
            # nobody to talk to.
            await self.tts.aclose()
            await self.health.stop()


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-daemon")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--health-port", type=int, default=9527)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.add(cfg.paths.log, rotation="10 MB", retention="14 days", level="INFO")
    try:
        asyncio.run(
            Daemon(
                cfg, health_port=args.health_port, config_path=args.config,
            ).run()
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
