import json
import httpx
import pytest
import respx

from jarvis.config import (
    DeepSeekConfig,
    OllamaConfig,
    SiliconFlowConfig,
    ZhipuConfig,
)
from jarvis.phrase.providers.deepseek import DeepSeekProvider
from jarvis.phrase.providers.ollama import OllamaProvider
from jarvis.phrase.providers.siliconflow import SiliconFlowProvider
from jarvis.phrase.providers.zhipu import ZhipuProvider

_MESSAGES = [
    {"role": "system", "content": "you are jarvis"},
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm /tmp/x"}'},
]


@pytest.mark.asyncio
async def test_deepseek_returns_assistant_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "先生，Claude 请求许可。"}}
                ]
            },
        )
        p = DeepSeekProvider(cfg)
        out = await p.generate(_MESSAGES)
    assert out == "先生，Claude 请求许可。"


@pytest.mark.asyncio
async def test_deepseek_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = DeepSeekProvider(DeepSeekConfig())
    with pytest.raises(RuntimeError):
        await p.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_deepseek_raises_on_http_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(500)
        p = DeepSeekProvider(cfg)
        with pytest.raises(httpx.HTTPStatusError):
            await p.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_ollama_returns_assistant_text():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200,
            json={"message": {"role": "assistant", "content": "先生，请过目。"}},
        )
        p = OllamaProvider(cfg)
        out = await p.generate(_MESSAGES)
    assert out == "先生，请过目。"


@pytest.mark.asyncio
async def test_ollama_raises_on_connection_error():
    cfg = OllamaConfig(base_url="http://127.0.0.1:1")  # nothing listens here
    p = OllamaProvider(cfg)
    with pytest.raises(httpx.HTTPError):
        await p.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_zhipu_returns_assistant_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    cfg = ZhipuConfig()
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    with respx.mock() as router:
        route = router.post(url).respond(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "先生，Claude 请求许可。"}}
                ]
            },
        )
        p = ZhipuProvider(cfg)
        out = await p.generate(_MESSAGES)
    assert out == "先生，Claude 请求许可。"
    # The compatible endpoint must NOT carry a /v1 segment (Zhipu 404s on it).
    sent = str(route.calls.last.request.url)
    assert sent.endswith("/api/paas/v4/chat/completions")
    assert "/v1/" not in sent


@pytest.mark.asyncio
async def test_zhipu_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    p = ZhipuProvider(ZhipuConfig())
    with pytest.raises(RuntimeError):
        await p.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_zhipu_inline_key_from_config_beats_env(monkeypatch: pytest.MonkeyPatch):
    # Inline api_key (from config.toml) is used even with no env var set, and
    # wins over the env var when both are present.
    monkeypatch.setenv("ZHIPU_API_KEY", "env-key")
    cfg = ZhipuConfig(api_key="inline-key")
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    with respx.mock() as router:
        route = router.post(url).respond(
            200, json={"choices": [{"message": {"content": "先生。"}}]},
        )
        out = await ZhipuProvider(cfg).generate(_MESSAGES)
    assert out == "先生。"
    assert route.calls.last.request.headers["authorization"] == "Bearer inline-key"


@pytest.mark.asyncio
async def test_siliconflow_returns_assistant_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    cfg = SiliconFlowConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "先生，请过目。"}}]},
        )
        out = await SiliconFlowProvider(cfg).generate(_MESSAGES)
    assert out == "先生，请过目。"


@pytest.mark.asyncio
async def test_siliconflow_inline_key_and_no_v1_duplication(
    monkeypatch: pytest.MonkeyPatch,
):
    # Inline key is used; standard OpenAI path is /v1/chat/completions (once).
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    cfg = SiliconFlowConfig(api_key="inline-sf")
    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "ok"}}]},
        )
        out = await SiliconFlowProvider(cfg).generate(_MESSAGES)
    assert out == "ok"
    sent = str(route.calls.last.request.url)
    assert sent.endswith("/v1/chat/completions")
    assert sent.count("/v1/") == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer inline-sf"


@pytest.mark.asyncio
async def test_siliconflow_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    p = SiliconFlowProvider(SiliconFlowConfig())
    with pytest.raises(RuntimeError):
        await p.generate(_MESSAGES)


@pytest.mark.asyncio
async def test_ollama_requests_bounded_context_window():
    """The payload must pin `num_ctx` so the KV cache is sized for Jarvis's
    ~1k-token prompts rather than Ollama's 32k server default."""
    cfg = OllamaConfig(num_ctx=2048)
    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/api/chat").respond(
            200, json={"message": {"role": "assistant", "content": "Sir."}}
        )
        await OllamaProvider(cfg).generate(_MESSAGES)
    sent = json.loads(route.calls.last.request.content)
    assert sent["options"]["num_ctx"] == 2048
