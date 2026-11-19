"""Тонкий клиент OpenAI-совместимого API llama-server.

Модуль не печатает в stdout: сообщения об ошибках и предупреждения
уходят в stderr, чтобы не ломать машиночитаемый вывод вызывающих скриптов.
"""

import logging
import os
import time

import requests

BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.getenv("LLM_MODEL", "t-lite-it-2.1")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

SEED = 42
TOP_P = 0.9
TOP_K = 20
REPEAT_PENALTY = 1.05
RETRY_COUNT = 2
RETRY_DELAY_S = 2.0

logger = logging.getLogger(__name__)


def _chat_url() -> str:
    return f"{BASE_URL.rstrip('/')}/chat/completions"


def _models_url() -> str:
    return f"{BASE_URL.rstrip('/')}/models"


HEALTHCHECK_ERROR = (
    "LLM недоступна. Запустите llama-server "
    "(см. README.md, раздел «Генерация ответа»)."
)


def _probe_error(exc: Exception) -> str:
    """Короткая причина недоступности без трассировки."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, requests.Timeout):
        return "таймаут соединения"
    if isinstance(exc, requests.ConnectionError):
        return f"нет соединения с {BASE_URL}"
    return str(exc)[:200]


def probe(timeout_s: float = 2.0) -> dict:
    """Быстрая проверка доступности LLM без исключения.

    Возвращает {"ok": bool, "model": str | None, "error": str | None}.
    """
    try:
        response = requests.get(_models_url(), timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"ok": False, "model": None, "error": _probe_error(exc)}

    if not isinstance(data, dict):
        return {"ok": False, "model": None, "error": "неожиданный формат ответа"}
    models = data.get("data") or []
    model = models[0].get("id", MODEL) if models else MODEL
    return {"ok": True, "model": model, "error": None}


def healthcheck() -> str:
    """Проверить доступность сервера и вернуть идентификатор модели."""
    try:
        response = requests.get(_models_url(), timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(HEALTHCHECK_ERROR) from exc

    models = data.get("data") or []
    if not models:
        raise RuntimeError(HEALTHCHECK_ERROR)
    return models[0].get("id", MODEL)


def _post_with_retries(payload: dict) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            response = requests.post(_chat_url(), json=payload, timeout=TIMEOUT)
        except requests.ConnectionError as exc:
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_S)
                continue
            raise RuntimeError(f"Не удалось подключиться к LLM: {exc}") from exc
        except requests.Timeout as exc:
            raise RuntimeError(f"LLM не ответила за {TIMEOUT:.0f} с") from exc

        if response.status_code >= 500:
            last_error = RuntimeError(f"LLM вернула HTTP {response.status_code}: {response.text[:300]}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_S)
                continue
            raise last_error

        return response

    raise RuntimeError(f"LLM недоступна: {last_error}")


def _request_chat(payload: dict) -> requests.Response:
    response = _post_with_retries(payload)
    if response.status_code == 400 and "chat_template_kwargs" in payload:
        logger.warning("llama-server не принял chat_template_kwargs, повтор без этого поля и с /no_think")
        fallback_payload = {key: value for key, value in payload.items() if key != "chat_template_kwargs"}
        fallback_payload["messages"] = _prepend_no_think(payload["messages"])
        response = _post_with_retries(fallback_payload)
    return response


def _prepend_no_think(messages: list[dict]) -> list[dict]:
    prepared = [dict(message) for message in messages]
    for message in prepared:
        if message.get("role") == "system":
            message["content"] = f"/no_think\n{message.get('content', '')}"
            return prepared
    return [{"role": "system", "content": "/no_think"}, *prepared]


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 512,
    think: bool = False,
) -> dict:
    """Отправить сообщения в чат и вернуть текст, usage и задержку."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": max_tokens,
        "seed": SEED,
        "stream": False,
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    started_at = time.perf_counter()
    response = _request_chat(payload)
    latency_s = time.perf_counter() - started_at

    if not response.ok:
        raise RuntimeError(f"LLM вернула HTTP {response.status_code}: {response.text[:500]}")

    try:
        data = response.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise RuntimeError("LLM вернула ответ в неожиданном формате") from exc

    return {
        "text": message.get("content") or "",
        "usage": data.get("usage") or {},
        "latency_s": latency_s,
    }
