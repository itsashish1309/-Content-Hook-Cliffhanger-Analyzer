"""
core/llm.py — the shared Mistral client and the STRICT-JSON chain helper.

Every analysis (hook, cliffhanger, emotion, verdict) needs the same contract:

    prompt | Mistral (JSON mode) | JsonOutputParser | validate keys
        └─ on bad JSON / missing keys ─► ONE repair attempt (LCEL with_fallbacks)
              └─ still bad ─► graceful fallback dict (the report still renders)

The repair attempt shows the model its own broken reply plus the exact
validation error — far more effective than blindly re-asking.

Reuse note: get_llm() replaces the three copies of get_llm() that lived in
summarizer.py, extractor.py and rag_engine.py in AI Video Assistant.
"""
from __future__ import annotations

import copy
import logging
import time
from functools import lru_cache
from typing import Any, Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

import config as cfg
from core.prompts import REPAIR_INSTRUCTION
from utils.errors import HookLensError

logger = logging.getLogger("hooklens.llm")

# Tests can inject a fake chat model: set_llm_factory(lambda temperature, json_mode: fake)
_llm_factory: Callable[[float, bool], Runnable] | None = None


def set_llm_factory(factory: Callable[[float, bool], Runnable] | None) -> None:
    """Swap the model factory (tests use langchain-core's fake chat models)."""
    global _llm_factory
    _llm_factory = factory
    get_llm.cache_clear()


def require_mistral_key() -> str:
    """Fail fast with a friendly message when no key is set (skipped when a fake model is injected)."""
    if _llm_factory is not None:
        return "injected-test-model"
    key = cfg.mistral_api_key()
    if not key:
        raise HookLensError(
            "MISTRAL_API_KEY is missing.",
            hint="Add it to your .env file (local) or the app's Secrets (Streamlit Cloud), then rerun.",
        )
    return key


@lru_cache(maxsize=8)
def get_llm(temperature: float = cfg.TEMPERATURE_SCORING, json_mode: bool = True) -> Runnable:
    """One cached Mistral client per (temperature, json_mode)."""
    if _llm_factory is not None:
        return _llm_factory(temperature, json_mode)
    from langchain_mistralai import ChatMistralAI  # lazy: offline tests don't need it

    kwargs: dict[str, Any] = dict(
        model=cfg.MISTRAL_MODEL,
        mistral_api_key=require_mistral_key(),
        temperature=temperature,
        max_tokens=cfg.LLM_MAX_TOKENS,
        max_retries=cfg.LLM_MAX_RETRIES,
        timeout=cfg.LLM_TIMEOUT_S,
    )
    try:
        # Fixed seed for temperature-0 scoring → same episode, same scores.
        llm = ChatMistralAI(**kwargs, random_seed=cfg.LLM_SEED) if temperature == 0 else ChatMistralAI(**kwargs)
    except Exception:  # older langchain-mistralai without random_seed
        llm = ChatMistralAI(**kwargs)
    # Mistral JSON mode guarantees syntactically valid JSON; we still validate keys.
    return llm.bind(response_format={"type": "json_object"}) if json_mode else llm


def text_llm(temperature: float = cfg.TEMPERATURE_SUMMARY) -> Runnable:
    """Plain-text model (used by the RAG chat chain)."""
    return get_llm(temperature, False)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def build_json_chain(
    prompt: ChatPromptTemplate,
    validate: Callable[[dict], dict],
    *,
    temperature: float = cfg.TEMPERATURE_SCORING,
    llm: Runnable | None = None,
    counter: dict | None = None,
) -> Runnable:
    """
    LCEL chain: prompt | llm | parse+validate, with one repair fallback.

    `validate(data) -> cleaned_dict` must raise ValueError/KeyError/TypeError
    when required keys are missing; that is converted into an
    OutputParserException (carrying the raw reply) which triggers the repair.
    """
    model = llm or get_llm(temperature, True)
    attempts = counter if counter is not None else {"n": 0}

    def parse_and_validate(message: Any) -> dict:
        attempts["n"] += 1
        raw = _message_text(message)
        try:
            data = JsonOutputParser().parse(raw)  # also strips ```json fences
            if not isinstance(data, dict):
                raise ValueError("the top-level JSON value must be an object")
            return validate(data)
        except OutputParserException:
            raise
        except Exception as exc:  # validation problems → same repair path
            raise OutputParserException(f"Invalid structure: {exc}", llm_output=raw) from exc

    def repair_messages(inputs: dict) -> list:
        error = inputs.get("_error")
        raw = getattr(error, "llm_output", "") or ""
        variables = {key: value for key, value in inputs.items() if key != "_error"}
        return prompt.format_messages(**variables) + [
            AIMessage(content=str(raw)[:4000] or "(empty reply)"),
            HumanMessage(content=REPAIR_INSTRUCTION.format(error=str(error)[:400])),
        ]

    first_try = prompt | model | RunnableLambda(parse_and_validate)
    repair = RunnableLambda(repair_messages) | model | RunnableLambda(parse_and_validate)
    return first_try.with_fallbacks(
        [repair], exceptions_to_handle=(OutputParserException,), exception_key="_error"
    )


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    text = str(exc).lower()
    return status in (401, 403) or ("401" in text and "unauthorized" in text)


def run_json_chain(
    prompt: ChatPromptTemplate,
    variables: dict,
    *,
    validate: Callable[[dict], dict],
    fallback: dict,
    name: str,
    temperature: float = cfg.TEMPERATURE_SCORING,
    llm: Runnable | None = None,
) -> dict:
    """
    Invoke a strict-JSON chain. Always returns a dict:
      • validated LLM output + "_meta" (attempts, latency), or
      • a deep copy of `fallback` with "_fallback": True and "_error".
    Auth errors are raised as HookLensError (no point continuing without a key).
    """
    counter = {"n": 0}
    started = time.perf_counter()
    try:
        chain = build_json_chain(prompt, validate, temperature=temperature, llm=llm, counter=counter)
        result = dict(chain.invoke(variables))
        result["_meta"] = {"name": name, "attempts": counter["n"],
                           "latency_s": round(time.perf_counter() - started, 2)}
        return result
    except HookLensError:
        raise
    except Exception as exc:
        if _is_auth_error(exc):
            raise HookLensError("Mistral rejected the API key (401/403).",
                                hint="Check MISTRAL_API_KEY in .env / Secrets.") from exc
        logger.warning("%s failed after %s attempt(s): %s", name, counter["n"], exc)
        out = copy.deepcopy(fallback)
        out["_fallback"] = True
        out["_error"] = f"{name}: {type(exc).__name__}: {str(exc)[:300]}"
        out["_meta"] = {"name": name, "attempts": counter["n"],
                        "latency_s": round(time.perf_counter() - started, 2)}
        return out
