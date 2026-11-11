"""The strict-JSON contract: valid → parsed; invalid → one repair; still invalid → fallback."""
from tests._runner import require, run


def _fake_list_llm(responses):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    return FakeListChatModel(responses=responses)


def _validate(data: dict) -> dict:
    if "score" not in data:
        raise ValueError("missing 'score'")
    return {"score": float(data["score"])}


def _prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages([("system", "Return JSON like {{\"score\": 1}}"), ("human", "{x}")])


def test_valid_json_first_try():
    require("langchain_core")
    from core.llm import run_json_chain

    out = run_json_chain(_prompt(), {"x": "hi"}, validate=_validate, fallback={"score": None}, name="t",
                         llm=_fake_list_llm(['```json\n{"score": 7}\n```']))
    assert out["score"] == 7.0 and out["_meta"]["attempts"] == 1 and not out.get("_fallback")


def test_repair_retry_fixes_bad_json():
    require("langchain_core")
    from core.llm import run_json_chain

    out = run_json_chain(_prompt(), {"x": "hi"}, validate=_validate, fallback={"score": None}, name="t",
                         llm=_fake_list_llm(["Sure! Here is the score: seven", '{"score": 6}']))
    assert out["score"] == 6.0 and out["_meta"]["attempts"] == 2


def test_missing_key_triggers_repair_then_fallback():
    require("langchain_core")
    from core.llm import run_json_chain

    out = run_json_chain(_prompt(), {"x": "hi"}, validate=_validate, fallback={"score": None}, name="t",
                         llm=_fake_list_llm(['{"points": 3}', '{"still": "wrong"}']))
    assert out["_fallback"] is True and out["score"] is None and "missing 'score'" in out["_error"]


if __name__ == "__main__":
    raise SystemExit(run(globals()))
