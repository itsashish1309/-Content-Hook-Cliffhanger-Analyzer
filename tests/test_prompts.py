"""Every prompt has exactly the expected variables and renders its JSON examples correctly."""
from tests._runner import require, run


def test_prompt_variables_match_contract():
    require("langchain_core")
    from core import prompts

    for name, expected in prompts.PROMPT_VARIABLES.items():
        prompt = getattr(prompts, name)
        actual = set(prompt.input_variables)
        assert actual == expected, f"{name}: unexpected {actual - expected}, missing {expected - actual}"


def test_prompts_render_with_literal_json_braces():
    require("langchain_core")
    from core import prompts

    for name, expected in prompts.PROMPT_VARIABLES.items():
        prompt = getattr(prompts, name)
        messages = prompt.format_messages(**{var: f"<{var}>" for var in expected})
        system = messages[0].content
        assert "{{" not in system and "}}" not in system, f"{name}: escaped braces leaked"
        if name != "CHAT_PROMPT":
            assert '{"' in system, f"{name}: JSON schema example missing"


def test_rubric_has_calibration_anchors():
    require("langchain_core")
    from core.prompts import CLIFF_SCORE_SYSTEM, HOOK_SCORE_SYSTEM

    for text in (HOOK_SCORE_SYSTEM, CLIFF_SCORE_SYSTEM):
        assert "2/10" in text and "5/10" in text and "9/10" in text


if __name__ == "__main__":
    raise SystemExit(run(globals()))
