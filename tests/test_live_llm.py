"""
Live Mistral check (costs a few API calls). Skips automatically without MISTRAL_API_KEY.

    python tests/test_live_llm.py
"""
from dotenv import load_dotenv

load_dotenv()

from tests._runner import SkipTest, require, run  # noqa: E402

import config as cfg  # noqa: E402
from tests.fixtures import SAMPLE_DURATION, sample_segments  # noqa: E402


def test_live_hook_scoring_returns_valid_json():
    require("langchain_mistralai")
    if not cfg.mistral_api_key():
        raise SkipTest("MISTRAL_API_KEY not set")
    from core.hook_analyzer import analyze_hook
    from core.llm import set_llm_factory

    set_llm_factory(None)
    hook = analyze_hook(sample_segments(), SAMPLE_DURATION, "hinglish")
    print(f"     hook {hook['score']}/10 · sub-scores {hook['sub_scores']}")
    for rewrite in hook["rewrites"]:
        print(f"     ✏️  {rewrite['line']}  |  EN: {rewrite['english']}")
    assert not hook["fallback"], hook["errors"]
    assert 0 <= hook["score"] <= 10 and len(hook["rewrites"]) == 3


if __name__ == "__main__":
    raise SystemExit(run(globals()))
