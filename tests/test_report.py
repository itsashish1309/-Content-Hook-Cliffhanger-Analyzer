"""PDF scorecard: builds a one-page PDF from a full synthetic analysis (offline)."""
import re

from tests._runner import require, run

import config as cfg


def _synthetic_result():
    require("langchain_core")
    from tests.test_pipeline_offline import run_offline_pipeline

    result, _ = run_offline_pipeline()
    return result


def test_pdf_is_generated_and_one_page():
    require("reportlab")
    from core.report import build_pdf_report

    pdf = build_pdf_report(_synthetic_result())
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000
    assert re.search(rb"/Count\s+1\b", pdf), "scorecard must be exactly one page"
    out = cfg.REPORTS_DIR / "sample_scorecard.pdf"
    out.write_bytes(pdf)
    print(f"     sample written to {out}")


def test_devanagari_falls_back_to_english():
    require("reportlab")
    from core.report import pdf_text

    assert pdf_text("नमस्ते दुनिया", fallback="Hello world") == "[EN] Hello world"
    assert pdf_text("<b>bold & risky</b>") == "&lt;b&gt;bold &amp; risky&lt;/b&gt;"   # markup is escaped
    assert "non-Latin" in pdf_text("नमस्ते दुनिया")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
