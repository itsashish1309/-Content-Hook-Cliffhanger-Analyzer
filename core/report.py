"""
core/report.py — one-page PDF scorecard (reportlab), returned as bytes for
st.download_button (or written to disk by the CLI).

Layout: header band → 3 score cards (Hook, Cliffhanger, Retention Risk) →
verdict → hook & cliffhanger sub-scores with bars → pacing stats + emotion
sparkline → top 3 fixes → best rewrite / best ending / suggested cut.
Everything is wrapped in KeepInFrame(mode="shrink") so it always fits one page.

Font note: the built-in PDF fonts (Helvetica) only cover Latin text. Lines in
Devanagari are replaced by their English translation (marked [EN]) so the PDF
never shows empty boxes; the app itself shows the original script.
"""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import KeepInFrame, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config as cfg
from core.common import fmt_ts, truncate

# ── Palette (print-friendly version of the app theme) ─────────────────────────
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#e5e7eb")
DARK = colors.HexColor("#111118")
PURPLE = colors.HexColor("#7c3aed")
CYAN = colors.HexColor("#06b6d4")
BAND = {"red": colors.HexColor("#ef4444"), "amber": colors.HexColor("#f59e0b"),
        "green": colors.HexColor("#10b981"), "none": colors.HexColor("#9ca3af")}
TINT = {"red": colors.HexColor("#fef2f2"), "amber": colors.HexColor("#fffbeb"),
        "green": colors.HexColor("#ecfdf5"), "none": colors.HexColor("#f9fafb")}

PAGE_W, PAGE_H = A4
MARGIN = 26


def _style(name: str, size: float = 8.5, leading: float | None = None, color=INK,
           font: str = "Helvetica", **kw) -> ParagraphStyle:
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading or size * 1.3,
                          textColor=color, alignment=TA_LEFT, **kw)


S = {
    "kicker": _style("kicker", 7, color=CYAN, font="Helvetica-Bold"),
    "title": _style("title", 15, 18, colors.white, "Helvetica-Bold"),
    "meta": _style("meta", 7.5, color=colors.HexColor("#c4c4dc")),
    "h": _style("h", 8, color=PURPLE, font="Helvetica-Bold", spaceAfter=2),
    "body": _style("body", 8.3, 10.8),
    "small": _style("small", 7.2, 9.2, MUTED),
    "card_label": _style("card_label", 7, color=MUTED, font="Helvetica-Bold"),
    "card_sub": _style("card_sub", 7, 8.8, MUTED),
    "fix_title": _style("fix_title", 8.5, 10.5, INK, "Helvetica-Bold"),
    "quote": _style("quote", 8.3, 10.6, INK, "Helvetica-Oblique"),
}


# ── Text safety ───────────────────────────────────────────────────────────────
def _latin_ok(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def pdf_text(text, fallback=None, limit: int | None = None) -> str:
    """
    Make text safe for a reportlab Paragraph:
      • XML-escape (&, <, >) — Paragraph parses markup, so LLM/transcript text
        must never be interpreted as tags;
      • non-Latin text (e.g. Devanagari) → English fallback marked [EN].
    """
    clean = " ".join(str(text or "").split())
    if limit:
        clean = truncate(clean, limit)
    if not _latin_ok(clean):
        alt = " ".join(str(fallback or "").split())
        if limit:
            alt = truncate(alt, limit)
        if alt and _latin_ok(alt):
            clean = f"[EN] {alt}"
        else:
            stripped = clean.encode("cp1252", "ignore").decode("cp1252")
            letters = sum(ch.isalpha() for ch in clean) or 1
            kept = sum(ch.isalpha() for ch in stripped)
            clean = stripped if kept / letters > 0.6 else "[non-Latin text — see the HookLens app]"
    return escape(clean)


def _p(text, style="body", fallback=None, limit=None) -> Paragraph:
    return Paragraph(pdf_text(text, fallback, limit), S[style])


def _fmt(score) -> str:
    return "–" if score is None else f"{score:.1f}"


# ── Small drawings ────────────────────────────────────────────────────────────
def _bar(score, width: float = 92, height: float = 6) -> Drawing:
    band = cfg.score_band(score)
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=LINE, strokeColor=None, rx=2, ry=2))
    if score is not None:
        drawing.add(Rect(0, 0, max(2.0, width * float(score) / 10.0), height,
                         fillColor=BAND[band], strokeColor=None, rx=2, ry=2))
    return drawing


def _sparkline(result: dict, width: float, height: float = 78) -> Drawing:
    """Emotion intensity over time, with hook/cliffhanger windows shaded and peaks marked."""
    emotion = result.get("emotion") or {}
    points = emotion.get("points") or []
    duration = float(result.get("duration") or 1.0)
    drawing = Drawing(width, height)
    top, bottom = height - 10, 12
    plot_h = top - bottom

    def x(t: float) -> float:
        return max(0.0, min(width, width * float(t) / duration))

    def y(v: float) -> float:
        return bottom + plot_h * float(v) / 10.0

    drawing.add(Rect(0, bottom, x(cfg.HOOK_WINDOW_S), plot_h, fillColor=colors.HexColor("#ecfeff"), strokeColor=None))
    cliff_start = max(0.0, duration - cfg.CLIFF_WINDOW_S)
    drawing.add(Rect(x(cliff_start), bottom, width - x(cliff_start), plot_h,
                     fillColor=colors.HexColor("#f5f3ff"), strokeColor=None))
    for level in (0, 5, 10):
        drawing.add(Line(0, y(level), width, y(level), strokeColor=LINE, strokeWidth=0.5))
    drawing.add(String(2, 2, "0:00", fontName="Helvetica", fontSize=6, fillColor=MUTED))
    drawing.add(String(width - 22, 2, fmt_ts(duration), fontName="Helvetica", fontSize=6, fillColor=MUTED))
    drawing.add(String(2, top + 2, "hook", fontName="Helvetica", fontSize=6, fillColor=CYAN))
    drawing.add(String(x(cliff_start) + 2, top + 2, "cliffhanger", fontName="Helvetica", fontSize=6, fillColor=PURPLE))
    drawing.add(String(width / 2 - 20, top + 2, "intensity 0-10", fontName="Helvetica", fontSize=6, fillColor=MUTED))

    coords = []
    for point in points:
        if point.get("intensity") is not None:
            coords.extend([x(point["mid"]), y(point["intensity"])])
    if len(coords) >= 4:
        drawing.add(PolyLine(coords, strokeColor=PURPLE, strokeWidth=1.4))
    for peak in emotion.get("peaks") or []:
        drawing.add(Circle(x(peak["time"]), y(peak["intensity"]), 2.2, fillColor=BAND["red"], strokeColor=None))
    for dip in emotion.get("dips") or []:
        drawing.add(Circle(x(dip["time"]), y(dip["intensity"]), 2.0, fillColor=CYAN, strokeColor=None))
    return drawing


# ── Building blocks ───────────────────────────────────────────────────────────
def _score_card(label: str, value_text: str, sub: str, band: str, width: float) -> Table:
    value_style = _style(f"v_{label}", 22, 24, BAND[band], "Helvetica-Bold")
    card = Table(
        [[Paragraph(escape(label.upper()), S["card_label"])],
         [Paragraph(escape(value_text), value_style)],
         [Paragraph(escape(sub), S["card_sub"])]],
        colWidths=[width],
    )
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT[band]),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BAND[band]),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return card


def _subscore_table(title: str, analysis: dict, labels: dict, weights: dict, width: float) -> Table:
    rows = [[Paragraph(escape(title), S["h"]), "", ""]]
    sub = analysis.get("sub_scores") or {}
    for key, label in labels.items():
        value = sub.get(key)
        rows.append([
            Paragraph(f"{escape(label)} <font color='#9ca3af'>x{weights.get(key, 0):.2f}</font>", S["small"]),
            _bar(value),
            Paragraph(f"<b>{_fmt(value)}</b>", S["body"]),
        ])
    table = Table(rows, colWidths=[width - 132, 98, 34])
    table.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return table


def _two_col(left, right, width: float, gap: float = 14) -> Table:
    col = (width - gap) / 2
    table = Table([[left, "", right]], colWidths=[col, gap, col])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return table


def _rule(width: float) -> Table:
    table = Table([[""]], colWidths=[width], rowHeights=[4])
    table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, LINE)]))
    return table


# ── Public API ────────────────────────────────────────────────────────────────
def build_pdf_report(result: dict) -> bytes:
    """Render the one-page scorecard and return the PDF bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN,
                            title=f"HookLens — {result.get('title', 'Episode')}", author=cfg.APP_NAME)
    width = PAGE_W - 2 * MARGIN
    hook = result.get("hook") or {}
    cliff = result.get("cliffhanger") or {}
    pacing = result.get("pacing") or {}
    emotion = result.get("emotion") or {}
    verdict = result.get("verdict") or {}
    story = []

    # Header band
    analysed = result.get("analysed_at", "")[:16].replace("T", " ")
    lang = cfg.LANGUAGES.get(result.get("language", ""), {}).get("label", result.get("language", ""))
    meta = (f"{lang} · {fmt_ts(result.get('duration'))} runtime · analysed {analysed} UTC · "
            f"{(result.get('transcript') or {}).get('engine', '')}")
    header = Table([[Paragraph("HOOKLENS · EPISODE SCORECARD", S["kicker"])],
                    [_p(result.get("title", "Untitled episode"), "title", limit=90)],
                    [Paragraph(pdf_text(meta), S["meta"])]], colWidths=[width])
    header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), DARK),
                                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                ("TOPPADDING", (0, 0), (0, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9)]))
    story += [header, Spacer(1, 9)]

    # Score cards
    card_w = (width - 16) / 3
    risk = verdict.get("risk", "Unknown")
    cards = Table([[
        _score_card("Hook · first 10 s", f"{_fmt(hook.get('score'))}/10", "Will viewers stay past 10 seconds?",
                    cfg.score_band(hook.get("score")), card_w),
        _score_card("Cliffhanger · last 15 s", f"{_fmt(cliff.get('score'))}/10", "Will viewers unlock the next episode?",
                    cfg.score_band(cliff.get("score")), card_w),
        _score_card("Retention risk", risk.upper(), f"Weighted score {_fmt(verdict.get('overall'))}/10",
                    cfg.RISK_BAND_COLOUR.get(risk, "none"), card_w),
    ]], colWidths=[card_w + 8, card_w + 8, card_w])
    cards.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (1, 0), 8)]))
    story += [cards, Spacer(1, 8)]

    # Verdict
    story.append(Paragraph("VERDICT", S["h"]))
    story.append(_p(verdict.get("summary", ""), "body", limit=420))
    if verdict.get("logline"):
        story.append(_p(f"Logline: {verdict['logline']}", "small", limit=240))
    story += [Spacer(1, 6), _rule(width), Spacer(1, 5)]

    # Sub-scores
    col = (width - 14) / 2
    story.append(_two_col(
        _subscore_table(f"HOOK SUB-SCORES  ({_fmt(hook.get('score'))}/10)", hook, cfg.HOOK_DIMENSIONS,
                        cfg.HOOK_WEIGHTS, col),
        _subscore_table(f"CLIFFHANGER SUB-SCORES  ({_fmt(cliff.get('score'))}/10)", cliff, cfg.CLIFF_DIMENSIONS,
                        cfg.CLIFF_WEIGHTS, col),
        width))
    story.append(Spacer(1, 4))
    story.append(_two_col(_p(hook.get("reason", ""), "small", limit=330),
                          _p(cliff.get("reason", ""), "small", limit=330), width))
    story += [Spacer(1, 6), _rule(width), Spacer(1, 5)]

    # Pacing + emotion
    slow = pacing.get("slow_sections") or []
    gap = pacing.get("longest_gap") or {}
    stats = emotion.get("stats") or {}
    pacing_rows = [
        [Paragraph(f"PACING  ({_fmt(pacing.get('score'))}/10)", S["h"]), ""],
        [_p("Words per second", "small"), _p(f"{pacing.get('wps', 0):.2f}  (target {cfg.TARGET_WPS})")],
        [_p("Dialogue coverage", "small"), _p(f"{pacing.get('dialogue_coverage', 0):.0%}")],
        [_p("Silence ratio", "small"), _p(f"{pacing.get('silence_ratio', 0):.0%}")],
        [_p("Longest gap", "small"), _p(f"{gap.get('length', 0):.1f} s at {fmt_ts(gap.get('start'))}")],
        [_p("Slow sections", "small"),
         _p(", ".join(f"{fmt_ts(s['start'])}-{fmt_ts(s['end'])} ({s.get('kind', 'slow')})" for s in slow[:3]) or "none")],
        [_p("Emotion", "small"),
         _p(f"mean {stats.get('mean', 0):.1f} · final 15 s {stats.get('end_mean', 0):.1f} · "
            f"mostly {stats.get('dominant', '-')}  ({_fmt(emotion.get('score'))}/10)")],
    ]
    pacing_table = Table(pacing_rows, colWidths=[78, col - 78])
    pacing_table.setStyle(TableStyle([("SPAN", (0, 0), (-1, 0)), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                      ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    emotion_block = Table([[Paragraph("EMOTION TIMELINE  (red = peak, cyan = dip)", S["h"])],
                           [_sparkline(result, col)]], colWidths=[col])
    emotion_block.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [_two_col(pacing_table, emotion_block, width), Spacer(1, 6), _rule(width), Spacer(1, 5)]

    # Top fixes
    story.append(Paragraph("TOP 3 FIXES  (ordered by estimated impact on the weighted score)", S["h"]))
    fixes = verdict.get("fixes") or []
    if fixes:
        rows = []
        for i, fix in enumerate(fixes, start=1):
            rows.append([
                Paragraph(f"<b>{i}</b>", _style(f"n{i}", 11, 12, PURPLE, "Helvetica-Bold")),
                [_p(fix.get("title", ""), "fix_title", limit=90),
                 _p(fix.get("action", ""), "body", fallback=fix.get("template_action"), limit=300)],
                Paragraph(f"+{fix.get('impact', 0):.2f} pts", _style(f"i{i}", 7.5, 9, CYAN, "Helvetica-Bold")),
            ])
        fix_table = Table(rows, colWidths=[18, width - 72, 54])
        fix_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                       ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                                       ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(fix_table)
    else:
        story.append(_p("No major fixes — this episode is in good shape.", "body"))
    story += [Spacer(1, 6), _rule(width), Spacer(1, 5)]

    # Best suggestions
    rewrite = (hook.get("rewrites") or [{}])[0]
    ending = (cliff.get("endings") or [{}])[0]
    left = [Paragraph("BEST OPENING REWRITE", S["h"])]
    if rewrite.get("line"):
        left.append(_p(f"“{rewrite['line']}”", "quote", fallback=f"“{rewrite.get('english', '')}”", limit=230))
        if rewrite.get("english") and rewrite["english"] != rewrite["line"] and _latin_ok(rewrite["line"]):
            left.append(_p(f"EN: {rewrite['english']}", "small", limit=230))
        left.append(_p(f"Technique: {rewrite.get('technique', '')}", "small"))
    else:
        left.append(_p("Not available.", "small"))
    right = [Paragraph("BEST ALTERNATIVE ENDING", S["h"])]
    if ending.get("ending"):
        right.append(_p(f"“{ending['ending']}”", "quote", fallback=f"“{ending.get('english', '')}”", limit=230))
        if ending.get("english") and ending["english"] != ending["ending"] and _latin_ok(ending["ending"]):
            right.append(_p(f"EN: {ending['english']}", "small", limit=230))
    else:
        right.append(_p("Not available.", "small"))
    if cliff.get("best_cut_s") is not None:
        trim = cliff.get("trim_s") or 0
        cut_text = (f"Suggested cut: {fmt_ts(cliff['best_cut_s'], True)}"
                    + (f" (trim the last {trim:.1f} s)" if trim >= 0.5 else " (current ending is best)"))
        right.append(_p(cut_text, "small"))
    story += [_two_col(left, right, width), Spacer(1, 8)]

    # Footer
    story.append(_p(f"Generated by {cfg.APP_NAME} v{cfg.APP_VERSION} · Scores are LLM-assisted estimates against a "
                    "fixed rubric; aggregation, pacing and risk bands are deterministic. Calibrate weights against "
                    "real retention and unlock data before using them for greenlight decisions.", "small"))

    doc.build([KeepInFrame(width, PAGE_H - 2 * MARGIN, story, mode="shrink")])
    return buffer.getvalue()


def save_pdf_report(result: dict, path=None) -> str:
    """CLI helper: write the PDF to data/reports/ (or `path`) and return the path."""
    target = path or cfg.REPORTS_DIR / f"hooklens_{result.get('episode_id', 'episode')}_" \
                                       f"{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    with open(target, "wb") as handle:
        handle.write(build_pdf_report(result))
    return str(target)
