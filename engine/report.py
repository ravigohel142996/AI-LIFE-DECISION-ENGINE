from __future__ import annotations

from io import BytesIO
import textwrap
from typing import Dict, List

from fpdf import FPDF


def _sanitize_pdf_text(value: object) -> str:
    """Convert text to latin-1 safe content for built-in FPDF fonts."""

    text = str(value)
    replacements = {
        "—": "-",
        "–": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "₹": "INR ",
        "⚡": "",
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _safe_multiline(pdf: FPDF, text: str, line_height: float = 6) -> None:
    """Render wrapped text using a deterministic width to avoid layout crashes."""

    effective_width = max(pdf.w - pdf.l_margin - pdf.r_margin, 20)
    wrapped_lines = textwrap.wrap(
        _sanitize_pdf_text(text),
        width=95,
        break_long_words=True,
        replace_whitespace=False,
    )
    content = "\n".join(wrapped_lines) if wrapped_lines else " "
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(effective_width, line_height, content)


def build_life_roadmap_pdf(profile: Dict, decisions: Dict, metrics: Dict, advice: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _sanitize_pdf_text("AI Life Decision Engine - Roadmap"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=11)
    _safe_multiline(pdf, "Profile Snapshot", line_height=7)
    for k, v in profile.items():
        _safe_multiline(pdf, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Decisions"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in decisions.items():
        _safe_multiline(pdf, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Core Metrics"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in metrics.items():
        _safe_multiline(pdf, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize_pdf_text("Advisor Bot Suggestions"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for idx, item in enumerate(advice, 1):
        _safe_multiline(pdf, f"{idx}. {item}")

    buffer = BytesIO()
    buffer.write(pdf.output(dest="S"))
    return buffer.getvalue()
