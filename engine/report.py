from __future__ import annotations

from io import BytesIO
from typing import Dict, List
import unicodedata

from fpdf import FPDF


def _to_pdf_text(value: object) -> str:
    """Normalize text so it always renders with FPDF core fonts."""
    text = str(value).replace("₹", "INR ").replace("⚡", "")
    normalized = unicodedata.normalize("NFKD", text)
    # Core fonts support latin-1 only; replace unsupported glyphs safely.
    return normalized.encode("latin-1", "replace").decode("latin-1")


def _write_lines(pdf: FPDF, lines: List[str]) -> None:
    line_width = max(getattr(pdf, "epw", 0), 20)
    for line in lines:
        pdf.multi_cell(line_width, 6, _to_pdf_text(line), wrapmode="CHAR")


def build_life_roadmap_pdf(profile: Dict, decisions: Dict, metrics: Dict, advice: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _to_pdf_text("AI Life Decision Engine - Roadmap"), ln=True)

    pdf.set_font("Helvetica", size=11)
    _write_lines(pdf, ["Profile Snapshot"])
    _write_lines(pdf, [f"- {k.replace('_', ' ').title()}: {v}" for k, v in profile.items()])

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _to_pdf_text("Decisions"), ln=True)
    pdf.set_font("Helvetica", size=11)
    _write_lines(pdf, [f"- {k.replace('_', ' ').title()}: {v}" for k, v in decisions.items()])

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _to_pdf_text("Core Metrics"), ln=True)
    pdf.set_font("Helvetica", size=11)
    _write_lines(pdf, [f"- {k.replace('_', ' ').title()}: {v}" for k, v in metrics.items()])

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _to_pdf_text("Advisor Bot Suggestions"), ln=True)
    pdf.set_font("Helvetica", size=11)
    _write_lines(pdf, [f"{idx}. {item}" for idx, item in enumerate(advice, 1)])

    buffer = BytesIO()
    buffer.write(pdf.output(dest="S"))
    return buffer.getvalue()
