from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from fpdf import FPDF


def build_life_roadmap_pdf(profile: Dict, decisions: Dict, metrics: Dict, advice: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI Life Decision Engine - Roadmap", ln=True)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, "Profile Snapshot")
    for k, v in profile.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Decisions", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in decisions.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Core Metrics", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in metrics.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Advisor Bot Suggestions", ln=True)
    pdf.set_font("Helvetica", size=11)
    for idx, item in enumerate(advice, 1):
        pdf.multi_cell(0, 6, f"{idx}. {item}")

    buffer = BytesIO()
    buffer.write(pdf.output(dest="S"))
    return buffer.getvalue()
