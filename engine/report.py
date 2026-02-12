from __future__ import annotations

from typing import Dict, List

from fpdf import FPDF


def build_life_roadmap_pdf(profile: Dict, decisions: Dict, metrics: Dict, advice: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI Life Decision Engine - Roadmap", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Profile Snapshot", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in profile.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Decisions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in decisions.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Core Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in metrics.items():
        pdf.multi_cell(0, 6, f"- {k.replace('_', ' ').title()}: {v}")

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Advisor Suggestions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for idx, item in enumerate(advice, 1):
        pdf.multi_cell(0, 6, f"{idx}. {item}")

    rendered = pdf.output(dest="S")
    return rendered if isinstance(rendered, (bytes, bytearray)) else rendered.encode("latin-1")
