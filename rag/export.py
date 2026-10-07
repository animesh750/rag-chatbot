"""Export a chat transcript (with page-level source citations) to PDF."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fpdf import FPDF


def _clean(text: str) -> str:
    # Core PDF fonts are latin-1 only; drop anything else instead of crashing.
    return text.encode("latin-1", "ignore").decode("latin-1")


def export_chat_to_pdf(messages: Sequence[dict], doc_names: Sequence[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    nl = {"new_x": "LMARGIN", "new_y": "NEXT"}

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _clean("RAG Chatbot - Conversation Export"), **nl)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _clean(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), **nl)
    pdf.cell(0, 6, _clean(f"Documents: {', '.join(doc_names)}"), **nl)
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    for msg in messages:
        is_user = msg["role"] == "user"
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*((60, 120, 220) if is_user else (40, 160, 80)))
        pdf.cell(0, 8, "You:" if is_user else "Assistant:", **nl)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(10)
        pdf.multi_cell(190, 7, _clean(msg["content"]), **nl)
        pdf.ln(3)

        sources = msg.get("sources") or []
        if sources:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            for i, hit in enumerate(sources, start=1):
                snippet = hit.chunk.text[:100].replace("\n", " ")
                label = f"[{i}] {hit.chunk.source[:30]}, p.{hit.chunk.page}: {snippet}"
                pdf.set_x(10)
                pdf.multi_cell(190, 5, _clean(label), **nl)
            pdf.ln(2)

        pdf.set_draw_color(230, 230, 230)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

    return bytes(pdf.output())
