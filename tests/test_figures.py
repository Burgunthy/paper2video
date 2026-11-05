"""Figure extraction: default DPI must be 300 (display width ~1168px @720p)."""
from pathlib import Path

import pymupdf

from paper2video.figures import extract_figures


def _make_paper(tmp_path: Path) -> Path:
    """One page with a wide vector rectangle above a wide 'Figure 1' caption."""
    pdf_path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # Wide white rect (vector drawing) in the upper body of the page
    page.draw_rect(pymupdf.Rect(50, 100, 550, 400), color=(0, 0, 0), fill=(1, 1, 1))
    # Wide caption block spanning >55% of the page width => full-width band
    page.insert_text((40, 700), "Figure 1: " + "wide caption text " * 8, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_default_extraction_dpi_is_300(tmp_path):
    pdf = _make_paper(tmp_path)
    out_dir = tmp_path / "figs"
    figs = extract_figures(pdf, out_dir)

    assert len(figs) == 1
    assert figs[0].id == "fig1"
    # Region spans roughly the drawn rect width (~500pt). At 300 DPI that is
    # ~2000+px; at 200 DPI only ~1400px.
    assert figs[0].width_px > 1900, (
        f"expected >=300dpi extraction, got width_px={figs[0].width_px}"
    )
