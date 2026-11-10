"""Extract figures from a paper PDF for figure-scene rendering.

Strategy:
  1. Locate caption blocks ("Figure N ..." / "Fig. N ..." / "그림 N ...").
  2. Crop the graphics region directly above each caption (within the
     caption's column band) and render it at high DPI.
  3. Fallback when no captions are found: extract large embedded images.

Output: out_dir/figNN.png + out_dir/figures.json manifest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf

CAPTION_RE = re.compile(r"^(figure|fig\.|그림)\s*(\d+)", re.IGNORECASE)
MAX_UP_PT = 620          # how far above a caption we look for graphics
MIN_REGION_PT = 40       # skip degenerate crops
MAX_FIGURES = 14


@dataclass
class FigureInfo:
    id: str          # "fig2"
    number: int
    page: int        # 1-based
    caption: str
    path: str        # png filename inside out_dir
    width_px: int
    height_px: int


def _caption_blocks(page) -> list[tuple[int, dict]]:
    """Return (figure_number, block_dict) for blocks that START a caption."""
    out: list[tuple[int, dict]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        first_text = "".join(
            span["text"] for span in lines[0].get("spans", [])
        ).strip()
        m = CAPTION_RE.match(first_text)
        if m:
            out.append((int(m.group(2)), block))
    return out


def _graphics_rects(page) -> list[pymupdf.Rect]:
    """Bounding boxes of images + vector drawings on the page."""
    rects: list[pymupdf.Rect] = []
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                rects.append(pymupdf.Rect(r))
        except Exception:
            continue
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r is not None and not r.is_empty:
                rects.append(pymupdf.Rect(r))
    except Exception:
        pass
    return rects


def _text_rects(page) -> list[pymupdf.Rect]:
    return [
        pymupdf.Rect(b["bbox"])
        for b in page.get_text("dict")["blocks"]
        if b.get("type") == 0
    ]


def _extract_with_captions(doc: pymupdf.Document, out_dir: Path, dpi: int) -> list[FigureInfo]:
    figures: dict[int, FigureInfo] = {}
    zoom = dpi / 72.0
    for pno, page in enumerate(doc, start=1):
        for number, cblock in _caption_blocks(page):
            if number in figures:
                continue
            cap = pymupdf.Rect(cblock["bbox"])
            page_rect = page.rect
            # Column band: wide captions imply full-width figures.
            if cap.width > 0.55 * page_rect.width:
                band = pymupdf.Rect(page_rect.x0 + 18, 0, page_rect.x1 - 18, page_rect.y1)
            else:
                band = pymupdf.Rect(max(page_rect.x0, cap.x0 - 26), 0,
                                    min(page_rect.x1, cap.x1 + 26), page_rect.y1)

            # Graphics above the caption, inside the band, not too far up.
            region: pymupdf.Rect | None = None
            for r in _graphics_rects(page):
                cx = (r.x0 + r.x1) / 2
                if not (band.x0 <= cx <= band.x1):
                    continue
                if r.y1 > cap.y0 + 4:          # not above the caption
                    continue
                if r.y0 < cap.y0 - MAX_UP_PT:  # too far up
                    continue
                region = pymupdf.Rect(r) if region is None else region | r
            if region is None or region.height < MIN_REGION_PT:
                # Fallback: fixed-height crop above the caption.
                region = pymupdf.Rect(band.x0, max(0.0, cap.y0 - 330),
                                      band.x1, cap.y0 - 2)
            region = pymupdf.Rect(region + (-6, -6, 6, 6)) & page_rect
            if region.height < MIN_REGION_PT or region.width < 60:
                continue

            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=region)
            if pix.width < 140 or pix.height < 100:
                continue
            name = f"fig{number}.png"
            pix.save(str(out_dir / name))
            caption = " ".join(
                "".join(span["text"] for span in line.get("spans", []))
                for line in cblock.get("lines", [])
            ).strip()
            figures[number] = FigureInfo(
                id=f"fig{number}", number=number, page=pno,
                caption=caption[:240], path=name,
                width_px=pix.width, height_px=pix.height,
            )
            if len(figures) >= MAX_FIGURES:
                break
        if len(figures) >= MAX_FIGURES:
            break
    return [figures[n] for n in sorted(figures)]


def _extract_embedded(doc: pymupdf.Document, out_dir: Path) -> list[FigureInfo]:
    """Fallback: pull large embedded raster images, in document order."""
    out: list[FigureInfo] = []
    seen: set[int] = set()
    idx = 0
    for pno, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            w, h = img[2], img[3]
            if w < 300 or h < 200:
                continue
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                idx += 1
                name = f"figx{idx}.png"
                pix.save(str(out_dir / name))
                out.append(FigureInfo(id=f"figx{idx}", number=idx, page=pno,
                                      caption="", path=name,
                                      width_px=pix.width, height_px=pix.height))
            except Exception:
                continue
            if len(out) >= MAX_FIGURES:
                return out
    return out


def extract_figures(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[FigureInfo]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    try:
        figures = _extract_with_captions(doc, out_dir, dpi)
        if not figures:
            figures = _extract_embedded(doc, out_dir)
    finally:
        doc.close()
    manifest = out_dir / "figures.json"
    manifest.write_text(
        json.dumps([asdict(f) for f in figures], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return figures
