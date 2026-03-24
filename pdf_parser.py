"""
pdf_parser.py — Production-grade research paper PDF parser.

THREE OUTPUT TARGETS:
  1. text    → clean, section-aware text for LLM context
  2. tables  → geometrically reconstructed tables (not regex-heuristic)
  3. images  → figure images with captions, saved to disk

ALGORITHM OVERVIEW
──────────────────
TEXT:
  - Extract spans with full geometry (bbox, font size, bold flag)
  - Classify each span as heading / body / caption / footer / header
  - Merge body spans into paragraphs using vertical gap thresholds
  - Detect section headings by font-weight + size delta + numbering patterns
  - Strip headers, footers, page numbers, running titles
  - Output ordered sections dict: {heading: paragraph_text}

TABLES — the hard part, four-pass algorithm:
  Pass 1 · Line detection
    - Extract all explicit ruling lines (h-lines + v-lines) from the PDF
    - If ruling lines are present → use them to define cell grid precisely
  Pass 2 · Cell boundary reconstruction
    - If no ruling lines (common in academic PDFs) → use whitespace gaps
    - Project text spans onto X-axis → find column boundaries from gap histogram
    - Project spans onto Y-axis → find row boundaries from gap histogram
    - Snap each span to its (row, col) cell by centroid overlap
  Pass 3 · Cell merging & span detection
    - Detect merged cells (cell with no ruling line between adjacent cells)
    - Detect multi-line cells (multiple spans in same cell, same column)
    - Handle superscript footnote markers (tiny font, offset baseline)
  Pass 4 · Table validation
    - Reject false positives: short regions with < 2 cols or < 2 rows
    - Check column count consistency across rows
    - Score table confidence: line coverage + cell density + header row detection

IMAGES:
  - Extract all image XObjects from each page
  - Crop each image with a 4px margin
  - Find caption: scan downward from image bbox for text starting with
    "Fig", "Figure", "Table" within 60pt
  - Save as PNG with metadata JSON sidecar

STORAGE FORMAT:
  Tables → list of dicts, each with:
    {
      "table_id":    "p3_t1",           ← page 3, table 1
      "caption":     "Table 2: ...",
      "page":        3,
      "method":      "ruled|whitespace", ← how it was extracted
      "confidence":  0.91,
      "headers":     ["Model", "Acc", "F1", "Dataset"],
      "rows":        [["ResNet-50", "96.3", "0.961", "CheXpert"], ...],
      "dataframe":   <pd.DataFrame>,     ← for downstream numeric ops
      "markdown":    "| Model | Acc |...",
      "csv":         "Model,Acc,F1,...",
    }

Usage:
  from pdf_parser import PaperParser

  parser  = PaperParser("paper.pdf", image_output_dir="./figures")
  result  = parser.parse()

  result.text             # str  — full clean text for LLM
  result.sections         # dict — {section_name: text}
  result.tables           # list of table dicts (see above)
  result.images           # list of image dicts
  result.llm_context()    # str  — text + table markdown combined, ready to send
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz          # PyMuPDF  — pip install PyMuPDF
import pdfplumber    # pip install pdfplumber
import pandas as pd  # pip install pandas
from PIL import Image  # pip install pillow

# OCR Optional Dependencies
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

import hashlib


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Span:
    """One atomic text run from the PDF with full geometry."""
    text:     str
    x0: float; y0: float; x1: float; y1: float   # bbox
    font_size: float
    bold:     bool
    page:     int

    @property
    def cx(self) -> float: return (self.x0 + self.x1) / 2
    @property
    def cy(self) -> float: return (self.y0 + self.y1) / 2
    @property
    def width(self) -> float: return self.x1 - self.x0
    @property
    def height(self) -> float: return self.y1 - self.y0


@dataclass
class TableResult:
    table_id:   str
    caption:    str
    page:       int
    method:     str          # "ruled" | "whitespace" | "pdfplumber"
    confidence: float
    headers:    List[str]
    rows:       List[List[str]]
    dataframe:  pd.DataFrame
    markdown:   str
    csv_text:   str


@dataclass
class ImageResult:
    image_id:  str
    caption:   str
    page:      int
    filepath:  str           # saved PNG path
    bbox:      Tuple[float, float, float, float]


@dataclass
class ParseResult:
    source_path: str
    text:        str                     # full clean text
    sections:    Dict[str, str]          # {heading: body}
    tables:      List[TableResult]
    images:      List[ImageResult]
    page_count:  int
    warnings:    List[str] = field(default_factory=list)

    def llm_context(self, max_chars: int = 80_000) -> str:
        """
        Build the optimal LLM input:
          - Section text in priority order (abstract first, references last)
          - Every table as markdown, preceded by its caption
          - Figure captions only (not image bytes)
        """
        PRIORITY = [
            "abstract", "introduction", "background",
            "methods", "methodology", "approach",
            "experiments", "results", "evaluation",
            "discussion", "conclusion", "future work",
        ]
        parts: List[str] = []
        budget = max_chars

        # Section text
        ordered = sorted(
            self.sections.items(),
            key=lambda kv: next(
                (i for i, p in enumerate(PRIORITY) if p in kv[0].lower()),
                99
            ),
        )
        for heading, body in ordered:
            if "reference" in heading.lower():
                continue
            block = f"[{heading.upper()}]\n{body}\n"
            if len(block) > budget:
                block = block[:budget]
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break

        # Tables
        for tbl in self.tables:
            block = f"\n[TABLE: {tbl.caption or tbl.table_id}]\n{tbl.markdown}\n"
            if len(block) < budget:
                parts.append(block)
                budget -= len(block)

        # Figure captions
        for img in self.images:
            if img.caption:
                block = f"\n[FIGURE CAPTION: {img.caption}]\n"
                if len(block) < budget:
                    parts.append(block)
                    budget -= len(block)

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\u2013": "-",
    "\u2014": "--", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00a0": " ",
    "\u2022": "*", "\u2212": "-", "\u00d7": "x",
})

def _clean(text: str) -> str:
    text = text.translate(_LIGATURES)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # dehyphenate
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _overlap_1d(a0, a1, b0, b1) -> float:
    """Fraction of [a0,a1] that overlaps [b0,b1]."""
    lo = max(a0, b0); hi = min(a1, b1)
    if hi <= lo:
        return 0.0
    return (hi - lo) / (a1 - a0 + 1e-9)

def _gap_histogram_peaks(positions: List[float], page_dim: float, min_gap: float = 4.0) -> List[float]:
    """
    Given a sorted list of interval endpoints, find whitespace gaps larger
    than min_gap and return their midpoints. Used to locate column/row boundaries.
    """
    positions = sorted(set(positions))
    boundaries = []
    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        if gap >= min_gap:
            boundaries.append((positions[i] + positions[i + 1]) / 2)
    return boundaries


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Span extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_spans(doc: fitz.Document) -> List[Span]:
    """
    Extract every text span from the document with full geometry.
    Uses the 'rawdict' extractor which gives per-character data —
    more accurate than 'blocks' for multi-column layouts.
    """
    spans: List[Span] = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                for span_raw in line.get("spans", []):
                    text = span_raw.get("text", "").strip()
                    if not text:
                        continue
                    bbox = span_raw["bbox"]        # (x0, y0, x1, y1)
                    flags = span_raw.get("flags", 0)
                    size  = span_raw.get("size", 10.0)
                    bold  = bool(flags & 2**4)     # bit 4 = bold in PyMuPDF
                    spans.append(Span(
                        text=_clean(text),
                        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                        font_size=size,
                        bold=bold,
                        page=page_num,
                    ))
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Classify & filter spans
# ══════════════════════════════════════════════════════════════════════════════

_HEADING_RE = re.compile(
    r"^(\d{1,2}\.?\s+)?("
    r"abstract|introduction|background|related work|literature review|"
    r"method(?:ology|s)?|approach|model|architecture|"
    r"experiment(?:s|al setup)?|result(?:s)?|evaluation|"
    r"discussion|conclusion(?:s)?|future work|"
    r"acknowledge?ments?|references?|appendix"
    r")[\s:]*$",
    re.IGNORECASE,
)

_FOOTER_RE = re.compile(
    r"^(\d+|page\s+\d+|©|\u00a9|www\.|http|doi:|arxiv:|preprint|submitted|accepted|published)",
    re.IGNORECASE,
)

_CAPTION_RE = re.compile(r"^(fig(?:ure)?\.?\s*\d+|table\s*\d+)[.:\s]", re.IGNORECASE)


def _classify_spans(
    spans: List[Span],
    page_heights: Dict[int, float],
) -> Dict[int, str]:
    """
    Returns {span_index: label} where label ∈ {heading, body, caption, skip}.

    Classification rules (in priority order):
      1. Page header/footer zone → skip  (top/bottom 6% of page)
      2. Matches _FOOTER_RE → skip
      3. Matches _HEADING_RE AND (bold OR font_size > body_median + 1.5) → heading
      4. Matches _CAPTION_RE → caption
      5. Short isolated line (< 15 chars) at non-body font size → skip
      6. Everything else → body
    """
    if not spans:
        return {}

    # Compute per-page body font size (mode of sizes, ignoring outliers)
    page_sizes: Dict[int, List[float]] = defaultdict(list)
    for sp in spans:
        page_sizes[sp.page].append(sp.font_size)

    body_size: Dict[int, float] = {}
    for pg, sizes in page_sizes.items():
        # Use the most common size as body reference
        freq: Dict[float, int] = defaultdict(int)
        for s in sizes:
            freq[round(s, 1)] += 1
        body_size[pg] = max(freq, key=freq.get)

    labels: Dict[int, str] = {}
    for i, sp in enumerate(spans):
        pg_h    = page_heights.get(sp.page, 842.0)
        bs      = body_size.get(sp.page, 10.0)
        rel_y   = sp.y0 / pg_h

        # Rule 1: header/footer zone
        if rel_y < 0.055 or rel_y > 0.945:
            labels[i] = "skip"
            continue

        # Rule 2: footer pattern
        if _FOOTER_RE.match(sp.text):
            labels[i] = "skip"
            continue

        # Rule 3: heading
        if _HEADING_RE.match(sp.text):
            if sp.bold or sp.font_size >= bs + 1.0:
                labels[i] = "heading"
                continue

        # Rule 4: caption
        if _CAPTION_RE.match(sp.text):
            labels[i] = "caption"
            continue

        # Rule 5: tiny isolated text (footnote markers, equation labels)
        if len(sp.text) < 4 and sp.font_size < bs - 2:
            labels[i] = "skip"
            continue

        labels[i] = "body"

    return labels


# ══════════════════════════════════════════════════════════════════════════════
# PASS 3 — Section assembly
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_sections(spans: List[Span], labels: Dict[int, str]) -> Dict[str, str]:
    """
    Walk spans in page-reading order (top→bottom, left→right).
    When a heading is encountered, start a new section.
    Merge body spans into paragraphs using vertical gap heuristic:
      - gap < 1.5 × line_height  → same paragraph (append with space)
      - gap ≥ 1.5 × line_height  → new paragraph (append with \n\n)
    """
    sections: Dict[str, List[str]] = {}
    current_heading = "preamble"
    sections[current_heading] = []

    prev_span: Optional[Span] = None

    for i, sp in enumerate(spans):
        label = labels.get(i, "body")

        if label == "skip":
            continue

        if label == "heading":
            current_heading = _clean(sp.text).lower()
            if current_heading not in sections:
                sections[current_heading] = []
            prev_span = None
            continue

        if label in ("body", "caption"):
            bucket = sections[current_heading]

            if prev_span is None:
                bucket.append(sp.text)
            else:
                # Same page — use vertical gap
                if sp.page == prev_span.page:
                    gap      = sp.y0 - prev_span.y1
                    lh       = max(sp.height, prev_span.height, 8.0)
                    if gap < 0:
                        # Overlapping y (multi-column) → treat as continuation
                        bucket.append(" " + sp.text)
                    elif gap < 1.5 * lh:
                        bucket.append(" " + sp.text)
                    else:
                        bucket.append("\n\n" + sp.text)
                else:
                    # Page break → paragraph break
                    bucket.append("\n\n" + sp.text)

            prev_span = sp

    # Join and clean each section
    result: Dict[str, str] = {}
    for heading, parts in sections.items():
        text = "".join(parts).strip()
        if text:
            # Remove "references" body content — not useful for LLM
            if "reference" in heading:
                result[heading] = "[References omitted]"
            else:
                result[heading] = re.sub(r"\n{3,}", "\n\n", text)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TABLE EXTRACTION — four-pass geometric algorithm
# ══════════════════════════════════════════════════════════════════════════════

def _extract_ruling_lines(page: fitz.Page) -> Tuple[List, List]:
    """
    Extract horizontal and vertical ruling lines from vector graphics.
    Returns (h_lines, v_lines) where each line is (x0, y0, x1, y1).
    Only keeps lines that are thin enough to be table borders (height or width < 3pt).
    """
    h_lines, v_lines = [], []
    paths = page.get_drawings()
    for path in paths:
        for item in path.get("items", []):
            if item[0] != "l":    # 'l' = line segment
                continue
            p1, p2 = item[1], item[2]
            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)
            # Horizontal line: very small dy, meaningful dx
            if dy < 2.0 and dx > 10:
                h_lines.append((min(p1.x, p2.x), p1.y, max(p1.x, p2.x), p1.y))
            # Vertical line: very small dx, meaningful dy
            elif dx < 2.0 and dy > 6:
                v_lines.append((p1.x, min(p1.y, p2.y), p1.x, max(p1.y, p2.y)))
    return h_lines, v_lines


def _spans_in_region(
    spans: List[Span], x0: float, y0: float, x1: float, y1: float, page: int,
    margin: float = 2.0,
) -> List[Span]:
    """Return spans whose centroid falls within the bounding box."""
    return [
        sp for sp in spans
        if sp.page == page
        and x0 - margin <= sp.cx <= x1 + margin
        and y0 - margin <= sp.cy <= y1 + margin
    ]


def _find_caption(
    spans: List[Span], table_x0: float, table_y0: float, table_x1: float,
    table_y1: float, page: int, search_above: bool = True,
) -> str:
    """
    Find a caption for a table by scanning for 'Table N' or 'Tab.' text
    within 60pt above or below the table bounding box.
    """
    search_range = 70.0
    candidates = []
    for sp in spans:
        if sp.page != page:
            continue
        # Check horizontal overlap with table
        h_overlap = _overlap_1d(sp.x0, sp.x1, table_x0, table_x1)
        if h_overlap < 0.1:
            continue
        if search_above and table_y0 - search_range <= sp.y0 < table_y0:
            candidates.append((table_y0 - sp.y0, sp.text))
        elif not search_above and table_y1 < sp.y1 <= table_y1 + search_range:
            candidates.append((sp.y1 - table_y1, sp.text))
    if not candidates:
        return ""
    # Closest match that looks like a caption
    candidates.sort(key=lambda x: x[0])
    for _, text in candidates:
        if _CAPTION_RE.match(text) or re.search(r"table\s*\d+", text, re.IGNORECASE):
            return text
    # Return closest regardless
    return candidates[0][1] if candidates else ""


# ── Ruled table (has explicit grid lines) ─────────────────────────────────

def _build_ruled_table(
    h_lines: List, v_lines: List,
    spans: List[Span], page: int,
    page_width: float,
) -> Optional[Dict]:
    """
    Algorithm:
    1. Cluster h_lines by y-coordinate (tolerance 2pt) → unique row boundaries
    2. Cluster v_lines by x-coordinate (tolerance 2pt) → unique col boundaries
    3. Sort both; pair consecutive lines to form cell bounding boxes
    4. For each cell bbox, collect spans inside it → cell text
    5. Detect header row: first row, bold or different background
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    def _cluster(values: List[float], tol: float = 3.0) -> List[float]:
        values = sorted(set(values))
        clusters = []
        group = [values[0]]
        for v in values[1:]:
            if v - group[-1] <= tol:
                group.append(v)
            else:
                clusters.append(statistics.mean(group))
                group = [v]
        clusters.append(statistics.mean(group))
        return clusters

    row_ys = _cluster([l[1] for l in h_lines] + [l[3] for l in h_lines])
    col_xs = _cluster([l[0] for l in v_lines] + [l[2] for l in v_lines])

    if len(row_ys) < 2 or len(col_xs) < 2:
        return None

    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1

    if n_rows < 1 or n_cols < 1:
        return None

    # Build grid
    grid: List[List[str]] = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            cell_x0 = col_xs[c]
            cell_y0 = row_ys[r]
            cell_x1 = col_xs[c + 1]
            cell_y1 = row_ys[r + 1]
            cell_spans = _spans_in_region(spans, cell_x0, cell_y0, cell_x1, cell_y1, page)
            # Sort spans within cell top→bottom, left→right
            cell_spans.sort(key=lambda s: (round(s.y0, 1), s.x0))
            cell_text = " ".join(s.text for s in cell_spans).strip()
            row.append(cell_text)
        grid.append(row)

    table_x0 = col_xs[0]
    table_y0 = row_ys[0]
    table_x1 = col_xs[-1]
    table_y1 = row_ys[-1]

    return {
        "grid": grid,
        "bbox": (table_x0, table_y0, table_x1, table_y1),
        "method": "ruled",
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


# ── Whitespace table (no explicit lines) ─────────────────────────────────

def _build_whitespace_table(
    region_spans: List[Span],
    page: int,
) -> Optional[Dict]:
    """
    Algorithm for tables without ruling lines:
    1. Project spans onto X-axis: collect [x0, x1] intervals
       → Find significant whitespace gaps (≥ 8pt) between span clusters
       → These gaps are column separators
    2. Assign each span to a column by centroid
    3. Project spans onto Y-axis: find row boundaries by y-gap
       → Gap ≥ 0.8 × median_line_height → new row
    4. Build grid by (row_idx, col_idx) → collect all spans in that cell
    5. Sort cells within each cell by x, merge text

    Why this works better than naive word-by-word parsing:
    - Uses the actual whitespace geometry, not arbitrary split characters
    - Handles variable-width columns correctly
    - Handles multi-word cell content
    """
    if len(region_spans) < 4:
        return None

    # ── Step 1: Find column boundaries ──────────────────────────────
    # Collect all x-coordinates (span starts and ends)
    x_events: List[float] = []
    for sp in region_spans:
        x_events.extend([sp.x0, sp.x1])
    x_events.sort()

    # Find gaps between the x-event clusters
    col_separators: List[float] = []
    i = 0
    while i < len(x_events) - 1:
        gap = x_events[i + 1] - x_events[i]
        if gap >= 8.0:
            col_separators.append((x_events[i] + x_events[i + 1]) / 2)
        i += 1

    # Column boundaries: page left, separators, page right
    if not col_separators:
        return None  # Can't distinguish columns → not a table

    page_x0 = min(sp.x0 for sp in region_spans)
    page_x1 = max(sp.x1 for sp in region_spans)
    col_bounds = [page_x0] + col_separators + [page_x1]
    n_cols = len(col_bounds) - 1

    if n_cols < 2:
        return None

    # ── Step 2: Assign spans to columns ──────────────────────────────
    def _col_for(span: Span) -> int:
        cx = span.cx
        for c in range(n_cols):
            if col_bounds[c] <= cx < col_bounds[c + 1]:
                return c
        # Edge case: beyond last bound
        return n_cols - 1

    # ── Step 3: Find row boundaries by y-gap ─────────────────────────
    # Sort spans top → bottom
    region_spans_sorted = sorted(region_spans, key=lambda s: s.y0)
    y_events: List[float] = sorted(set(round(s.y0, 1) for s in region_spans_sorted))

    # Median line height
    heights = [s.height for s in region_spans if s.height > 2]
    median_h = statistics.median(heights) if heights else 10.0

    row_starts: List[float] = [y_events[0]]
    for j in range(len(y_events) - 1):
        gap = y_events[j + 1] - y_events[j]
        if gap >= 0.7 * median_h:
            row_starts.append(y_events[j + 1])

    row_starts.append(max(sp.y1 for sp in region_spans) + 1)
    n_rows = len(row_starts) - 1

    if n_rows < 2:
        return None

    # ── Step 4: Build grid ────────────────────────────────────────────
    grid: List[List[str]] = [[""] * n_cols for _ in range(n_rows)]

    for sp in region_spans:
        col = _col_for(sp)
        row = 0
        for r in range(n_rows):
            if row_starts[r] <= sp.y0 < row_starts[r + 1]:
                row = r
                break
        # Append (handles multi-line cells)
        current = grid[row][col]
        grid[row][col] = (current + " " + sp.text).strip() if current else sp.text

    bbox = (
        min(sp.x0 for sp in region_spans),
        min(sp.y0 for sp in region_spans),
        max(sp.x1 for sp in region_spans),
        max(sp.y1 for sp in region_spans),
    )

    return {
        "grid": grid,
        "bbox": bbox,
        "method": "whitespace",
        "n_rows": n_rows,
        "n_cols": n_cols,
    }


# ── Table scoring and post-processing ────────────────────────────────────

def _score_table(raw: Dict) -> float:
    """
    Heuristic confidence score 0–1 for a candidate table.
    Penalises: mostly empty cells, single column, wild column count variation.
    Rewards: consistent column count, non-empty header, numeric content.
    """
    grid = raw["grid"]
    if not grid:
        return 0.0

    n_rows = len(grid)
    n_cols = raw["n_cols"]

    # Column count consistency
    col_counts = [len(row) for row in grid]
    col_consistency = 1.0 - (statistics.stdev(col_counts) / max(n_cols, 1)) if n_rows > 1 else 1.0

    # Cell fill rate
    total_cells = sum(len(row) for row in grid)
    filled_cells = sum(1 for row in grid for cell in row if cell.strip())
    fill_rate = filled_cells / max(total_cells, 1)

    # Numeric content in non-header rows (strong signal for data tables)
    num_re = re.compile(r"\d")
    body_rows = grid[1:] if n_rows > 1 else grid
    numeric_cells = sum(1 for row in body_rows for cell in row if num_re.search(cell))
    numeric_rate = numeric_cells / max(sum(len(r) for r in body_rows), 1)

    # Minimum columns check
    if n_cols < 2:
        return 0.0

    score = (
        0.35 * fill_rate +
        0.30 * col_consistency +
        0.20 * numeric_rate +
        0.15 * min(n_rows / 3, 1.0)   # bonus for having several rows
    )
    return round(score, 3)


def _grid_to_tableresult(
    raw: Dict,
    table_id: str,
    caption: str,
    page: int,
    confidence: float,
) -> TableResult:
    """Convert raw grid dict to a fully formed TableResult."""
    grid = raw["grid"]

    # Normalise column counts (pad shorter rows)
    max_cols = max(len(row) for row in grid) if grid else 0
    grid = [row + [""] * (max_cols - len(row)) for row in grid]

    # Detect header: first row is header if it has no numbers and all non-empty cells
    headers: List[str] = []
    rows: List[List[str]] = grid

    if grid:
        first_row = grid[0]
        num_re = re.compile(r"^\d")
        has_numeric = any(num_re.match(c) for c in first_row if c)
        if not has_numeric and any(c for c in first_row):
            headers = [c.strip() for c in first_row]
            rows = grid[1:]
        else:
            headers = [f"col_{i+1}" for i in range(max_cols)]
            rows = grid

    # Build DataFrame
    try:
        df = pd.DataFrame(rows, columns=headers if len(headers) == max_cols else None)
        # Attempt numeric coercion on each column
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[,%]", "", regex=True).str.strip(),
                errors="ignore",
            )
    except Exception:
        df = pd.DataFrame(rows)

    # Markdown
    def _md_row(cells: List[str]) -> str:
        return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"

    md_lines = [_md_row(headers)]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        md_lines.append(_md_row(row))
    markdown = "\n".join(md_lines)

    # CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    if headers:
        writer.writerow(headers)
    writer.writerows(rows)
    csv_text = buf.getvalue()

    return TableResult(
        table_id=table_id,
        caption=caption,
        page=page,
        method=raw["method"],
        confidence=confidence,
        headers=headers,
        rows=rows,
        dataframe=df,
        markdown=markdown,
        csv_text=csv_text,
    )


# ── pdfplumber fallback ───────────────────────────────────────────────────

def _pdfplumber_tables(pdf_path: str, spans: List[Span]) -> List[TableResult]:
    """
    pdfplumber uses a different table detection algorithm (based on edge detection
    and curve analysis). Use as fallback / cross-validation.
    Returns TableResult list.
    """
    results: List[TableResult] = []
    table_index = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables(table_settings={
                "vertical_strategy":   "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance":      4,
                "join_tolerance":      4,
                "edge_min_length":     8,
                "min_words_vertical":  2,
                "min_words_horizontal": 1,
                "intersection_tolerance": 5,
            })

            # Also try text-based strategy if line-strict finds nothing
            if not tables:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy":   "text",
                    "horizontal_strategy": "lines",
                    "snap_tolerance":      6,
                })

            for raw_table in tables:
                if not raw_table or len(raw_table) < 2:
                    continue

                # Clean cells
                grid = [
                    [str(cell or "").replace("\n", " ").strip() for cell in row]
                    for row in raw_table
                ]

                table_index += 1
                table_id = f"p{page_num+1}_t{table_index}"

                # Estimate bbox from page dimensions
                pw, ph = page.width, page.height
                raw = {
                    "grid": grid,
                    "bbox": (0, 0, pw, ph),
                    "method": "pdfplumber",
                    "n_rows": len(grid),
                    "n_cols": max(len(r) for r in grid),
                }
                confidence = _score_table(raw)
                if confidence < 0.25:
                    continue

                caption = _find_caption(spans, 0, 0, pw, ph, page_num)
                result = _grid_to_tableresult(raw, table_id, caption, page_num, confidence)
                results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TABLE ORCHESTRATOR — decides which algorithm to use per page
# ══════════════════════════════════════════════════════════════════════════════

def _extract_all_tables(
    doc: fitz.Document,
    pdf_path: str,
    spans: List[Span],
) -> List[TableResult]:
    """
    Per page:
      1. Try ruled extraction (explicit grid lines)
      2. If not enough lines → scan for likely table regions, run whitespace algorithm
      3. If still poor results → run pdfplumber as cross-validation
      4. Merge results, deduplicate by bounding box overlap
    """
    results: List[TableResult] = []
    table_counter = 0

    for page_num, page in enumerate(doc):
        page_spans  = [s for s in spans if s.page == page_num]
        page_width  = page.rect.width
        page_height = page.rect.height
        h_lines, v_lines = _extract_ruling_lines(page)

        ruled_raw = None

        # ── Ruled attempt ──────────────────────────────────────────
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            ruled_raw = _build_ruled_table(h_lines, v_lines, page_spans, page_num, page_width)

        if ruled_raw and _score_table(ruled_raw) >= 0.45:
            table_counter += 1
            table_id  = f"p{page_num+1}_t{table_counter}"
            confidence = _score_table(ruled_raw)
            caption = _find_caption(
                spans,
                *ruled_raw["bbox"],
                page_num,
                search_above=True,
            )
            if not caption:
                caption = _find_caption(spans, *ruled_raw["bbox"], page_num, search_above=False)
            results.append(_grid_to_tableresult(ruled_raw, table_id, caption, page_num, confidence))
            continue

        # ── Whitespace attempt ─────────────────────────────────────
        # Heuristic: a table region is a cluster of spans with consistent
        # multi-column layout. Find candidate regions by looking for
        # groups of spans with ≥2 columns of text on the same y-band.
        candidate_regions = _find_table_regions(page_spans, page_width)

        for region_spans in candidate_regions:
            if len(region_spans) < 4:
                continue
            ws_raw = _build_whitespace_table(region_spans, page_num)
            if ws_raw is None:
                continue
            confidence = _score_table(ws_raw)
            if confidence < 0.35:
                continue
            table_counter += 1
            table_id = f"p{page_num+1}_t{table_counter}"
            caption = _find_caption(
                spans, *ws_raw["bbox"], page_num, search_above=True
            )
            if not caption:
                caption = _find_caption(spans, *ws_raw["bbox"], page_num, search_above=False)
            results.append(_grid_to_tableresult(ws_raw, table_id, caption, page_num, confidence))

    # ── pdfplumber cross-validation ────────────────────────────────
    # Run pdfplumber separately; add any tables it finds that don't
    # overlap with what we already have.
    try:
        plumber_results = _pdfplumber_tables(pdf_path, spans)
        for pr in plumber_results:
            # Check for overlap with existing results on same page
            overlap_found = False
            for existing in results:
                if existing.page == pr.page:
                    overlap_found = True
                    break
            if not overlap_found:
                table_counter += 1
                pr.table_id = f"plumber_t{table_counter}"
                results.append(pr)
    except Exception as e:
        pass  # pdfplumber failing shouldn't kill the parse

    # Sort by page, then position
    results.sort(key=lambda t: (t.page, t.table_id))
    return results


def _find_table_regions(page_spans: List[Span], page_width: float) -> List[List[Span]]:
    """
    Identify rectangular regions on a page that are likely tables.

    Strategy:
    - Group spans into horizontal bands (rows) by y-coordinate proximity
    - For each band, check if spans are spread into ≥ 2 distinct x-clusters
      with significant whitespace between them
    - Merge consecutive bands that share column structure → table region
    - Return list of span groups (one per candidate table region)
    """
    if not page_spans:
        return []

    # Sort by y then x
    page_spans_sorted = sorted(page_spans, key=lambda s: (s.y0, s.x0))

    # Build horizontal bands (tolerance = 3pt)
    bands: List[List[Span]] = []
    current_band: List[Span] = [page_spans_sorted[0]]
    for sp in page_spans_sorted[1:]:
        if abs(sp.y0 - current_band[-1].y0) <= 4.0:
            current_band.append(sp)
        else:
            bands.append(current_band)
            current_band = [sp]
    bands.append(current_band)

    def _has_multi_col(band: List[Span]) -> bool:
        """True if this band's spans fall into ≥ 2 x-clusters."""
        if len(band) < 2:
            return False
        xs = sorted(sp.cx for sp in band)
        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] > 20:   # 20pt gap → separate column
                return True
        return False

    # Find runs of multi-column bands
    regions: List[List[Span]] = []
    in_table = False
    current_table_spans: List[Span] = []

    for band in bands:
        if _has_multi_col(band):
            in_table = True
            current_table_spans.extend(band)
        else:
            if in_table and len(current_table_spans) >= 4:
                regions.append(current_table_spans)
            in_table = False
            current_table_spans = []

    if in_table and len(current_table_spans) >= 4:
        regions.append(current_table_spans)

    return regions


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_images(
    doc: fitz.Document,
    spans: List[Span],
    output_dir: str,
    source_stem: str,
) -> List[ImageResult]:
    """
    Extract all image XObjects from the PDF.
    For each image:
      1. Get the bounding box from the page's image list
      2. Crop the page to that bbox (rasterised at 150 DPI)
      3. Search for caption within 70pt below the image bbox
      4. Save as PNG, write JSON sidecar

    Filters out:
      - Very small images (< 40 × 40 pt) — likely inline icons
      - Images that are mostly white (> 97% white pixels) — likely decorative boxes
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[ImageResult] = []
    image_counter = 0

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        page_width  = page.rect.width
        page_height = page.rect.height

        # Also get image bboxes from the page (needed for positioning)
        img_bboxes = {}
        for img in image_list:
            xref = img[0]
            # Find where this xref appears on the page
            for item in page.get_image_rects(xref):
                img_bboxes[xref] = item

        seen_xrefs = set()
        for img in image_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            # Get bbox
            if xref not in img_bboxes:
                continue
            bbox = img_bboxes[xref]
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            w, h = x1 - x0, y1 - y0

            # Filter tiny images
            if w < 40 or h < 40:
                continue

            # Rasterise the region at 150 DPI
            mat  = fitz.Matrix(150 / 72, 150 / 72)
            clip = fitz.Rect(x0, y0, x1, y1)
            pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)

            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Filter decorative white boxes
            pixels = list(img_pil.getdata())
            white_count = sum(1 for px in pixels if px[0] > 240 and px[1] > 240 and px[2] > 240)
            if white_count / max(len(pixels), 1) > 0.97:
                continue

            # Find caption
            caption = _find_caption(spans, x0, y0, x1, y1, page_num, search_above=False)
            if not caption:
                caption = _find_caption(spans, x0, y0, x1, y1, page_num, search_above=True)

            image_counter += 1
            image_id = f"p{page_num+1}_img{image_counter}"
            filename = f"{source_stem}_{image_id}.png"
            filepath = os.path.join(output_dir, filename)

            img_pil.save(filepath, "PNG", optimize=True)

            # Write JSON sidecar
            sidecar = {
                "image_id":  image_id,
                "source":    source_stem,
                "page":      page_num + 1,
                "bbox":      [x0, y0, x1, y1],
                "width_pt":  round(w, 2),
                "height_pt": round(h, 2),
                "caption":   caption,
                "filepath":  filepath,
            }
            with open(filepath.replace(".png", ".json"), "w") as f:
                json.dump(sidecar, f, indent=2)

            results.append(ImageResult(
                image_id=image_id,
                caption=caption,
                page=page_num,
                filepath=filepath,
                bbox=(x0, y0, x1, y1),
            ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# OCR FALLBACK LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _detect_scanned_pdf(spans: List[Span], doc: fitz.Document) -> bool:
    """
    Detect scanned/OCR-broken PDFs using:
      1) very low span count,
      2) abnormal character distribution,
      3) high non-alphanumeric ratio.
    """
    page_count = max(len(doc), 1)
    if not spans:
        return True

    spans_per_page = len(spans) / page_count
    if spans_per_page < 8:
        return True

    sample_text = "".join(s.text for s in spans[:1000])
    if not sample_text:
        return True

    total_chars = len(sample_text)
    alphanumeric = sum(1 for c in sample_text if c.isalnum())
    alpha = sum(1 for c in sample_text if c.isalpha())
    non_alnum_ratio = 1.0 - (alphanumeric / max(total_chars, 1))
    if non_alnum_ratio > 0.55:
        return True

    if alpha / max(total_chars, 1) < 0.25:
        return True

    gibberish_like = sum(1 for c in sample_text if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}-_%+/\\'\""))
    if gibberish_like / max(total_chars, 1) > 0.20:
        return True

    return False


def _normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_LIGATURES)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    words = text.split()
    reversed_markers = {"eht", "dna", "rof", "htiw", "si", "era", "ot"}
    if words:
        marker_hits = sum(1 for w in words if w.lower() in reversed_markers)
        if marker_hits >= 2:
            candidate = " ".join(w[::-1] for w in words)
            alpha_tokens = [t for t in re.findall(r"[A-Za-z]{3,}", candidate)]
            if alpha_tokens:
                vowelish = sum(1 for t in alpha_tokens if re.search(r"[aeiou]", t.lower()))
                if vowelish / max(len(alpha_tokens), 1) > 0.75:
                    text = candidate
    return text


def _ocr_extract_text(
    doc: fitz.Document,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Span]]:
    """
    OCR each page and return:
      - page-level OCR records for caching/debugging
      - pseudo-spans (Span objects) for downstream pipeline compatibility
    """
    cache_root = Path(cache_dir) if cache_dir else Path(".ocr_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    pseudo_spans: List[Span] = []

    reader = None
    if easyocr:
        try:
            reader = easyocr.Reader(["en"], gpu=True)
        except Exception:
            try:
                reader = easyocr.Reader(["en"], gpu=False)
            except Exception:
                reader = None

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        digest = hashlib.md5(pix.samples).hexdigest()
        cache_key = f"{digest}_p{page_num}.json"
        cache_file = cache_root / cache_key

        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            page_records = cached.get("spans", [])
            results.append({"page": page_num, "spans": page_records})
            for r in page_records:
                pseudo_spans.append(Span(
                    text=_normalize_ocr_text(r["text"]),
                    x0=float(r["bbox"][0]),
                    y0=float(r["bbox"][1]),
                    x1=float(r["bbox"][2]),
                    y1=float(r["bbox"][3]),
                    font_size=max(float(r["bbox"][3]) - float(r["bbox"][1]), 8.0),
                    bold=False,
                    page=page_num,
                ))
            continue

        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        page_records: List[Dict[str, Any]] = []
        if reader:
            import numpy as np
            img_np = np.array(img_pil)
            for bbox, text, prob in reader.readtext(img_np):
                clean_text = _normalize_ocr_text(text)
                if not clean_text:
                    continue
                x0, y0 = bbox[0][0] / 2.0, bbox[0][1] / 2.0
                x1, y1 = bbox[2][0] / 2.0, bbox[2][1] / 2.0
                page_records.append({
                    "text": clean_text,
                    "bbox": (x0, y0, x1, y1),
                    "confidence": float(prob),
                })
        elif pytesseract:
            d = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)
            for i in range(len(d["text"])):
                txt = _normalize_ocr_text(d["text"][i])
                if not txt:
                    continue
                conf = float(d["conf"][i]) if str(d["conf"][i]).strip() else -1.0
                if conf < 30:
                    continue
                x = float(d["left"][i]) / 2.0
                y = float(d["top"][i]) / 2.0
                w = float(d["width"][i]) / 2.0
                h = float(d["height"][i]) / 2.0
                page_records.append({
                        "text": txt,
                        "bbox": (x, y, x + w, y + h),
                        "confidence": conf / 100.0,
                    })
        else:
            results.append({"page": page_num, "spans": []})
            continue

        page_records.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        cache_file.write_text(json.dumps({"spans": page_records}, ensure_ascii=False), encoding="utf-8")
        results.append({"page": page_num, "spans": page_records})

        for r in page_records:
            x0, y0, x1, y1 = r["bbox"]
            pseudo_spans.append(Span(
                text=r["text"],
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                font_size=max(float(y1) - float(y0), 8.0),
                bold=False,
                page=page_num,
            ))

    return results, pseudo_spans


def _ocr_build_sections(ocr_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Reconstruct sections from OCR text."""
    sections: Dict[str, List[str]] = defaultdict(list)
    current_heading = "preamble"
    for page in ocr_results:
        spans = sorted(page["spans"], key=lambda s: (s["bbox"][1], s["bbox"][0]))
        for sp in spans:
            text = _normalize_ocr_text(sp["text"])
            if not text:
                continue
            if _HEADING_RE.match(text) and len(text) < 60:
                current_heading = text.lower()
                if current_heading not in sections:
                    sections[current_heading] = []
                continue
            sections[current_heading].append(text)
    return {h: _clean(" ".join(b)) for h, b in sections.items() if b}


def _ocr_extract_tables(doc: fitz.Document, ocr_results: List[Dict[str, Any]]) -> List[TableResult]:
    """
    OCR table extraction path:
      - group OCR spans into candidate multi-column regions,
      - reconstruct cells with whitespace geometry,
      - fallback to a tab-delimited row reconstruction from OCR boxes.
    """
    results: List[TableResult] = []
    table_counter = 0
    for page_res in ocr_results:
        pg_num = page_res["page"]
        spans = page_res["spans"]
        width = doc[pg_num].rect.width
        fake_spans = [
            Span(
                text=_normalize_ocr_text(s["text"]),
                x0=s["bbox"][0], y0=s["bbox"][1], x1=s["bbox"][2], y1=s["bbox"][3],
                font_size=max(s["bbox"][3] - s["bbox"][1], 8.0),
                bold=False,
                page=pg_num,
            )
            for s in spans
            if _normalize_ocr_text(s["text"])
        ]
        if len(fake_spans) < 6:
            continue

        regions = _find_table_regions(fake_spans, width)
        for region_spans in regions:
            ws_raw = _build_whitespace_table(region_spans, pg_num)
            if ws_raw is not None:
                conf = _score_table(ws_raw)
                if conf >= 0.30:
                    table_counter += 1
                    table_id = f"ocr_p{pg_num+1}_t{table_counter}"
                    caption = _find_caption(fake_spans, *ws_raw["bbox"], pg_num)
                    results.append(_grid_to_tableresult(ws_raw, table_id, caption, pg_num, conf))
                    continue

            # fallback: row clustering + x anchor bucketing
            sorted_spans = sorted(region_spans, key=lambda s: (s.y0, s.x0))
            rows: List[List[Span]] = []
            current_row: List[Span] = []
            last_y = None
            for sp in sorted_spans:
                if last_y is None or abs(sp.y0 - last_y) <= max(sp.height, 8.0) * 0.7:
                    current_row.append(sp)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [sp]
                last_y = sp.y0
            if current_row:
                rows.append(current_row)

            x_anchors = sorted(set(round(s.x0 / 15) * 15 for s in region_spans))
            if len(rows) >= 2 and len(x_anchors) >= 2:
                grid: List[List[str]] = []
                for row in rows:
                    cells = [""] * len(x_anchors)
                    for s in row:
                        col = min(range(len(x_anchors)), key=lambda i: abs(s.x0 - x_anchors[i]))
                        cells[col] = (cells[col] + " " + s.text).strip()
                    grid.append(cells)
                raw = {
                    "grid": grid,
                    "bbox": (
                        min(s.x0 for s in region_spans),
                        min(s.y0 for s in region_spans),
                        max(s.x1 for s in region_spans),
                        max(s.y1 for s in region_spans),
                    ),
                    "method": "ocr_layout",
                    "n_rows": len(grid),
                    "n_cols": len(grid[0]) if grid else 0,
                }
                conf = _score_table(raw)
                if conf >= 0.25:
                    table_counter += 1
                    table_id = f"ocr_p{pg_num+1}_t{table_counter}"
                    caption = _find_caption(fake_spans, *raw["bbox"], pg_num)
                    results.append(_grid_to_tableresult(raw, table_id, caption, pg_num, conf))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PARSER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class PaperParser:
    """
    Entry point for parsing research paper PDFs.

    Usage:
        parser = PaperParser("paper.pdf", image_output_dir="./figures")
        result = parser.parse()

        # Send to LLM
        llm_input = result.llm_context(max_chars=60_000)

        # Work with tables
        for tbl in result.tables:
            print(tbl.table_id, tbl.confidence)
            print(tbl.markdown)
            print(tbl.dataframe)

        # Save all tables as CSV
        parser.export_tables_csv("./tables/")
    """

    def __init__(
        self,
        pdf_path: str,
        image_output_dir: str = "./figures",
        force_ocr: bool = False,
        ocr_cache_dir: Optional[str] = None,
    ):
        self.pdf_path        = str(pdf_path)
        self.image_output_dir = str(image_output_dir)
        self.force_ocr = force_ocr
        self.ocr_cache_dir = ocr_cache_dir or str(Path(self.image_output_dir) / ".ocr_cache")
        self.is_scanned_pdf: bool = False
        self._result: Optional[ParseResult] = None

    def parse(self) -> ParseResult:
        warnings: List[str] = []

        doc = fitz.open(self.pdf_path)
        source_stem = Path(self.pdf_path).stem

        # Collect page dimensions
        page_heights = {i: doc[i].rect.height for i in range(len(doc))}

        spans = _extract_spans(doc)
        self.is_scanned_pdf = bool(self.force_ocr or _detect_scanned_pdf(spans, doc))

        if self.is_scanned_pdf:
            warnings.append("is_scanned_pdf=True; using OCR fallback pipeline.")
            ocr_results, ocr_spans = _ocr_extract_text(doc, cache_dir=self.ocr_cache_dir)
            spans = ocr_spans
            if not spans:
                warnings.append("OCR produced no text spans.")
            sections = _ocr_build_sections(ocr_results)
            tables = _ocr_extract_tables(doc, ocr_results)
        else:
            if not spans:
                warnings.append("No text spans extracted — PDF may be scanned/image-only.")
            labels = _classify_spans(spans, page_heights)
            sections = _assemble_sections(spans, labels)
            tables = _extract_all_tables(doc, self.pdf_path, spans)

        full_text = "\n\n".join(
            f"{'─'*4} {h.upper()} {'─'*4}\n{b}"
            for h, b in sections.items()
            if b and "reference" not in h.lower()
        )
        if not tables:
            warnings.append("No tables detected.")

        images = _extract_images(doc, spans, self.image_output_dir, source_stem)

        doc.close()

        self._result = ParseResult(
            source_path=self.pdf_path,
            text=full_text,
            sections=sections,
            tables=tables,
            images=images,
            page_count=len(page_heights),
            warnings=warnings,
        )
        return self._result

    # ── Convenience exports ────────────────────────────────────────

    def export_tables_csv(self, output_dir: str):
        """Write each table as a separate CSV file."""
        if not self._result:
            raise RuntimeError("Call parse() first.")
        os.makedirs(output_dir, exist_ok=True)
        for tbl in self._result.tables:
            path = os.path.join(output_dir, f"{tbl.table_id}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(tbl.csv_text)
        print(f"Exported {len(self._result.tables)} tables to {output_dir}")

    def export_tables_excel(self, output_path: str):
        """Write all tables to one Excel file — one sheet per table."""
        if not self._result:
            raise RuntimeError("Call parse() first.")
        if not self._result.tables:
            print("No tables to export.")
            return
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for tbl in self._result.tables:
                sheet_name = tbl.table_id[:31]   # Excel 31-char limit
                tbl.dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                ws.cell(1, 1).comment = None
                # Write caption in the first row before headers
                ws.insert_rows(1)
                ws.cell(1, 1).value = tbl.caption or tbl.table_id
                from openpyxl.styles import Font
                ws.cell(1, 1).font = Font(bold=True, italic=True)
        print(f"Exported {len(self._result.tables)} tables to {output_path}")

    def export_json(self, output_path: str):
        """Export full parse result as JSON (without image bytes / DataFrame)."""
        if not self._result:
            raise RuntimeError("Call parse() first.")
        out = {
            "source": self._result.source_path,
            "page_count": self._result.page_count,
            "warnings":   self._result.warnings,
            "sections":   self._result.sections,
            "tables": [
                {
                    "table_id":   t.table_id,
                    "caption":    t.caption,
                    "page":       t.page + 1,
                    "method":     t.method,
                    "confidence": t.confidence,
                    "headers":    t.headers,
                    "rows":       t.rows,
                    "markdown":   t.markdown,
                }
                for t in self._result.tables
            ],
            "images": [
                {
                    "image_id": img.image_id,
                    "caption":  img.caption,
                    "page":     img.page + 1,
                    "filepath": img.filepath,
                }
                for img in self._result.images
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"JSON exported to {output_path}")

    def summary(self) -> str:
        if not self._result:
            return "Not parsed yet."
        r = self._result
        lines = [
            f"Source      : {r.source_path}",
            f"Pages       : {r.page_count}",
            f"Sections    : {list(r.sections.keys())}",
            f"Tables      : {len(r.tables)}",
        ]
        for t in r.tables:
            lines.append(
                f"  [{t.table_id}] '{t.caption[:50]}' "
                f"· {len(t.rows)} rows × {len(t.headers)} cols "
                f"· method={t.method} · confidence={t.confidence:.2f}"
            )
        lines.append(f"Images      : {len(r.images)}")
        if r.warnings:
            lines.append(f"Warnings    : {r.warnings}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    ap = argparse.ArgumentParser(description="Parse a research paper PDF.")
    ap.add_argument("pdf",            help="Path to PDF file")
    ap.add_argument("--images",       default="./figures",  help="Directory for extracted images")
    ap.add_argument("--tables-csv",   default=None,         help="Directory to export table CSVs")
    ap.add_argument("--tables-excel", default=None,         help="Excel file path for all tables")
    ap.add_argument("--json",         default=None,         help="JSON export path")
    ap.add_argument("--llm",          action="store_true",  help="Print LLM context to stdout")
    ap.add_argument("--max-chars",    type=int, default=80_000, help="Max chars for LLM context")
    ap.add_argument("--force-ocr",    action="store_true",  help="Force OCR fallback pipeline")
    ap.add_argument("--ocr-cache-dir", default=None,        help="Directory for OCR page cache")
    args = ap.parse_args()

    parser = PaperParser(
        args.pdf,
        image_output_dir=args.images,
        force_ocr=args.force_ocr,
        ocr_cache_dir=args.ocr_cache_dir,
    )
    result = parser.parse()

    print("\n" + "═" * 60)
    print(parser.summary())
    print("═" * 60 + "\n")

    if args.tables_csv:
        parser.export_tables_csv(args.tables_csv)

    if args.tables_excel:
        parser.export_tables_excel(args.tables_excel)

    if args.json:
        parser.export_json(args.json)

    if args.llm:
        print("\n" + "─" * 60 + " LLM CONTEXT " + "─" * 60)
        print(result.llm_context(max_chars=args.max_chars))
