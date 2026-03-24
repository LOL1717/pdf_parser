"""
run_papers.py — Batch parse all PDFs in a folder using pdf_parser.py

Usage:
    python run_papers.py                        # reads from ./papers by default
    python run_papers.py --input ./papers --output ./results
    python run_papers.py --input ./papers --skip-images
    python run_papers.py --input ./papers --llm-only   # only save llm_context txt files
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make sure pdf_parser.py is findable (same directory as this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdf_parser import PaperParser


def parse_folder(
    input_dir: str,
    output_dir: str,
    skip_images: bool = False,
    llm_only: bool = False,
    max_llm_chars: int = 80_000,
):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    pdfs = sorted(input_path.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in: {input_path.resolve()}")
        sys.exit(1)

    print(f"\nFound {len(pdfs)} PDF(s) in '{input_path.resolve()}'")
    print(f"Output  → '{output_path.resolve()}'\n")
    print("─" * 60)

    summary_rows = []
    failed       = []

    for i, pdf_path in enumerate(pdfs, start=1):
        stem = pdf_path.stem
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")

        # One sub-folder per paper
        paper_out = output_path / stem
        paper_out.mkdir(parents=True, exist_ok=True)

        img_dir = str(paper_out / "figures") if not skip_images else str(paper_out / "_no_images")

        t0 = time.time()
        try:
            parser = PaperParser(str(pdf_path), image_output_dir=img_dir)
            result = parser.parse()
            elapsed = round(time.time() - t0, 1)

            # ── Always save: LLM context text ─────────────────────
            llm_text = result.llm_context(max_chars=max_llm_chars)
            (paper_out / "llm_context.txt").write_text(llm_text, encoding="utf-8")

            if not llm_only:
                # ── Tables → CSV (one file per table) ─────────────
                if result.tables:
                    tables_dir = paper_out / "tables"
                    tables_dir.mkdir(exist_ok=True)
                    for tbl in result.tables:
                        csv_path = tables_dir / f"{tbl.table_id}.csv"
                        csv_path.write_text(tbl.csv_text, encoding="utf-8")

                    # All tables in one Excel workbook
                    parser.export_tables_excel(str(paper_out / "tables_all.xlsx"))

                    # All tables as one markdown file (easy to diff / inspect)
                    md_lines = []
                    for tbl in result.tables:
                        md_lines.append(f"## {tbl.table_id}  —  {tbl.caption or '(no caption)'}")
                        md_lines.append(f"*Page {tbl.page+1} · method={tbl.method} · confidence={tbl.confidence:.2f}*\n")
                        md_lines.append(tbl.markdown)
                        md_lines.append("")
                    (paper_out / "tables_all.md").write_text("\n".join(md_lines), encoding="utf-8")

                # ── Full JSON export ───────────────────────────────
                parser.export_json(str(paper_out / "parsed.json"))

                # ── Sections as plain text ─────────────────────────
                sections_txt = "\n\n".join(
                    f"{'='*4} {h.upper()} {'='*4}\n{b}"
                    for h, b in result.sections.items()
                )
                (paper_out / "sections.txt").write_text(sections_txt, encoding="utf-8")

            summary_rows.append({
                "file":       pdf_path.name,
                "pages":      result.page_count,
                "sections":   len(result.sections),
                "tables":     len(result.tables),
                "images":     len(result.images),
                "warnings":   len(result.warnings),
                "time_s":     elapsed,
                "status":     "ok",
            })

            # Console summary line
            tbl_info = f"{len(result.tables)} tables" if result.tables else "no tables"
            img_info = f"{len(result.images)} images" if not skip_images else "images skipped"
            warn_str = f"  ⚠ {result.warnings}" if result.warnings else ""
            print(f"  ✓ {result.page_count} pages · {tbl_info} · {img_info} · {elapsed}s{warn_str}")

            # Print table details
            for tbl in result.tables:
                print(f"      [{tbl.table_id}] '{tbl.caption[:55]}' "
                      f"— {len(tbl.rows)} rows × {len(tbl.headers)} cols "
                      f"(conf={tbl.confidence:.2f}, {tbl.method})")

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            err_msg = str(e)
            print(f"  ✗ FAILED in {elapsed}s: {err_msg}")
            traceback.print_exc()
            failed.append({"file": pdf_path.name, "error": err_msg})
            summary_rows.append({
                "file": pdf_path.name, "status": "failed",
                "error": err_msg, "time_s": elapsed,
            })

        print()

    # ── Write run summary ──────────────────────────────────────────────
    summary_path = output_path / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "input_dir":    str(input_path.resolve()),
            "output_dir":   str(output_path.resolve()),
            "total_pdfs":   len(pdfs),
            "successful":   len(pdfs) - len(failed),
            "failed":       len(failed),
            "papers":       summary_rows,
        }, f, indent=2)

    # ── Final report ───────────────────────────────────────────────────
    print("═" * 60)
    print(f"Done.  {len(pdfs) - len(failed)}/{len(pdfs)} parsed successfully.")
    print(f"Summary → {summary_path}")
    if failed:
        print(f"\nFailed PDFs:")
        for f in failed:
            print(f"  ✗ {f['file']}: {f['error']}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch parse PDFs with pdf_parser.py")
    ap.add_argument("--input",        default="./papers",  help="Folder containing PDFs (default: ./papers)")
    ap.add_argument("--output",       default="./results", help="Output folder (default: ./results)")
    ap.add_argument("--skip-images",  action="store_true", help="Skip image extraction (faster)")
    ap.add_argument("--llm-only",     action="store_true", help="Only save llm_context.txt per paper")
    ap.add_argument("--max-chars",    type=int, default=80_000, help="Max chars in llm_context.txt")
    args = ap.parse_args()

    parse_folder(
        input_dir    = args.input,
        output_dir   = args.output,
        skip_images  = args.skip_images,
        llm_only     = args.llm_only,
        max_llm_chars= args.max_chars,
    )
