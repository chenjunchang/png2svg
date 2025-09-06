from __future__ import annotations

import argparse
import sys

from .config import load_config
from .io_utils import list_pngs, run_parallel
from .pipeline import process_image


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="png2svg – convert math PNGs to semantic SVG")
    ap.add_argument("--config", type=str, default="config.yaml", help="config yaml path")
    ap.add_argument("--input", type=str, help="input directory containing pngs")
    ap.add_argument("--output", type=str, help="output directory for svg/geojson")
    ap.add_argument("--jobs", type=int, help="parallel jobs")
    ap.add_argument("--single", type=str, help="single file within input dir to process")
    ap.add_argument("--no-yolo", dest="no_yolo", action="store_true", help="disable YOLO symbols")
    ap.add_argument("--no-paddleocr", dest="no_paddleocr", action="store_true", help="disable PaddleOCR")
    ap.add_argument("--no-constraints", dest="no_constraints", action="store_true", help="disable constraint solver")
    ap.add_argument("--dry-run", action="store_true", help="load config and list files without processing")
    return ap


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args)
    files = [args.single] if args.single else list_pngs(cfg.input_dir)

    if args.dry_run:
        print(f"Found {len(files)} png(s) in {cfg.input_dir}")
        for f in files[:10]:
            print(" -", f)
        if len(files) > 10:
            print(" ...")
        sys.exit(0)

    results = run_parallel(files, process_image, cfg, jobs=cfg.jobs)
    failed = [r.path for r in results if not r.ok]
    if failed:
        print(f"[WARN] failed: {len(failed)} items", file=sys.stderr)
        for p in failed:
            print(" -", p, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

