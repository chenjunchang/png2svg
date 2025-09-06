# Repository Guidelines

## Project Structure & Modules
- `png2svg/`: core library — `cli.py` (CLI), `pipeline.py` (orchestrator), `preprocess.py`, `detect_primitives.py`, `detect_symbols.py`, `ocr_text.py`, `topology.py`, `constraints.py`, `svg_writer.py`, `geojson_writer.py`, `io_utils.py`, `config.py`, `__main__.py`.
- `train_symbols/`: dataset converters and YOLO training scripts under `scripts/`.
- `pngs/` sample inputs, `out/` outputs (`.svg`, `.geo.json`), plus `config.yaml`, `config.example.yaml`, `requirements.txt`, `README.md`.

## Build, Run, Develop
- Create env and install deps:
  - Windows: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
  - Unix: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Dry run: `python -m png2svg.cli --dry-run`
- Process directory: `python -m png2svg.cli --config config.yaml` or `--input ./pngs --output ./out`
- Single file: `python -m png2svg.cli --input ./pngs --single demo.png`
- Optional models: place `models/symbols_yolo.onnx` or use `--no-yolo`.

## Coding Style & Naming
- Python 3.10+, use type hints and `@dataclass`; keep functions small and pure where possible.
- Naming: snake_case (functions/vars), PascalCase (classes), UPPER_CASE (constants).
- Use `logging` (respect `config.log_level`); avoid `print` in library code.
- Prefer vectorized NumPy/OpenCV over Python loops in hot paths.

## Testing Guidelines
- No formal suite yet; prefer `pytest` with files under `tests/` named `test_*.py`.
- Use synthetic fixtures and small PNGs in `pngs/` to verify: SVG/GeoJSON creation, element counts, and basic schema.
- For new flags or I/O, add a smoke test invoking `python -m png2svg.cli --dry-run` and real runs on tiny images.

## Commit & Pull Requests
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- PRs must include: rationale/scope, CLI examples used, before/after artifacts (input PNG and generated SVG), and any config changes. Update `README.md` and `config.example.yaml` if behavior/flags change.
- Keep changes focused and minimally invasive; avoid unrelated refactors.

## Configuration Tips
- Optional deps auto-disable (YOLO/PaddleOCR/SciPy). Be explicit when needed: `--no-yolo`, `--no-paddleocr`, `--no-constraints`.
- On Windows, install Tesseract and add it to PATH for OCR fallback.

