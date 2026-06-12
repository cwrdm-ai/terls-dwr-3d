"""Batch-process a folder of L2B polar files through the pipeline.

Each file runs in its own subprocess (a crash in one file cannot kill the
batch, and VTK/render memory is fully released between files). Files whose
gridded cube already exists in output/ are skipped, so an interrupted batch
resumes where it left off.

Usage:
    python src/batch_pipeline.py <folder> [options]

Options:
    --pattern GLOB   which files to pick (default: *L2B_STD.nc)
    --render STYLE   extra render per cube: mosdac | smooth | imd | none
                     (default: mosdac; pipeline itself already saves the
                     volumetric render)
    --force          reprocess even if the gridded cube already exists

Examples:
    python src/batch_pipeline.py "data/Jun26_182862"
    python src/batch_pipeline.py "data/Jun26_182954" --render smooth
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUTPUT_DIR = SRC.parent / "output"

RENDER_SCRIPTS = {
    "mosdac": SRC / "render_mosdac_exact.py",
    "smooth": SRC / "render_stacked_smooth.py",
    "imd": SRC / "render_imd_exact.py",
}


def run_one(script: Path, arg: Path) -> tuple[bool, str]:
    """Run a pipeline/render script on one file; return (ok, tail of output)."""
    proc = subprocess.run(
        [sys.executable, str(script), str(arg)],
        capture_output=True, text=True,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
    return proc.returncode == 0, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--pattern", default="*L2B_STD.nc")
    ap.add_argument("--render", choices=[*RENDER_SCRIPTS, "none"], default="mosdac")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = sorted(args.folder.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern!r} in {args.folder}")
        return 1
    print(f"Batch: {len(files)} files from {args.folder}  "
          f"(render={args.render}, force={args.force})\n")

    t0 = time.time()
    done = skipped = failed = 0
    for i, f in enumerate(files, 1):
        cube = OUTPUT_DIR / f"{f.stem}_gridded.nc"
        tag = f"[{i:>2}/{len(files)}] {f.name}"

        if cube.exists() and not args.force:
            print(f"{tag}: cube exists, pipeline skipped")
            skipped += 1
        else:
            ok, tail = run_one(SRC / "pipeline.py", f)
            if not ok:
                print(f"{tag}: PIPELINE FAILED\n    {tail}")
                failed += 1
                continue
            print(f"{tag}: gridded ok")
            done += 1

        if args.render != "none":
            ok, tail = run_one(RENDER_SCRIPTS[args.render], cube)
            if not ok:
                print(f"{tag}: render failed\n    {tail}")
                failed += 1
            else:
                print(f"{tag}: {args.render} render ok")

    mins = (time.time() - t0) / 60.0
    print(f"\nBatch finished in {mins:.1f} min — "
          f"{done} processed, {skipped} skipped (cube existed), {failed} failures")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
