"""Promote one teacher run output to the fixed teacher-winner location."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT.parent / "PathoML-runs"


def _write_teacher_note(dest: Path, condition: str, source: Path) -> None:
  note = "\n".join(
    [
      "# Current Teacher Winner",
      "",
      f"- condition: `{condition}`",
      f"- promoted_at: `{datetime.now().isoformat(timespec='seconds')}`",
      f"- source: `{source}`",
      "",
      "This directory is the fixed teacher artifact consumed by distillation.",
    ]
  )
  (dest / "TEACHER.md").write_text(note + "\n", encoding="utf-8")


def promote_teacher_winner(
  condition: str,
  *,
  runs_root: Path,
  source: Path | None,
  force: bool,
) -> Path:
  source_dir = source or runs_root / "teacher" / condition
  dest_dir = runs_root / "teacher-winners"
  manifest_path = source_dir / "manifest.json"
  if not manifest_path.exists():
    raise FileNotFoundError(f"Missing teacher manifest: {manifest_path}")
  if dest_dir.exists():
    if not force:
      raise FileExistsError(f"Destination exists; rerun with --force: {dest_dir}")
    shutil.rmtree(dest_dir)
  dest_dir.parent.mkdir(parents=True, exist_ok=True)
  shutil.move(str(source_dir), str(dest_dir))
  _write_teacher_note(dest_dir, condition, source_dir)
  return dest_dir


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("condition", help="Teacher condition name to promote.")
  parser.add_argument(
    "--runs-root",
    type=Path,
    default=Path(os.environ.get("PATHOML_RUNS_ROOT", DEFAULT_RUNS_ROOT)),
  )
  parser.add_argument("--source", type=Path, default=None)
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()
  dest = promote_teacher_winner(
    args.condition,
    runs_root=args.runs_root,
    source=args.source,
    force=args.force,
  )
  print(f"Promoted teacher winner to {dest}")


if __name__ == "__main__":
  main()
