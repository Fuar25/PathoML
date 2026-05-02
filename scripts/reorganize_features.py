"""重组 Features 目录：从按染色组织 → 按 patient/tissue 组织。

Before:
  GigaPath-Slide-Feature/HE/B2022-01475B-HE.h5
  GigaPath-Slide-Feature/CD20/B2022-01475B-cd20.h5

After:
  GigaPath-Slide-Feature/B2022-01475/B/B2022-01475B-HE.h5
  GigaPath-Slide-Feature/B2022-01475/B/B2022-01475B-cd20.h5

Additionally generates stain_index.json at Features root:
  {
    "stains": ["CD20", "CD21", "CD3", "CK-pan", "HE", "Ki-67"],
    "patients": {
      "B2022-01475": {
        "B": ["CD20", "CD21", "CD3", "CK-pan", "HE", "Ki-67"]
      }
    }
  }

Usage:
  python scripts/reorganize_features.py                 # dry-run
  python scripts/reorganize_features.py --execute       # actually move files
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict

PATIENT_ID_PATTERN = re.compile(r"((?:xs)?B\d{4}-\d{5})")
TISSUE_PATTERN = re.compile(r"([A-Za-z0-9])-")


def parse_filename(filename):
  """Extract (patient_id, tissue_id, stain) from H5 filename.

  Example: "B2022-01475B-cd20.h5" → ("B2022-01475", "B", "cd20")
  """
  m = PATIENT_ID_PATTERN.search(filename)
  if not m:
    return None
  patient_id = m.group(1)
  remaining = filename[m.end():]
  tm = TISSUE_PATTERN.match(remaining)
  if not tm:
    return None
  tissue_id = tm.group(1)
  # stain is between the hyphen after tissue_id and .h5
  stain_part = remaining[tm.end():]
  stain = stain_part.replace('.h5', '')
  return patient_id, tissue_id, stain


def scan_feature_dir(feat_dir):
  """Scan a feature type directory (Slide or Patch). Returns list of move operations.

  Each entry: (src_path, dst_path, patient_id, tissue_id, stain)
  """
  ops = []
  for stain_dir in sorted(os.listdir(feat_dir)):
    stain_path = os.path.join(feat_dir, stain_dir)
    if not os.path.isdir(stain_path):
      continue
    for filename in os.listdir(stain_path):
      if not filename.endswith('.h5'):
        continue
      parsed = parse_filename(filename)
      if parsed is None:
        print(f"  WARNING: cannot parse '{filename}', skipping")
        continue
      patient_id, tissue_id, stain = parsed
      src = os.path.join(stain_path, filename)
      dst = os.path.join(feat_dir, patient_id, tissue_id, filename)
      ops.append((src, dst, patient_id, tissue_id, stain_dir))
  return ops


def build_stain_index(all_ops):
  """Build stain index from all move operations.

  Returns dict: {stains: [...], patients: {patient_id: {tissue_id: [stain, ...]}}}
  """
  stains = set()
  patients = defaultdict(lambda: defaultdict(list))
  for _, _, patient_id, tissue_id, stain_dir in all_ops:
    stains.add(stain_dir)
    if stain_dir not in patients[patient_id][tissue_id]:
      patients[patient_id][tissue_id].append(stain_dir)

  # Sort everything
  for pid in patients:
    for tid in patients[pid]:
      patients[pid][tid].sort()

  return {
    "stains": sorted(stains),
    "patients": {k: dict(v) for k, v in sorted(patients.items())},
  }


def main():
  parser = argparse.ArgumentParser(description="Reorganize Features: stain-based → patient-based")
  parser.add_argument("--features-root", default=None,
                      help="Features root directory (default: auto-detect)")
  parser.add_argument("--execute", action="store_true",
                      help="Actually move files (default: dry-run)")
  args = parser.parse_args()

  # (1) Locate Features root
  if args.features_root:
    feat_root = args.features_root
  else:
    # Auto-detect relative to script location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feat_root = os.path.join(repo_root, "Features")

  if not os.path.isdir(feat_root):
    print(f"ERROR: Features root not found: {feat_root}")
    return

  # (2) Scan both feature types
  all_ops = []
  for feat_type in ["GigaPath-Slide-Feature", "GigaPath-Patch-Feature"]:
    feat_dir = os.path.join(feat_root, feat_type)
    if not os.path.isdir(feat_dir):
      print(f"  Skipping {feat_type} (not found)")
      continue
    ops = scan_feature_dir(feat_dir)
    all_ops.extend(ops)
    print(f"  {feat_type}: {len(ops)} files to move")

  if not all_ops:
    print("No files found to reorganize.")
    return

  # (3) Check for conflicts (two files mapping to same destination)
  dst_set = {}
  conflicts = []
  for src, dst, *_ in all_ops:
    if dst in dst_set:
      conflicts.append((dst, dst_set[dst], src))
    else:
      dst_set[dst] = src
  if conflicts:
    print(f"\nERROR: {len(conflicts)} destination conflicts found:")
    for dst, src1, src2 in conflicts[:5]:
      print(f"  {dst}")
      print(f"    ← {src1}")
      print(f"    ← {src2}")
    return

  # (4) Summary
  patients = set()
  for _, _, pid, *_ in all_ops:
    patients.add(pid)
  print(f"\nTotal: {len(all_ops)} files, {len(patients)} patients")

  # (5) Preview first 10
  print("\nPreview (first 10 moves):")
  for src, dst, *_ in all_ops[:10]:
    print(f"  {os.path.relpath(src, feat_root)}")
    print(f"    → {os.path.relpath(dst, feat_root)}")

  # (6) Build stain index
  stain_index = build_stain_index(all_ops)
  index_path = os.path.join(feat_root, "stain_index.json")
  print(f"\nStain index: {len(stain_index['stains'])} stains, {len(stain_index['patients'])} patients")

  if not args.execute:
    print("\n[DRY-RUN] No files moved. Use --execute to apply.")
    return

  # (7) Execute moves
  print("\nMoving files...")
  moved = 0
  for src, dst, *_ in all_ops:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    moved += 1
  print(f"  Moved {moved} files")

  # (8) Remove ONLY the original stain directories (not the new patient dirs)
  known_stain_dirs = {"CD20", "CD21", "CD3", "CK-pan", "HE", "Ki-67"}
  removed_dirs = 0
  for feat_type in ["GigaPath-Slide-Feature", "GigaPath-Patch-Feature"]:
    feat_dir = os.path.join(feat_root, feat_type)
    if not os.path.isdir(feat_dir):
      continue
    for dirname in list(os.listdir(feat_dir)):
      if dirname not in known_stain_dirs:
        continue
      dirpath = os.path.join(feat_dir, dirname)
      if not os.path.isdir(dirpath):
        continue
      # Only remove if truly empty (no files at all, recursively)
      has_files = any(files for _, _, files in os.walk(dirpath))
      if not has_files:
        shutil.rmtree(dirpath)
        removed_dirs += 1
  print(f"  Removed {removed_dirs} empty stain directories")

  # (9) Write stain index
  with open(index_path, 'w', encoding='utf-8') as f:
    json.dump(stain_index, f, indent=2, ensure_ascii=False)
  print(f"  Stain index written to: {index_path}")

  print("\nDone.")


if __name__ == "__main__":
  main()
