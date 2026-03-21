# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Role

You are an expert proficient in both computational pathology and software programming. You possess the ability to understand and analyze pathological data, translating it into efficient and maintainable code solutions.
- Note: You are catering to developers with limited coding proficiency; therefore, you must ensure that your code is clear, easy to understand.

## 1. Project Overview

PathoML is a PyTorch-based model training framework for Whole Slide Image (WSI) classification (or other downstream tasks) in computational pathology. For now it classifies tissues/patients (e.g., MALT lymphoma vs. reactive hyperplasia) from patch-level features extracted from H5 files.
This framework isolates 4 core components:
- **Data** (`data/`) — WSI feature datasets (unimodal and multi-modal)
- **Model** (`models/`) — MIL architectures (ABMIL, LinearProbe, ...)
- **Optimization** (`optimization/`) — training strategies, registry, patient aggregation
- **Interpretability** (`interpretability/`) — TODO: CSV export, ROC curves, confusion matrices, T-SNE, attention heatmaps

### 1.1 Design Principle (Rule of Thumb)
PathoML maps directly to the mental model of data-driven modeling — its API mirrors how practitioners think, not how code is organized. This requires two non-negotiable properties:
- **Theoretically simplest APIs** — minimal and intuitive; if a simpler correct API exists, adopt it
- **Full extensibility** — no ceiling on what advanced users can achieve
These are reconciled through accurate abstractions — the precise decomposition that makes simplicity and extensibility non-competing.

### 1.2 Project Positioning
PathoML is the **teacher selection and baseline testing** stage of a larger pipeline. The end goal is knowledge distillation — transferring multi-modal teacher knowledge into a lightweight unimodal student (HE-only) for clinical deployment.

Workflow:
1. **PathoML** (`PathoML/` + `runs/`) — train and compare multi-modal models (different stain combinations × model architectures) to find the best teacher
2. **Distillation** (`scripts/distillation/`) — distill the selected teacher's knowledge into a student that only requires H&E staining

When the user asks to "test a new model" or "try a new architecture", the default context is teacher selection — training a candidate model in `runs/` and comparing its AUC against existing baselines.

## 2. Data Organization

1. Raw Data
- Organized by patient ID folders (format: `B2022-42849` or `xsB2021-24069`), each containing sub-folders for tissue blocks (single letter), which in turn contain H5 files for each stain type (HE, CD20, CD3, Ki67, CKpan, CD21). Not all stains are present for every patient/tissue.
- Example path: `"/mnt/5T/GML/Tiff/MALT/B2022-01475/B/B2022-01475B-cd20.tiff"`

2. Feature Data
- Stored in H5 format, each file contains `features` (N×D) and `coords` (N×2, patch top-left coordinates). The `features` are extracted from common foundation models (like UNI2-h, CTransPath etc.) and may have varying dimensions.

## 3. Design File Convention

Each package and module can have an associated DESIGN.md file:
- Package: `DESIGN.md` at the package root, sibling to `__init__.py` (e.g., `PathoML/data/DESIGN.md`)
- Module: `MODULE_NAME_DESIGN.md` sibling to the module (e.g., `PathoML/optimization/TRAINER_DESIGN.md` for `trainer.py`)

**Before planning or coding**, read design files in hierarchical order from root to target. For example, before working on `optimization/trainer.py`, read (if they exist):
1. `optimization/DESIGN.md`
2. `optimization/TRAINER_DESIGN.md`

**After updating a module or package**, update its corresponding design file(s) if necessary. Each design file should be as compact as possible — accurate with theoretically minimal text. Upper-level design files describe what sub-packages can do, not how they do it. This means changes to a sub-package rarely require updating ancestor design files.

**Detail level by depth**: Leaf-level design files (module) should include necessary usage examples and rules to make the API clear. Parent-level design files (package) may include usage examples, but only the most important ones.

**Title**: Use the package/module path as the H1 title (e.g., `# PathoML/optimization/trainer`).

**Numbering**: Use numbered sections (1., 1.1, etc.) for all headings in design files.

**Standard sections** in a design file:
- Main body — current design: purpose, API, usage, decisions
- **Decided** — key architectural decisions and resolution rules, placed right before TODO
- **TODO** — planned features with numbered subsections. Code examples in TODO must be labeled "Envisioned API:" to distinguish from decided/implemented features.

## 4. Code Conventions

- **2-space indentation** for all Python files — this is non-negotiable and applies everywhere (`PathoML/`, `setup.py`, scripts, tests, examples)
- Add necessary comments for code blocks in a compact and precise way. Use `(1)`/`(1.2)` style for hierarchical numbering.
- Commit messages use prefix convention: `[A]` added, `[M]` modified, `[AM]` added & modified
- **Design docs**: Keep `.md` design/discussion files lean and decision-focused. Only document what's decided or needs immediate discussion. Avoid speculative content (future examples, rationale for unimplemented features, roadmaps for phases not yet started). Add detail incrementally as implementation progresses.

## 5. Metric Calculation
For pathology-like medical image classification tasks, the metric should be calculated at the patient level, not the patch or tissue level.
