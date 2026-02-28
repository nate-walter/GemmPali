# GemmPali Dataset Manifest (2026-02-28)

Policy: HDD-first persistent storage, NVMe ad-hoc staging for active training phases.

## Canonical Sources (locked)

### AHS-uni MP-DocVQA/DUDE (IR split canonical)
- `AHS-uni/mpdocvqa-corpus`
  - SHA: `33ada50ffeee4c67f6cbe39c6ef3f7afdd5e547b`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/mpdocvqa-corpus`
- `AHS-uni/mpdocvqa-qa`
  - SHA: `8d2bc8492866082401857a0639da0d09ba1211ff`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/mpdocvqa-qa`
- `AHS-uni/dude-corpus`
  - SHA: `736195e24b6001e227b6b52da5a2432233cd6707`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/dude-corpus`
- `AHS-uni/dude-qa`
  - SHA: `ab6b22d605d0fbb40d73ed8d486681b29fe0129e`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/dude-qa`

### ViDoRe / DocVQA
- `vidore/colpali_train_set`
  - SHA: `e13d3594064836f7fd69fad7e3d2b51065b335c7`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/vidore-colpali-train-set`
- `vidore/docvqa_train`
  - SHA: `e24fa95e0dec4d7ef0a8fdb951ba3f9a1cc69159`
  - local: `/mnt/ripped_media/GemmPali/datasets/raw/vidore-docvqa-train`

## Internal supplemental corpus
- CGI annual-report corpus is retained as supplemental stress/hard-negative domain.
- Not the central model identity; GemmPali remains general-domain multi-page.

## Download command pattern

```bash
~/.local/bin/hf download <repo> --repo-type dataset --local-dir <target_dir>
```

## Notes
- Do not pivot to alternate MP-DocVQA/DUDE distributions unless explicitly approved.
- Keep corpus/qa split architecture intact.
- Capture new SHAs when source revisions are pulled.
