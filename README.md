# VCF: Variable-Order Cross-Channel Flow

> DRAFT TEMPLATE. This file does not certify that a public release exists. Replace all square-bracket placeholders and implement and verify the referenced materials before removing this notice. The availability statuses below describe a proposed release plan.

Code and reproducibility materials for VCF, a method for zero-shot anomaly detection in yarn-dyed woven fabrics.

- Paper: [PAPER_TITLE]
- Authors: [AUTHORS]
- Paper link: [PAPER_URL_OR_DOI]
- Code repository: [REPOSITORY_URL]
- Article release: [RELEASE_TAG]
- Full commit SHA: [COMMIT_SHA]
- Archived release: [VERSION_SPECIFIC_DOI_OR_ARCHIVE_URL]

## Data and code availability

| Material | Availability for the article release | Location |
|---|---|---|
| YDFID-1 images and ground-truth masks | Obtain from the original provider under its access conditions; not redistributed here | [Official provider and application instructions](https://github.com/ZHW-AI/YDFID-1) |
| Processed file lists and experimental splits | Planned for public release | `dataset/mvisa/data/meta_ydfid.json`, `meta_visa.json`, `meta_mvtec.json`; [FILE_MAPPING_PATH] |
| Data preprocessing code | Planned for public release | `dataset/`; [PREPROCESSING_INSTRUCTIONS_PATH] |
| Model, training and inference code | Planned for public release | `train.py`, `test.py`, `datasets.py`, `loss.py`, `models/`; required local dependencies |
| Training configurations, ablations and random seeds | Planned for public release | `configs/`, `reproduce/run_manifest.csv` |
| Evaluation and statistical analysis | Planned for public release | `models/evaluate.py`, `models/metric_and_visualization.py`, [AGGREGATION_SCRIPT_PATH] |
| Code and inputs for Tables 1–10 and Figures 5–11 | Planned for public release; update numbering to match the final article | `reproduce/README.md`, [TABLE_FIGURE_INDEX_PATH], `results/paper/` |
| Trained VCF checkpoints | [STATE: public at a specific link / not distributed, retraining required] | [CHECKPOINT_LOCATION_OR_RETRAINING_INSTRUCTIONS] |

YDFID-1 is provided by the Zhang Hongwei Artificial Intelligence Research Group at Xi'an Polytechnic University. Request **version 1** following the provider's instructions. Its original image archive and masks are not included in this repository. The provider's page also describes later versions; they must not be substituted for the version used in this study.

MVTec-AD and VisA should be obtained from their original providers: [MVTEC_OFFICIAL_DATA_URL] and [VISA_OFFICIAL_DATA_URL]. The fixed manifests identify the actual samples and their roles in our experiments. Dataset access terms are separate from the software license.

## Installation

Tested environment: [PYTHON_VERSION], [PYTORCH_VERSION], [TORCHVISION_VERSION], [CUDA_VERSION], [OS], [GPU].

```bash
python -m pip install -r requirements.txt
```

Obtain the CLIP initialization weights from [OFFICIAL_WEIGHT_SOURCE]. The required artifact is [WEIGHT_FILENAME], with SHA-256 [WEIGHT_SHA256]. Place it at [WEIGHT_PATH] or pass its location as documented below.

## Data preparation

Follow [PREPROCESSING_INSTRUCTIONS_PATH] to prepare the source datasets and regenerate the processed file lists. This document must specify original-to-processed file mapping, inclusion/exclusion rules, duplicate handling, mask conversion, and image/mask transformations.

The file manifests use paths relative to the data root. [FILE_MAPPING_PATH] records original file identifiers, processed paths and checksums. Run [MANIFEST_VERIFICATION_COMMAND] to check the prepared data against the released manifests.

Experimental roles are recorded explicitly as source training, monitoring/validation and target evaluation. These roles must be read from the experiment manifests rather than inferred from directory names.

## Training and evaluation

The source-to-target protocols used in the paper are:

- VisA → YDFID-1.
- VisA → MVTec-AD.
- MVTec-AD → VisA.

`configs/` and `reproduce/run_manifest.csv` record the complete settings for every main experiment, ablation and parameter study, including learning rate, batch size, resolution, optimizer, prompt settings, flow settings, feature layers, loss weights, module settings, stage-switching rule, checkpoint-selection rule and seeds.

Actual independent training seeds: [RECOVER_AND_LIST_ACTUAL_TRAINING_SEEDS]. Evaluation seeds: [ACTUAL_EVALUATION_SEEDS_BY_RUN]. Data preprocessing randomness, if used: [ACTUAL_PREPROCESSING_SEED_OR_NOT_APPLICABLE]. Seeds used only for plot jitter or selecting examples are documented separately.

Run the exact commands in `reproduce/README.md`. Do not use example defaults as a substitute for the recorded paper configurations. That file must contain executable commands for data preparation, each training protocol, evaluation, ablations and report generation; placeholders are not release-ready commands.

## Reproducing the paper

`reproduce/README.md` maps every requested table and figure to its final article number/title, input files, run IDs, configuration, command and output file. Extend the map if the final revision adds or renumbers tables.

The workflow has two parts:

1. Run preprocessing, training/inference and evaluation to obtain experiment outputs.
2. Use the documented aggregation and plotting commands to regenerate tables and figures from the archived outputs.

Numerical source files are in `results/paper/`. Per-run and per-category values, class ordering, rounding, macro-averaging, and mean/standard-deviation conventions are documented in [METRIC_PROTOCOL_PATH]. Threshold-dependent metrics and visualizations specify how thresholds were obtained.

For qualitative figures, [FIGURE_SAMPLE_MANIFEST_PATH] records the exact sample identifiers, run/checkpoint, display thresholds, normalization, crops, annotation boxes and panel order. For score histograms, release per-sample scores and the score-extraction and binning rules. For training curves, release the selected run's epoch-level data and selection rule.

Baseline implementations are identified in [BASELINE_MANIFEST_PATH] by upstream repository, exact commit, configuration and any local modifications. Results rerun by the authors are distinguished from values quoted from publications. Upstream code may be obtained from its original repository; any modifications needed to reproduce this paper are included here.

If trained checkpoints are not released, reproduction of model predictions requires retraining. State which archived numerical outputs or permitted cached predictions support rebuilding the published figures. Numerical differences across hardware and software environments may occur; exact agreement should only be claimed to the extent verified.

## License and citation

Author-owned software is released under [ACTUAL_SOFTWARE_LICENSE]. Third-party code retains its applicable notices and licenses; see `THIRD_PARTY_NOTICES.md`. Dataset and pretrained-weight terms remain those of their respective providers.

Please cite [PAPER_CITATION] and the article-specific archived software release [VERSION_SPECIFIC_DOI]. See `CITATION.cff` for machine-readable citation information.

For questions about this implementation, use [ISSUE_TRACKER_OR_CONTACT]. Requests for YDFID-1 access should be sent to the original provider.
