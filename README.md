# VCF: Variable-Order Cross-Channel Flow


## Data and code availability

| Material | Availability for the article release | Location |
|---|---|---|
| YDFID-1 images and ground-truth masks | Obtain from the original provider under its access conditions; not redistributed here | [Official provider and application instructions](https://github.com/ZHW-AI/YDFID-1) |
| Processed file lists and experimental splits | Planned for public release | `dataset/mvisa/data/meta_ydfid.json`, `meta_visa.json`, `meta_mvtec.json`|
| Data preprocessing code | Planned for public release | `dataset/`|
| Model, training and inference code | Planned for public release | `train.py`, `test.py`, `loss.py`, `models/` |
| Evaluation and statistical analysis | Planned for public release | `models/evaluate.py`, `models/metric_and_visualization.py`|

YDFID-1 is provided by the Zhang Hongwei Artificial Intelligence Research Group at Xi'an Polytechnic University. Request **version 1** following the provider's instructions. Its original image archive and masks are not included in this repository. The provider's page also describes later versions; they must not be substituted for the version used in this study.

MVTec-AD and VisA should be obtained from their original providers: [MVTEC_OFFICIAL_DATA_URL] and [VISA_OFFICIAL_DATA_URL]. The fixed manifests identify the actual samples and their roles in our experiments. Dataset access terms are separate from the software license.
