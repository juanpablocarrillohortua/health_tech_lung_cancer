<div style="width:100%; background-color:#00522c; border-bottom:3px solid #34a853; padding:20px 0; text-align:center;">
  <img src="https://pattern-lab-externado-prod.web.app/images/logo-uec.svg" alt="Banner del proyecto" width="40%">
</div>

![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C.svg?style=for-the-badge&logo=python&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0.svg?style=for-the-badge&logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-3D6E9C.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-D7FF64.svg?style=for-the-badge&logo=ruff&logoColor=black)

# Lung Cancer Risk from Screening CT — a LUNA25 replication

Replicating a published deep-learning study on pulmonary nodule malignancy,
using the public **LUNA25** dataset.

## Introduction

This project replicates:

> **Deep Learning for Malignancy Risk Estimation of Pulmonary Nodules
> Detected at Low-Dose Screening CT**
> Venkadesh KV, Setio AAA, Schreuder A, Scholten ET, Chung K, Winkler Wille MM,
> van Ginneken B, Prokop M, Jacobs C.
> *Radiology* (2021) **300**(2), 438-447.
> [doi:10.1148/radiol.2021204433](https://doi.org/10.1148/radiol.2021204433)

The paper develops and validates a deep-learning algorithm that estimates the
**malignancy risk of pulmonary nodules** found on low-dose screening CT. The
model was developed on **16,077 nodules (1,249 malignant)** collected between
2002 and 2004 from the **National Lung Screening Trial (NLST)**, and validated
externally on the **Danish Lung Cancer Screening Trial (DLCST)** — a baseline
cohort of 883 nodules (65 malignant, 818 benign) plus two cancer-enriched
subsets. It was compared against the PanCan 2b risk calculator and against 11
clinicians in an observer study, and concluded that the algorithm generalised
across screening populations and protocols with discriminative performance
comparable to clinical experts.

**Why LUNA25.** The replication uses the public
[LUNA25](https://luna25.grand-challenge.org/) training set, which is drawn from
the same NLST trial the paper developed on. The baseline dataloader vendored in
`data_loader/` comes from `DIAGNijmegen/luna25-baseline-public` — the same
Nijmegen group that authored the paper.

**Where the project stands.** The current work is the data layer and the
exploratory analysis: consolidating LUNA25 labels with the NLST clinical
tables, profiling the variables, and testing which of them relate to
malignancy. The modelling track is staged (`config.py`, `data_loader/`) but
**not yet trained**.

| Local data | Content |
| --- | --- |
| `data/labels/` | LUNA25 training CSV — 6,163 annotated nodules, 2,120 patients, 555 malignant (9.0%) |
| `data/nlst_data/` | Five NLST tables (nodule `ctab`, participant `prsn`, screening, cancer) |
| `data/images/` | Nodule blocks, shipped as a split zip |

## Project structure

```
health_tech_lung_cancer/
├── notebooks/                     # the analysis; entry point
│   ├── initial_eda.ipynb          # first pass over the LUNA25 labels
│   └── image_caracteristics.ipynb # LUNA25 x NLST consolidation + EDA
├── utils/                         # generic analysis toolkit (see utils/README.md)
│   ├── fancy_barplot.py           # categorical distributions
│   ├── fancy_hist.py              # numeric distributions
│   ├── segmented_bars.py          # small multiples, one panel per segment
│   ├── target_against_features.py # auto-selected hypothesis tests
│   ├── cat_num_dep.py             # numeric vs categorical test hub
│   ├── triple_plot.py             # normality report
│   ├── corr_vector.py             # correlation against one target
│   ├── bivaraite_boxplot.py       # segmented boxplot
│   ├── category_vs_numeric_plot.py# faceted histograms
│   ├── mutual_inf_plot.py         # relevance = correlation + mutual info
│   └── fast_OLS.py                # OLS diagnostics
├── domain_utils/                  # LUNA25/NLST-specific logic
│   └── nodule_match.py            # matches each nodule to its NLST abnormality
├── data_loader/
│   ├── dataloader.py              # VENDORED from the LUNA25 baseline (excluded from ruff)
│   └── unzip_data.py              # joins and extracts the split image zip
├── data/                          # labels, NLST tables, nodule blocks (gitignored)
├── tools/                         # cross-platform helpers for the Makefile
├── outputs/                       # generated plots and tables
├── config.py                      # paths and model hyperparameters
├── Makefile                       # every command routes through here
├── ruff.toml                      # lint/format configuration
└── pulmones_paper.pdf             # the paper being replicated
```

## Getting started

```bash
# 1. Environment
python -m venv .venv
source .venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere

# 2. Dependencies
pip install -r requirements.txt

# 3. Image data — joins the split zip and extracts it
python data_loader/unzip_data.py

# 4. Run the notebooks
jupyter lab notebooks/
```

Two things that will bite you on a fresh clone:

- **The LUNA25 CSV is not in the repo.** `.gitignore` starts with `*.csv`, so
  `data/labels/LUNA25_Public_Training_Development_Data.csv` and the NLST tables
  must be downloaded separately and placed under `data/`.
- **`requirements.txt` pins `torch==2.13.0+cu132`**, a local CUDA build that
  will not resolve from plain PyPI. Install torch from the PyTorch index first,
  or drop the `+cu132` pins if you only need the EDA — the notebooks do not use
  torch.

Everything else routes through `make`, which resolves the venv interpreter for
you:

| Command | What it does |
| --- | --- |
| `make help` | list every documented target |
| `make install-dev` | install the lint toolchain |
| `make lint` | ruff check over `utils/`, `data_loader/`, `notebooks/` |
| `make format` | auto-fix imports and layout, then `ruff format` |
| `make quality` | `clean` + `lint` + `format-check` — the CI gate |
| `make clean` | drop caches and notebook checkpoints |

> The `utils/` API — every parameter and return value — is documented in
> **[utils/README.md](utils/README.md)**.

## Variable dictionary

The consolidated dataset joins LUNA25 annotations with the NLST nodule and
participant tables. One row per nodule.

### Nodule geometry and identifiers (LUNA25)

| Variable | Description | Values |
| --- | --- | --- |
| **`CoordX`** | Nodule centre on the X axis (mm) | Continuous, in mm |
| **`CoordY`** | Nodule centre on the Y axis (mm) | Continuous, in mm |
| **`CoordZ`** | Nodule centre on the Z axis (slice/depth) | Continuous, or slice number |
| **`LesionID`** | Identifier assigned to the individual lesion | Unique numeric/alphanumeric id |
| **`AnnotationID`** | Id of the radiological annotation made on the nodule | Unique numeric/alphanumeric id |
| **`NoduleID`** | Unique identifier of the evaluated nodule | Unique numeric/alphanumeric id |
| **`label`** | Target label (ground truth) | `0`: Benign<br>`1`: Malignant |

### Nodule descriptors (NLST `sct_*`)

| Variable | Description | Values |
| --- | --- | --- |
| **`sct_long_dia`** | Longest/longitudinal diameter of the nodule (mm) | *Numeric*: longest diameter in mm for non-calcified nodules or masses $\ge 4$ mm<br>`.N`: Not applicable (the description is not a nodule/mass $\ge 4$ mm)<br>`.S`: Impossible to determine |
| **`sct_margins`** | Characteristics of the nodule's borders or margins | `1`: Spiculated / Stellate<br>`2`: Smooth<br>`3`: Poorly defined<br>`9`: Unable to determine<br>`.N`: Not applicable |
| **`sct_pre_att`** | Predominant attenuation/density of the nodule | `1`: Soft tissue<br>`2`: Ground glass<br>`3`: Mixed<br>`4`: Fluid / water<br>`6`: Fat<br>`7`: Other<br>`9`: Unable to determine<br>`.M`: Missing<br>`.N`: Not applicable |
| **`sct_slice_num`** | CT slice holding the abnormality's largest dimension | *Numeric*: number of the CT slice containing the nodule's largest diameter<br>`999`: Missing<br>`.N`: Not applicable |

### Participant (NLST)

| Variable | Description | Values |
| --- | --- | --- |
| **`Age_at_StudyDate`** | Participant's age on the exact date of the exam | Continuous numeric (years) |
| **`Gender`** | Gender as recorded in LUNA25 | Participant's gender category |
| **`race`** | Participant's race or ethnicity | `1`: White<br>`2`: Black or African-American<br>`3`: Asian<br>`4`: American Indian or Alaskan Native<br>`5`: Native Hawaiian or Other Pacific Islander<br>`6`: More than one race<br>`7`: Participant refused to answer<br>`95`: Missing data form — form is not expected to be completed<br>`96`: Missing — no response<br>`98`: Missing — form was submitted and the item was left blank<br>`99`: Unknown / declined to answer |
| **`cigsmok`** | Smoking history and cumulative consumption | `0`: Former smoker<br>`1`: Current smoker |

### Tumour location (NLST)

Binary indicators; largely null in this cohort, so they are dropped early in
the analysis.

| Variable | Description | Values |
| --- | --- | --- |
| **`loclhil` / `locrhil`** | Cancer located in the left / right hilum | `0`: Absent / not in the hilum<br>`1`: Present / located in the hilum |
| **`locllow` / `locrlow`** | Cancer in the left / right lower lobe | `0`: Absent in this lobe<br>`1`: Located in the lower lobe |
| **`loclup` / `locrup`** | Cancer in the left / right upper lobe | `0`: Absent in this lobe<br>`1`: Located in the upper lobe |
| **`locrmid`** | Cancer in the right middle lobe | `0`: Absent in the right middle lobe<br>`1`: Located in the right middle lobe |
| **`locunk`** | Unknown pulmonary location | `0`: Known location<br>`1`: Unspecified / unknown location |

## References

APA 7. A BibTeX version of this list, for LaTeX or a reference manager, is in
[references.bib](references.bib).

### Replicated study

Venkadesh, K. V., Setio, A. A. A., Schreuder, A., Scholten, E. T., Chung, K.,
Winkler Wille, M. M., van Ginneken, B., Prokop, M., & Jacobs, C. (2021). Deep
learning for malignancy risk estimation of pulmonary nodules detected at
low-dose screening CT. *Radiology, 300*(2), 438–447.
https://doi.org/10.1148/radiol.2021204433

### Data sources

Peeters, D., Obreja, B., Antonissen, N., & Jacobs, C. (2025). *The LUNA25
challenge: Public training and development set — Imaging data* (Version 1.0.0)
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.14223624

Peeters, D., Obreja, B., Antonissen, N., & Jacobs, C. (2025). *The LUNA25
challenge: Public training and development set — Annotation data*
(Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14673658

National Lung Screening Trial Research Team. (2013). *Data from the National
Lung Screening Trial (NLST)* [Data set]. The Cancer Imaging Archive.
https://doi.org/10.7937/TCIA.HMQ8-J677

National Lung Screening Trial Research Team, Aberle, D. R., Adams, A. M.,
Berg, C. D., Black, W. C., Clapp, J. D., Fagerstrom, R. M., Gareen, I. F.,
Gatsonis, C., Marcus, P. M., & Sicks, J. D. (2011). Reduced lung-cancer
mortality with low-dose computed tomographic screening. *New England Journal
of Medicine, 365*(5), 395–409. https://doi.org/10.1056/nejmoa1102873

### Tools and implementations

Diagnostic Image Analysis Group, Radboud University Medical Center. (2025).
*LUNA25: Lung Nodule Analysis 2025 challenge*.
https://luna25.grand-challenge.org/

Diagnostic Image Analysis Group, Radboud University Medical Center. (2025).
*LUNA25 baseline (public)* [Computer software]. GitHub.
https://github.com/DIAGNijmegen/luna25-baseline-public

Jacobs, C., Venkadesh, K. V., & van Ginneken, B. (2021). *Pulmonary nodule
malignancy prediction* [Algorithm]. Grand Challenge.
https://grand-challenge.org/algorithms/pulmonary-nodule-malignancy-prediction/

Hasson, Y. (2017). *kinetics_i3d_pytorch: Inflated 3D ConvNet models and
weights transferred from TensorFlow to PyTorch* [Computer software]. GitHub.
https://github.com/hassony2/kinetics_i3d_pytorch

Carreira, J., & Zisserman, A. (2017). Quo vadis, action recognition? A new
model and the Kinetics dataset. In *Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition* (pp. 6299–6308).
https://doi.org/10.1109/CVPR.2017.502

### Software libraries

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R.,
Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A.,
del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array
programming with NumPy. *Nature, 585*(7825), 357–362.
https://doi.org/10.1038/s41586-020-2649-2

McKinney, W. (2010). Data structures for statistical computing in Python. In
*Proceedings of the 9th Python in Science Conference* (pp. 56–61).
https://doi.org/10.25080/Majora-92bf1922-00a

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J.,
van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N.,
Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … van Mulbregt, P. (2020).
SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature
Methods, 17*(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in
Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of
Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021

Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical
modeling with Python. In *Proceedings of the 9th Python in Science Conference*
(pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A.,
Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B.,
Fang, L., … Chintala, S. (2019). PyTorch: An imperative style,
high-performance deep learning library. In *Advances in Neural Information
Processing Systems 32* (pp. 8024–8035).
https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library

Pérez, F., & Granger, B. E. (2007). IPython: A system for interactive
scientific computing. *Computing in Science & Engineering, 9*(3), 21–29.
https://doi.org/10.1109/MCSE.2007.53

Kluyver, T., Ragan-Kelley, B., Pérez, F., Granger, B., Bussonnier, M.,
Frederic, J., Kelley, K., Hamrick, J., Grout, J., Corlay, S., Ivanov, P.,
Avila, D., Abdalla, S., & Willing, C. (2016). Jupyter Notebooks — a publishing
format for reproducible computational workflows. In *Positioning and Power in
Academic Publishing: Players, Agents and Agendas* (pp. 87–90). IOS Press.
https://doi.org/10.3233/978-1-61499-649-1-87

Astral Software Inc. (2025). *Ruff: An extremely fast Python linter and code
formatter* (Version 0.16.5) [Computer software]. GitHub.
https://github.com/astral-sh/ruff

The two below are imported by `utils/mutual_inf_plot.py` and
`utils/fast_OLS.py` respectively, but are **not installed** in the current
environment, so those two modules do not run as shipped:

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &
Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of
Machine Learning Research, 12*, 2825–2830.
https://jmlr.org/papers/v12/pedregosa11a.html

Plotly Technologies Inc. (2015). *Collaborative data science*. Plotly
Technologies Inc. https://plot.ly

Library versions are those pinned in [requirements.txt](requirements.txt):
NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, Matplotlib 3.11.1, seaborn 0.13.2,
statsmodels 0.15.0, PyTorch 2.13.0, IPython 9.17.0, ipykernel 7.3.0,
Ruff 0.16.5.

## Licence

MIT — see [LICENSE](LICENSE).
