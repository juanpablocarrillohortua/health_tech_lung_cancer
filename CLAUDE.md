# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Exploratory data analysis of the LUNA25 public training set (`data/labels/LUNA25_Public_Training_Development_Data.csv`: 6,163 annotated lung nodules, binary `label`, plus patient metadata). A vendored copy of the LUNA25 baseline dataloader is staged for a later modelling phase but is not part of the current EDA.

## Commands

Everything routes through `make`. It resolves `PYTHON` to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python`, and calls ruff as `$(PYTHON) -m ruff` to avoid PATH/`.exe` resolution problems — so run `make lint`, not bare `ruff`.

| Command | Purpose |
| --- | --- |
| `make help` | list every documented target |
| `make install-dev` | install the lint toolchain (`requirements/dev.txt`) |
| `make lint` | ruff check over `utils/`, `data_loader/`, `notebooks/` |
| `make format` | auto-fix imports/layout, then `ruff format` |
| `make format-check` | verify formatting without rewriting |
| `make quality` | `clean` + `lint` + `format-check` — the CI gate |
| `make clean` | drop `__pycache__`, `.ipynb_checkpoints`, tool caches |

Lint scope is `PY_DIRS := utils data_loader` and `NB_DIRS := notebooks`. **`config.py` and `tools/` are not linted** — don't reformat them to satisfy a gate that never runs on them. `data_loader/dataloader.py` is excluded in `ruff.toml` via `extend-exclude` + `force-exclude` (see Known gaps); `data_loader/unzip_data.py` still is linted.

`.github/workflows/quality.yml` runs the same `make quality` on Python 3.13, installing only `requirements/dev.txt`.

**Non-functional targets**: `make test` runs `pytest tests -q` but there is no `tests/` directory and pytest isn't installed. `make scrape`/`scrape-venta` invoke `python -m scraper`, leftovers from a scraper template; no `scraper` package exists. `make install` reads an empty `requirements.txt`.

## Architecture

Notebooks are the entry point and consume everything else:

- **`notebooks/`** — the analysis itself. Cell 0 of `initial_eda.ipynb` does `sys.path.insert(0, str(Path.cwd().parent))` before `from utils.fancy_barplot import fancy_bars`. That sys.path hop is the only reason `utils` and `config` resolve from inside a notebook; keep it in any new notebook.
- **`utils/`** — one self-contained module per analysis idea (`fancy_barplot`, `triple_plot`, `corr_vector`, `cat_num_dep`, `fast_OLS`, …). Modules **do not import each other**, and `utils/__init__.py` re-exports nothing; import the submodule directly.
- **`config.py` + `data_loader/dataloader.py`** — vendored near-verbatim from `DIAGNijmegen/luna25-baseline-public` (each says so in its docstring; the dataloader is explicitly marked `PLACEHOLDER`). This is the future model track, unused by the EDA.
- **`tools/clean.py`, `tools/mk_help.py`** — Python instead of shell recipes so the Makefile behaves the same under cmd.exe, Git Bash, macOS and Linux.

## Writing a new `utils/` module

Match the existing house style — `utils/fancy_barplot.py` and `utils/triple_plot.py` are the reference implementations:

- **Layout**: module docstring → style constants (`INK, TXT, GRID`, `PALETTE`, `C_DARK`…) → private `_`-prefixed helpers → public function last.
- **Signature**: `(df, <column names>, ..., title=None, palette=None, ax=None, figsize=(w, h))`, returning `plt.Axes`. Build the figure only when `ax is None`, so callers can compose subplots.
- **`title=None` means auto-generate** from the column name via `_pretty()` (`"foo_bar"` → `"Foo Bar"`).
- **Validate literal-string arguments explicitly** and `raise ValueError` with a Spanish message — see [fancy_barplot.py:107](utils/fancy_barplot.py#L107).
- **Language split** (stated in [triple_plot.py:3-6](utils/triple_plot.py#L3-L6)): text drawn on the figure and user-facing docstrings in Spanish, code identifiers in English.
- **Reuse rather than re-derive**: `_pretty()`, `_colors()`, `_style()`, `_bar_labels()` in [fancy_barplot.py](utils/fancy_barplot.py); `_fd_bins()` (Freedman–Diaconis bin count) and `_hist_density()` in [triple_plot.py:101](utils/triple_plot.py#L101). Since modules don't cross-import, copy the helper if you need it elsewhere.

## Conventions

- ruff, line length **79**, `target-version = "py313"`; notebooks linted cell by cell.
- `F401`/`F811`/`F841` are reported but marked `unfixable` on purpose: in exploratory work you routinely write an import or assignment before using it, and `--fix` would delete it. `make format` also runs with `--exit-zero`, so run `make lint` afterwards to see what remains.
- `*.ipynb` additionally ignores `E402` (imports after prose cells) and `F401` (toolkit imported in cell 0, used much later).
- `requirements/dev.txt` pins `ruff==0.16.1` — what CI installs — while the local venv currently has 0.16.5. If a formatting diff appears only in CI, that skew is the first suspect.

## Known gaps

- **Missing dependencies**: `utils/fast_OLS.py` needs `statsmodels` and `plotly`; `utils/mutual_inf_plot.py` needs `scikit-learn`. None are installed in `.venv` — importing those modules fails today. Only numpy, pandas, seaborn, matplotlib, scipy, torch and IPython are available.
- **`requirements.txt` is empty**, so runtime deps are undeclared; `requirements/dev.txt` covers lint tooling only.
- **`config.py` is stale for local use**: `DATADIR`/`CSV_DIR` point at `V:/projects/luna25/...` and `MODEL_RGB_I3D` at a `resources/` folder that doesn't exist, while the real data lives in `data/labels/` and `data/images/`. It also hardcodes an absolute `WORKDIR` and creates `results/` as an import side effect.
- **`.gitignore` line 1 is `*.csv`**, so the LUNA25 labels CSV is deliberately untracked.
- **`data_loader/dataloader.py` is excluded from ruff on purpose** (`extend-exclude` + `force-exclude` in `ruff.toml`). It is vendored near-verbatim from the upstream baseline and marked `PLACEHOLDER`; its 59 findings were upstream style, 33 of them camelCase naming (`N802`/`N803`/`N806`). Treat that as vendor debt to settle when the module is rewritten — don't "fix" it, and don't remove the exclusion to make a number look better. `force-exclude` is what makes the exclusion hold for editor integrations, which pass explicit file paths that a plain `extend-exclude` would not catch.
- **`make quality` still fails**, now solely on `notebooks/image_caracteristics.ipynb`: 16 `E501` long lines in the mapping dictionaries and the `hub_pruebas_num_cat` calls (cells 39, 48, 52, 58). That is the project's own code, not vendored, so it is a real cleanup rather than something to exclude.
