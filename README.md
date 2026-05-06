# PEBG (PyTorch, Multi-Dataset)

This repository contains a refactored PyTorch implementation of PEBG and PEBG+DKT, with a unified pipeline for 4 datasets:
- ASSIST2009
- ASSIST2017
- STATICS2011
- XES3G5M

## 1. Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- pandas
- scikit-learn
- tqdm

Install dependencies:

```bash
pip install torch numpy scipy pandas scikit-learn tqdm
```

## 2. Key Scripts

- `Data/preprocess_assist09.py`
- `Data/preprocess_assist17.py`
- `Data/preprocess_statics2011.py`
- `Data/preprocess_xes3g5m.py`

These scripts convert raw dataset files into a unified intermediate format.

- `prepare_dataset.py`

Converts unified intermediate files into PEBG-compatible files.

- `extract.py`

Builds `pro_pro_sparse.npz` and `skill_skill_sparse.npz` from `pro_skill_sparse.npz`.

- `pebg.py`

PEBG pretraining.

- `pebg_dkt.py`

PEBG+DKT training with per-epoch test AUC/ACC and best-model saving.

## 3. Unified Data Pipeline

### Step A: Run dataset preprocessing

Run one of the following from repo root:

```bash
python Data/preprocess_assist09.py
python Data/preprocess_assist17.py
python Data/preprocess_statics2011.py
python Data/preprocess_xes3g5m.py
```

Each script now resolves paths relative to its own file location, so running from repo root is supported.

Intermediate output directory format:
- `Data/data/<DATASET_NAME>/`

Expected files:
- `train_question.txt`
- `test_question.txt`
- `train_skill.txt`
- `test_skill.txt`
- `ques_skill.csv`

### Step B: Build PEBG-compatible dataset files

```bash
python prepare_dataset.py \
  --input-dir <intermediate_dir> \
  --output-dir <pebg_data_dir> \
  --dataset-name <dataset_name>
```

Example:

```bash
python prepare_dataset.py --input-dir Data/data/ASSIST17 --output-dir datasets/assist17 --dataset-name assist17
```

Generated files:
- `<dataset_name>.npz`
- `pro_skill_sparse.npz`
- `pro_feat.npz`
- `skill_id_dict.txt`

### Step C: Build graph similarity files

```bash
python extract.py --data-dir datasets/assist17
```

Generated files:
- `pro_pro_sparse.npz`
- `skill_skill_sparse.npz`

## 4. Training

### 4.1 Train PEBG

```bash
python pebg.py --data-dir datasets/assist17 --epochs 200
```

Outputs:
- `datasets/assist17/embedding_200.npz`
- `datasets/assist17/pebg_model/pebg_*.pt`

### 4.2 Train PEBG+DKT

```bash
python pebg_dkt.py \
  --data-dir datasets/assist17 \
  --dataset-name assist17 \
  --use-pretrain \
  --embedding-file embedding_200.npz \
  --epochs 200 \
  --save-root runs
```

Per epoch output:
- train loss
- test AUC
- test ACC

Saved artifacts (per dataset):
- `runs/<dataset_name>/best_model.pt`
- `runs/<dataset_name>/metrics.csv`

`best_model.pt` is selected by best test AUC.

## 5. End-to-End Commands

### ASSIST2009

```bash
python Data/preprocess_assist09.py
python prepare_dataset.py --input-dir Data/data/ASSIST09 --output-dir datasets/assist09 --dataset-name assist09
python extract.py --data-dir datasets/assist09
python pebg.py --data-dir datasets/assist09 --epochs 200
python pebg_dkt.py --data-dir datasets/assist09 --dataset-name assist09 --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

### ASSIST2017

```bash
python Data/preprocess_assist17.py
python prepare_dataset.py --input-dir Data/data/ASSIST17 --output-dir datasets/assist17 --dataset-name assist17
python extract.py --data-dir datasets/assist17
python pebg.py --data-dir datasets/assist17 --epochs 200
python pebg_dkt.py --data-dir datasets/assist17 --dataset-name assist17 --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

### STATICS2011

```bash
python Data/preprocess_statics2011.py
python prepare_dataset.py --input-dir Data/data/STATICS2011 --output-dir datasets/statics2011 --dataset-name statics2011
python extract.py --data-dir datasets/statics2011
python pebg.py --data-dir datasets/statics2011 --epochs 200
python pebg_dkt.py --data-dir datasets/statics2011 --dataset-name statics2011 --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

### XES3G5M

```bash
python Data/preprocess_xes3g5m.py
python prepare_dataset.py --input-dir Data/data/XES3G5M --output-dir datasets/xes3g5m --dataset-name xes3g5m
python extract.py --data-dir datasets/xes3g5m
python pebg.py --data-dir datasets/xes3g5m --epochs 200
python pebg_dkt.py --data-dir datasets/xes3g5m --dataset-name xes3g5m --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

## 6. Important Arguments

### prepare_dataset.py
- `--input-dir`: directory with `train/test_question.txt` and `ques_skill.csv`
- `--output-dir`: output directory for PEBG files
- `--dataset-name`: name used for `<dataset_name>.npz`
- `--max-len`: max sequence length (default: `200`)
- `--min-len`: min sequence length (default: `3`)
- `--split`: `all` or `train` (default: `all`)

### extract.py
- `--data-dir`: directory containing `pro_skill_sparse.npz`

### pebg.py
- `--data-dir`
- `--epochs`
- `--batch-size`
- `--embed-dim`
- `--hidden-dim`
- `--keep-prob`
- `--lr`
- `--resume-epoch`

### pebg_dkt.py
- `--data-dir`
- `--dataset-name`
- `--use-pretrain`
- `--embedding-file`
- `--epochs`
- `--batch-size`
- `--hidden-dim`
- `--lr`
- `--train-embed`
- `--save-root`

## 7. Quick Sanity Check

Run short training before full runs:

```bash
python pebg.py --data-dir datasets/assist17 --epochs 2
python pebg_dkt.py --data-dir datasets/assist17 --dataset-name assist17 --use-pretrain --embedding-file embedding_2.npz --epochs 2 --save-root runs
```

## 8. Common Issues

1. Missing `pro_pro_sparse.npz` or `skill_skill_sparse.npz`
- Run: `python extract.py --data-dir <dir>`

2. Missing `embedding_200.npz`
- Run PEBG first, or set `--embedding-file` to the actual exported file.

3. `ques_skill.csv` format mismatch
- Required columns: `problem_id,skill_id`

4. Slow training
- Check whether PyTorch is using GPU.

## 9. Notes

- The training code is dataset-agnostic after preprocessing.
- Switching datasets only requires changing directory arguments.
