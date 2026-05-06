# PEBG (PyTorch)

This project is now organized into **3 steps only**:

1. Data Preprocessing
2. Pretraining (PEBG)
3. Training (PEBG+DKT)

Supported datasets:
- `assist09`
- `assist17`
- `statics2011`
- `xes3g5m`

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- pandas
- scikit-learn
- tqdm

Install:

```bash
pip install torch numpy scipy pandas scikit-learn tqdm
```

## Step 1: Data Preprocessing

Use the unified preprocessing entry:

```bash
python preprocess.py --dataset <dataset_name>
```

Examples:

```bash
python preprocess.py --dataset assist09
python preprocess.py --dataset assist17
python preprocess.py --dataset statics2011
python preprocess.py --dataset xes3g5m
```

What Step 1 does internally:
- Runs dataset-specific raw preprocessing script in `Data/`
- Converts intermediate files to PEBG format (`prepare_dataset.py`)
- Builds graph similarity files (`extract.py`)

Step 1 output directory:
- `datasets/assist09`
- `datasets/assist17`
- `datasets/statics2011`
- `datasets/xes3g5m`

Each prepared dataset directory includes:
- `<dataset_name>.npz`
- `pro_feat.npz`
- `pro_skill_sparse.npz`
- `pro_pro_sparse.npz`
- `skill_skill_sparse.npz`
- `skill_id_dict.txt`

Optional args for Step 1:

```bash
python preprocess.py --dataset assist17 --max-len 200 --min-len 3 --split all
```

## Step 2: Pretraining (PEBG)

Run PEBG pretraining:

```bash
python pebg.py --data-dir datasets/<dataset_name> --epochs 200
```

Example:

```bash
python pebg.py --data-dir datasets/assist17 --epochs 200
```

Outputs:
- `datasets/<dataset_name>/embedding_200.npz`
- `datasets/<dataset_name>/pebg_model/pebg_*.pt`

For large datasets, start with faster settings:

```bash
python pebg.py --data-dir datasets/assist09 --epochs 20 --batch-size 2048 --neg-k 10 --skill-sample-size 512
```

## Step 3: Training (PEBG+DKT)

Train downstream KT model:

```bash
python pebg_dkt.py \
  --data-dir datasets/<dataset_name> \
  --dataset-name <dataset_name> \
  --use-pretrain \
  --embedding-file embedding_200.npz \
  --epochs 200 \
  --save-root runs
```

Example:

```bash
python pebg_dkt.py --data-dir datasets/assist17 --dataset-name assist17 --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

Per epoch output:
- train loss
- test AUC
- test ACC

Saved model/logs (per dataset):
- `runs/<dataset_name>/best_model.pt`
- `runs/<dataset_name>/metrics.csv`

`best_model.pt` is selected by best test AUC.

## Minimal End-to-End Example

Using `assist17`:

```bash
python preprocess.py --dataset assist17
python pebg.py --data-dir datasets/assist17 --epochs 200
python pebg_dkt.py --data-dir datasets/assist17 --dataset-name assist17 --use-pretrain --embedding-file embedding_200.npz --epochs 200 --save-root runs
```

## Common Issues

1. Missing raw CSV files in `Data/<DATASET_DIR>/`
- Put dataset files into expected locations under `Data/` first.

2. Missing `embedding_200.npz`
- Finish Step 2 before Step 3.
- If you used different pretrain epochs, set matching `--embedding-file`.

3. Slow pretraining on very large datasets
- Reduce epochs for smoke test
- Increase `--batch-size` if memory allows
- Reduce `--neg-k` and `--skill-sample-size`
