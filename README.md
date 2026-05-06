# PEBG

This repository contains the source code of the mdoel PEBG in our paper "Improving Knowledge Tracing via Pre-training Question Embeddings", which is accepted by IJCAI 2020.

## Data Preparation
### Assist09
- data_assist09.py: Data pre-process. You should download original assist09 dataset from [here](https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view). Details are [here](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010).

### EdNet
- ednet/data.py: Data pre-process for EdNet dataset. 
we use the 'ednet-kt1' dataset. You can download it [here](https://drive.google.com/file/d/1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw/view). The question information file is [here](https://drive.google.com/file/d/117aYJAWG3GU48suS66NPaB82HwFj6xWS/view). For more information about [EdNet](https://github.com/riiid/ednet)

Once you have the ednet-kt1 dataset, you can enter the folder "ednet" and run 'data.py' to pre-process EdNet dataset.


## Model Code
- extract.py: Extract the implicit similarity between questions and skills.
- PNN.py: Implement the product layer.
- pebg.py: The PEBG model.
- pebg_dkt.py: The PEBG+DKT model. 

## Runtime Dependencies
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- tqdm

## How To Train
The project now uses PyTorch.

### 1) Prepare Data
1. Prepare Assist09 or EdNet raw data as described above.
2. Run preprocessing scripts:
- Assist09: `python data_assist09.py`
- EdNet: `python ednet/data.py`
3. Build graph relation files (problem-skill / problem-problem / skill-skill):
- `python extract.py`

After this step, the dataset folder (for example `assist09/` or `ednet/`) should contain files like:
- `<dataset>.npz`
- `pro_feat.npz`
- `pro_skill_sparse.npz`
- `pro_pro_sparse.npz`
- `skill_skill_sparse.npz`
- `skill_id_dict.txt`

### 2) Train PEBG (Question Embedding Pretraining)
Run:

```bash
python pebg.py
```

Notes:
- Set `data_folder` in `pebg.py` (default is `ednet`).
- This step saves pretrained embeddings to:
- `<data_folder>/embedding_200.npz`
- Model checkpoints are saved to:
- `<data_folder>/pebg_model/pebg_*.pt`

### 3) Train PEBG+DKT
Run:

```bash
python pebg_dkt.py
```

Notes:
- Set `data_folder` in `pebg_dkt.py` (default is `assist09`).
- The script loads pretrained embedding from:
- `<data_folder>/embedding_200.npz`
- During training, it prints test AUC and ACC each epoch.

### 4) Recommended Training Order
1. Preprocess data (`data_assist09.py` or `ednet/data.py`)
2. Build relation graph (`extract.py`)
3. Pretrain embeddings (`pebg.py`)
4. Train KT model (`pebg_dkt.py`)

### 5) Quick Configuration Tips
- In `pebg.py`:
- `epochs`, `bs`, `lr`, `embed_dim`, `hidden_dim`, `data_folder`
- In `pebg_dkt.py`:
- `epochs`, `bs`, `lr`, `hidden_dim`, `data_folder`, `use_pretrain`, `train_embed`
- For faster debug, reduce `epochs` first (for example 5-10).



If you need more information about our experiments, you can contact us. 
email: liuyunfei@sjtu.edu.cn
