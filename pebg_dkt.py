import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm


def train_test_split_npz(data_dict, split=0.8):
    n_samples = data_dict['y'].shape[0]
    split_point = int(n_samples * split)
    out = {}
    for k, v in data_dict.items():
        out[k] = (v[:split_point], v[split_point:])
    return out


class PEBGDKT(nn.Module):
    def __init__(self, pro_num, embed_dim, hidden_dim, use_pretrain, pre_pro_embed, train_embed):
        super().__init__()
        self.pro_embedding = nn.Embedding(pro_num + 1, embed_dim, padding_idx=0)
        if use_pretrain and pre_pro_embed is not None:
            self.pro_embedding.weight.data[1:] = torch.from_numpy(pre_pro_embed).float()
        self.pro_embedding.weight.requires_grad = train_embed
        self.lstm = nn.LSTM(input_size=embed_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + embed_dim, 1)

    def forward(self, pro, y, real_seq_len):
        all_pro_embed = self.pro_embedding(pro)
        pro_embed = all_pro_embed[:, :-1, :]
        next_pro_embed = all_pro_embed[:, 1:, :]

        zero_vec = torch.zeros_like(pro_embed)
        y_prev = y[:, :-1].unsqueeze(-1)
        left = torch.where(y_prev > 0, pro_embed, zero_vec)
        right = torch.where(y_prev > 0, zero_vec, pro_embed)
        rnn_inputs = torch.cat([left, right], dim=-1)

        lengths = torch.clamp(real_seq_len, min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(rnn_inputs, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=rnn_inputs.size(1))

        outputs_reshape = outputs.reshape(-1, outputs.size(-1))
        next_pro_reshape = next_pro_embed.reshape(-1, next_pro_embed.size(-1))
        logits = self.fc(torch.cat([outputs_reshape, next_pro_reshape], dim=1)).squeeze(-1)
        return logits


def get_filtered(logits, y_batch):
    targets = y_batch[:, 1:].reshape(-1)
    mask = targets != -1
    return targets[mask], logits[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--embedding-file', default='embedding_200.npz')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use-pretrain', action='store_true')
    parser.add_argument('--train-embed', action='store_true')
    parser.add_argument('--save-root', default='runs', help='Root directory to save per-dataset checkpoints')
    args = parser.parse_args()

    data = np.load(os.path.join(args.data_dir, f'{args.dataset_name}.npz'))
    y = data['y']
    problem = data['problem']
    real_len = data['real_len']
    skill_num = int(data['skill_num'])
    pro_num = int(data['problem_num'])

    splits = train_test_split_npz({'y': y, 'problem': problem, 'real_len': real_len})
    train_y, test_y = splits['y']
    train_problem, test_problem = splits['problem']
    train_real_len, test_real_len = splits['real_len']

    pre_pro_embed = None
    embed_dim = 64
    if args.use_pretrain:
        embed_data = np.load(os.path.join(args.data_dir, args.embedding_file))
        pre_pro_embed = embed_data['pro_final_repre']
        embed_dim = pre_pro_embed.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PEBGDKT(pro_num, embed_dim, args.hidden_dim, args.use_pretrain, pre_pro_embed, args.train_embed).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset_run_dir = os.path.join(args.save_root, args.dataset_name)
    os.makedirs(dataset_run_dir, exist_ok=True)
    best_model_path = os.path.join(dataset_run_dir, 'best_model.pt')
    metric_log_path = os.path.join(dataset_run_dir, 'metrics.csv')

    train_steps = int(math.ceil(train_y.shape[0] / float(args.batch_size)))
    test_steps = int(math.ceil(test_y.shape[0] / float(args.batch_size)))
    best_auc = -1.0
    best_acc = -1.0
    best_epoch = -1

    if not os.path.exists(metric_log_path):
        with open(metric_log_path, 'w', encoding='utf-8') as f:
            f.write('epoch,train_loss,test_auc,test_acc,is_best\n')

    for epoch in tqdm(range(args.epochs)):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_start_time = time.time()

        for j in range(train_steps):
            b, e = j * args.batch_size, (j + 1) * args.batch_size
            batch_y = torch.from_numpy(train_y[b:e].astype(np.float32)).to(device)
            batch_pro = torch.from_numpy(train_problem[b:e].astype(np.int64)).to(device)
            batch_real_len = torch.from_numpy((train_real_len[b:e] - 1).astype(np.int64)).to(device)

            logits = model(batch_pro, batch_y, batch_real_len)
            filtered_targets, filtered_logits = get_filtered(logits, batch_y)
            if filtered_targets.numel() == 0:
                continue

            loss = criterion(filtered_logits, filtered_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, train_steps)
        train_time = time.time() - train_start_time

        model.eval()
        test_start_time = time.time()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for j in range(test_steps):
                b, e = j * args.batch_size, (j + 1) * args.batch_size
                test_y_ = torch.from_numpy(test_y[b:e].astype(np.float32)).to(device)
                test_pro_ = torch.from_numpy(test_problem[b:e].astype(np.int64)).to(device)
                test_real_len_ = torch.from_numpy((test_real_len[b:e] - 1).astype(np.int64)).to(device)

                logits = model(test_pro_, test_y_, test_real_len_)
                targets_, logits_ = get_filtered(logits, test_y_)
                if targets_.numel() == 0:
                    continue
                preds_ = torch.sigmoid(logits_)
                test_preds.append(preds_.cpu().numpy())
                test_targets.append(targets_.cpu().numpy())

        test_time = time.time() - test_start_time

        if len(test_preds) == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, train loss:{train_loss:.5f}, train_time {train_time:.2f}s, test_time {test_time:.2f}s, no valid test samples.')
            with open(metric_log_path, 'a', encoding='utf-8') as f:
                f.write(f'{epoch + 1},{train_loss:.6f},nan,nan,0\n')
            continue

        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        test_auc = metrics.roc_auc_score(test_targets, test_preds)
        test_binary = (test_preds > 0.5).astype(np.float32)
        test_acc = metrics.accuracy_score(test_targets, test_binary)
        is_best = 0

        if test_auc > best_auc:
            best_auc = float(test_auc)
            best_acc = float(test_acc)
            best_epoch = epoch + 1
            torch.save(
                {
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc,
                    'best_acc': best_acc,
                    'args': vars(args),
                },
                best_model_path,
            )
            is_best = 1

        print(f'Epoch {epoch + 1}/{args.epochs}, train loss:{train_loss:.5f}, train_time {train_time:.2f}s, test_time {test_time:.2f}s, test auc:{test_auc:.6f}, test acc:{test_acc:.5f}')
        with open(metric_log_path, 'a', encoding='utf-8') as f:
            f.write(f'{epoch + 1},{train_loss:.6f},{test_auc:.6f},{test_acc:.6f},{is_best}\n')

    if best_epoch > 0:
        print(f'Best model saved: {best_model_path}')
        print(f'Best epoch: {best_epoch}, best auc: {best_auc:.6f}, best acc: {best_acc:.6f}')
    else:
        print('No valid test metrics were produced; best model was not saved.')


if __name__ == '__main__':
    main()
