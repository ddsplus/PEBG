import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


class PEBGModel(nn.Module):
    def __init__(self, pro_num, skill_num, diff_feat_dim, embed_dim, hidden_dim, keep_prob):
        super().__init__()
        self.pro_embedding = nn.Embedding(pro_num, embed_dim)
        self.skill_embedding = nn.Embedding(skill_num, embed_dim)
        self.diff_embedding = nn.Parameter(torch.randn(diff_feat_dim, embed_dim) * 0.1)

        self.row = [0, 0, 1]
        self.col = [1, 2, 2]
        self.fc1 = nn.Linear(3 * embed_dim + 3, hidden_dim)
        self.dropout = nn.Dropout(p=1 - keep_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def pnn_forward(self, inputs):
        xw = torch.cat(inputs, dim=1)
        xw3d = xw.view(-1, 3, xw.shape[1] // 3)
        p = xw3d[:, self.row, :]
        q = xw3d[:, self.col, :]
        ip = torch.sum(p * q, dim=-1)
        l = torch.cat([xw, ip], dim=1)
        h = torch.relu(self.fc1(l))
        h = self.dropout(h)
        pred = self.fc2(h).squeeze(-1)
        return h, pred


def load_skill_id_dict(path, fallback_skill_num):
    if not os.path.exists(path):
        return {str(i): i for i in range(fallback_skill_num)}
    with open(path, 'r', encoding='utf-8') as f:
        return eval(f.read())


def sample_pair_loss(src_embed, dst_embed, pos_dst_idx, dst_size, neg_k):
    # src_embed: [N, D], pos_dst_idx: [N]
    n = src_embed.size(0)
    device = src_embed.device
    pos_dst = dst_embed(pos_dst_idx)
    pos_logits = torch.sum(src_embed * pos_dst, dim=1)
    pos_labels = torch.ones_like(pos_logits)

    neg_dst_idx = torch.randint(low=0, high=dst_size, size=(n, neg_k), device=device)
    neg_dst = dst_embed(neg_dst_idx)  # [N, K, D]
    src_expand = src_embed.unsqueeze(1)  # [N, 1, D]
    neg_logits = torch.sum(src_expand * neg_dst, dim=2)  # [N, K]
    neg_labels = torch.zeros_like(neg_logits)

    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def choose_positive_from_row(csr, row_ids, fallback_max):
    pos = np.empty(len(row_ids), dtype=np.int64)
    for i, rid in enumerate(row_ids):
        cols = csr.indices[csr.indptr[rid] : csr.indptr[rid + 1]]
        if cols.size == 0:
            pos[i] = np.random.randint(0, fallback_max)
        else:
            pos[i] = cols[np.random.randint(0, cols.size)]
    return pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--keep-prob', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume-epoch', type=int, default=0)
    parser.add_argument('--neg-k', type=int, default=10, help='negative samples per positive edge')
    parser.add_argument('--skill-sample-size', type=int, default=512, help='sampled skills per step for skill-skill loss')
    args = parser.parse_args()

    saved_model_folder = os.path.join(args.data_dir, 'pebg_model')
    os.makedirs(saved_model_folder, exist_ok=True)

    pro_skill_csr = sparse.load_npz(os.path.join(args.data_dir, 'pro_skill_sparse.npz')).tocsr()
    pro_pro_csr = sparse.load_npz(os.path.join(args.data_dir, 'pro_pro_sparse.npz')).tocsr()
    skill_skill_csr = sparse.load_npz(os.path.join(args.data_dir, 'skill_skill_sparse.npz')).tocsr()
    pro_feat = np.load(os.path.join(args.data_dir, 'pro_feat.npz'))['pro_feat'].astype(np.float32)

    pro_num, skill_num = pro_skill_csr.shape
    diff_feat_dim = pro_feat.shape[1] - 1

    print(f'problem number {pro_num}, skill number {skill_num}')
    print(f'problem feature shape {pro_feat.shape}')
    print(f'fast sampled training: neg_k={args.neg_k}, batch_size={args.batch_size}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PEBGModel(pro_num, skill_num, diff_feat_dim, args.embed_dim, args.hidden_dim, args.keep_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    if args.resume_epoch > 0:
        ckpt_path = os.path.join(saved_model_folder, f'pebg_{args.resume_epoch}.pt')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

    train_steps = int(math.ceil(pro_num / float(args.batch_size)))

    all_pro_ids = np.arange(pro_num, dtype=np.int64)
    for epoch in range(args.resume_epoch, args.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_start_time = time.time()

        np.random.shuffle(all_pro_ids)
        for m in range(train_steps):
            b = m * args.batch_size
            e = min((m + 1) * args.batch_size, pro_num)
            batch_np = all_pro_ids[b:e]

            # Sample one positive edge for each relation per source node
            ps_pos = choose_positive_from_row(pro_skill_csr, batch_np, skill_num)
            pp_pos = choose_positive_from_row(pro_pro_csr, batch_np, pro_num)

            skill_sample_size = min(args.skill_sample_size, skill_num)
            skill_src_np = np.random.randint(0, skill_num, size=(skill_sample_size,), dtype=np.int64)
            ss_pos = choose_positive_from_row(skill_skill_csr, skill_src_np, skill_num)

            batch_pro = torch.from_numpy(batch_np).to(device)
            ps_pos_t = torch.from_numpy(ps_pos).to(device)
            pp_pos_t = torch.from_numpy(pp_pos).to(device)
            skill_src_t = torch.from_numpy(skill_src_np).to(device)
            ss_pos_t = torch.from_numpy(ss_pos).to(device)

            diff_feat = torch.from_numpy(pro_feat[batch_np, :-1]).to(device)
            auxiliary_targets = torch.from_numpy(pro_feat[batch_np, -1]).to(device)

            pro_embed = model.pro_embedding(batch_pro)
            diff_feat_embed = torch.matmul(diff_feat, model.diff_embedding)

            # graph reconstruction losses via sampled edges
            loss_pro_skill = sample_pair_loss(pro_embed, model.skill_embedding, ps_pos_t, skill_num, args.neg_k)
            loss_pro_pro = sample_pair_loss(pro_embed, model.pro_embedding, pp_pos_t, pro_num, args.neg_k)
            skill_src_embed = model.skill_embedding(skill_src_t)
            loss_skill_skill = sample_pair_loss(skill_src_embed, model.skill_embedding, ss_pos_t, skill_num, args.neg_k)

            # PNN auxiliary regression
            skill_embed = model.skill_embedding(ps_pos_t)
            _, pred_aux = model.pnn_forward([pro_embed, skill_embed, diff_feat_embed])
            loss_aux = mse_loss(pred_aux, auxiliary_targets)

            loss = loss_aux + loss_pro_skill + loss_pro_pro + loss_skill_skill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, train_steps)
        train_time = time.time() - train_start_time

        print(f'epoch {epoch}, loss {train_loss:.4f}, train_time {train_time:.2f}s')

        if epoch + 1 in [50, 100, 200, 500, 1000, 1500, 2000] or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), os.path.join(saved_model_folder, f'pebg_{epoch + 1}.pt'))

    print('finish training')

    eval_start_time = time.time()
    model.eval()
    with torch.no_grad():
        pro_repre = model.pro_embedding.weight.detach().cpu().numpy()
        skill_repre = model.skill_embedding.weight.detach().cpu().numpy()

        # Efficient skill aggregation using sparse matrix multiplication.
        skill_sum = np.asarray(pro_skill_csr.sum(axis=1)).reshape(-1, 1).astype(np.float32)
        skill_sum[skill_sum == 0.0] = 1.0
        mean_skill_embed = (pro_skill_csr @ skill_repre).astype(np.float32) / skill_sum

        all_pro_embed = torch.from_numpy(pro_repre).to(device)
        all_skill_embed = torch.from_numpy(mean_skill_embed).to(device)
        all_diff_feat = torch.from_numpy(pro_feat[:, :-1]).to(device)
        all_diff_embed = torch.matmul(all_diff_feat, model.diff_embedding)

        pro_final_repre, _ = model.pnn_forward([all_pro_embed, all_skill_embed, all_diff_embed])
        pro_final_repre = pro_final_repre.detach().cpu().numpy()

    eval_time = time.time() - eval_start_time
    print(f'evaluation time {eval_time:.2f}s')

    skill_id_dict = load_skill_id_dict(os.path.join(args.data_dir, 'skill_id_dict.txt'), skill_num)
    join_skill_num = len(skill_id_dict)
    skill_repre_new = np.zeros((join_skill_num, skill_repre.shape[1]), dtype=np.float32)
    skill_repre_new[: min(skill_num, join_skill_num), :] = skill_repre[: min(skill_num, join_skill_num), :]

    np.savez(
        os.path.join(args.data_dir, f'embedding_{args.epochs}.npz'),
        pro_repre=pro_repre,
        skill_repre=skill_repre_new,
        pro_final_repre=pro_final_repre,
    )
    print(f'saved embedding_{args.epochs}.npz')


if __name__ == '__main__':
    main()
