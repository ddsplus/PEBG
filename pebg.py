import argparse
import math
import os

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

        # 3 inputs -> 3 pairwise interactions
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--keep-prob', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume-epoch', type=int, default=0)
    args = parser.parse_args()

    saved_model_folder = os.path.join(args.data_dir, 'pebg_model')
    os.makedirs(saved_model_folder, exist_ok=True)

    pro_skill = sparse.load_npz(os.path.join(args.data_dir, 'pro_skill_sparse.npz')).toarray().astype(np.float32)
    pro_pro = sparse.load_npz(os.path.join(args.data_dir, 'pro_pro_sparse.npz')).toarray().astype(np.float32)
    skill_skill = sparse.load_npz(os.path.join(args.data_dir, 'skill_skill_sparse.npz')).toarray().astype(np.float32)
    pro_feat = np.load(os.path.join(args.data_dir, 'pro_feat.npz'))['pro_feat'].astype(np.float32)

    pro_num, skill_num = pro_skill.shape
    diff_feat_dim = pro_feat.shape[1] - 1

    print(f'problem number {pro_num}, skill number {skill_num}')
    print(f'problem feature shape {pro_feat.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PEBGModel(pro_num, skill_num, diff_feat_dim, args.embed_dim, args.hidden_dim, args.keep_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    if args.resume_epoch > 0:
        ckpt_path = os.path.join(saved_model_folder, f'pebg_{args.resume_epoch}.pt')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

    skill_skill_targets = torch.from_numpy(skill_skill).to(device)
    train_steps = int(math.ceil(pro_num / float(args.batch_size)))

    for epoch in range(args.resume_epoch, args.epochs):
        model.train()
        train_loss = 0.0

        for m in range(train_steps):
            b = m * args.batch_size
            e = min((m + 1) * args.batch_size, pro_num)

            batch_pro = torch.arange(b, e, dtype=torch.long, device=device)
            pro_skill_targets = torch.from_numpy(pro_skill[b:e]).to(device)
            pro_pro_targets = torch.from_numpy(pro_pro[b:e]).to(device)
            diff_feat = torch.from_numpy(pro_feat[b:e, :-1]).to(device)
            auxiliary_targets = torch.from_numpy(pro_feat[b:e, -1]).to(device)

            pro_embed = model.pro_embedding(batch_pro)
            skill_embed_matrix = model.skill_embedding.weight
            pro_embed_matrix = model.pro_embedding.weight
            diff_feat_embed = torch.matmul(diff_feat, model.diff_embedding)

            pro_skill_logits = torch.matmul(pro_embed, skill_embed_matrix.t()).reshape(-1)
            pro_pro_logits = torch.matmul(pro_embed, pro_embed_matrix.t()).reshape(-1)
            skill_skill_logits = torch.matmul(skill_embed_matrix, skill_embed_matrix.t()).reshape(-1)

            loss_pro_skill = bce_loss(pro_skill_logits, pro_skill_targets.reshape(-1))
            loss_pro_pro = bce_loss(pro_pro_logits, pro_pro_targets.reshape(-1))
            loss_skill_skill = bce_loss(skill_skill_logits, skill_skill_targets.reshape(-1))

            skill_sum = torch.sum(pro_skill_targets, dim=1, keepdim=True).clamp_min(1e-8)
            skill_embed = torch.matmul(pro_skill_targets, skill_embed_matrix) / skill_sum
            _, pred_aux = model.pnn_forward([pro_embed, skill_embed, diff_feat_embed])
            loss_aux = mse_loss(pred_aux, auxiliary_targets)

            loss = loss_aux + loss_pro_skill + loss_pro_pro + loss_skill_skill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, train_steps)
        print(f'epoch {epoch}, loss {train_loss:.4f}')

        if epoch + 1 in [50, 100, 200, 500, 1000, 1500, 2000] or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), os.path.join(saved_model_folder, f'pebg_{epoch + 1}.pt'))

    print('finish training')

    model.eval()
    with torch.no_grad():
        pro_repre = model.pro_embedding.weight.detach().cpu().numpy()
        skill_repre = model.skill_embedding.weight.detach().cpu().numpy()

        all_pro = torch.arange(pro_num, dtype=torch.long, device=device)
        all_diff_feat = torch.from_numpy(pro_feat[:, :-1]).to(device)
        all_pro_skill_targets = torch.from_numpy(pro_skill).to(device)

        all_pro_embed = model.pro_embedding(all_pro)
        all_skill_embed = torch.matmul(all_pro_skill_targets, model.skill_embedding.weight) / torch.sum(all_pro_skill_targets, dim=1, keepdim=True).clamp_min(1e-8)
        all_diff_embed = torch.matmul(all_diff_feat, model.diff_embedding)
        pro_final_repre, _ = model.pnn_forward([all_pro_embed, all_skill_embed, all_diff_embed])
        pro_final_repre = pro_final_repre.detach().cpu().numpy()

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
