"""
    bipartite graph node embedding --> item embedding, skill embedding
       (item is question.)

    plus: item difficutly features

    three different feature use PNN to fuse, and with the help of auxilary target
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse


# load data

data_folder = 'ednet'
if data_folder == 'assist09':
    con_sym = '-'
elif data_folder == 'ednet':
    con_sym = ';'
elif data_folder == 'assist12':
    con_sym = '$$$'
else:
    print('no such dataset!')
    exit()

saved_model_folder = os.path.join(data_folder, 'pebg_model')
if not os.path.exists(saved_model_folder):
    os.mkdir(saved_model_folder)


pro_skill_coo = sparse.load_npz(os.path.join(data_folder, 'pro_skill_sparse.npz'))
skill_skill_coo = sparse.load_npz(os.path.join(data_folder, 'skill_skill_sparse.npz'))
pro_pro_coo = sparse.load_npz(os.path.join(data_folder, 'pro_pro_sparse.npz'))
[pro_num, skill_num] = pro_skill_coo.shape
print('problem number %d, skill number %d' % (pro_num, skill_num))
print('pro-skill edge %d, pro-pro edge %d, skill-skill edge %d' % (pro_skill_coo.nnz, pro_pro_coo.nnz, skill_skill_coo.nnz))

pro_skill_dense = pro_skill_coo.toarray().astype(np.float32)
pro_pro_dense = pro_pro_coo.toarray().astype(np.float32)
skill_skill_dense = skill_skill_coo.toarray().astype(np.float32)

pro_feat = np.load(os.path.join(data_folder, 'pro_feat.npz'))['pro_feat']    # [pro_diff_feat, auxiliary_target]
print('problem feature shape', pro_feat.shape)
print(pro_feat[:, 0].min(), pro_feat[:, 0].max())
print(pro_feat[:, 1].min(), pro_feat[:, 1].max())

diff_feat_dim = pro_feat.shape[1] - 1
embed_dim = 64      # node embedding dim in bipartite
hidden_dim = 128    # hidden dim in PNN
keep_prob = 0.5
lr = 0.001
bs = 256
epochs = 200
model_flag = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


model = PEBGModel(pro_num, skill_num, diff_feat_dim, embed_dim, hidden_dim, keep_prob).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

skill_skill_targets = torch.from_numpy(skill_skill_dense).to(device)

print('finish building graph')

# begin train
train_steps = int(math.ceil(pro_num / float(bs)))
if model_flag > 0:
    ckpt_path = os.path.join(saved_model_folder, 'pebg_%d.pt' % model_flag)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

for i in range(model_flag, epochs):
    model.train()
    train_loss = 0.0
    for m in range(train_steps):
        b, e = m * bs, min((m + 1) * bs, pro_num)

        batch_pro = np.arange(b, e).astype(np.int64)
        batch_pro_skill_targets = pro_skill_dense[b:e, :]
        batch_pro_pro_targets = pro_pro_dense[b:e, :]
        batch_diff_feat = pro_feat[b:e, :-1].astype(np.float32)
        batch_auxiliary_targets = pro_feat[b:e, -1].astype(np.float32)

        pro_idx = torch.from_numpy(batch_pro).to(device)
        pro_skill_targets = torch.from_numpy(batch_pro_skill_targets).to(device)
        pro_pro_targets = torch.from_numpy(batch_pro_pro_targets).to(device)
        diff_feat = torch.from_numpy(batch_diff_feat).to(device)
        auxiliary_targets = torch.from_numpy(batch_auxiliary_targets).to(device)

        pro_embed = model.pro_embedding(pro_idx)
        skill_embed_matrix = model.skill_embedding.weight
        pro_embed_matrix = model.pro_embedding.weight
        diff_feat_embed = torch.matmul(diff_feat, model.diff_embedding)

        pro_skill_logits = torch.matmul(pro_embed, skill_embed_matrix.t()).reshape(-1)
        pro_pro_logits = torch.matmul(pro_embed, pro_embed_matrix.t()).reshape(-1)
        skill_skill_logits = torch.matmul(skill_embed_matrix, skill_embed_matrix.t()).reshape(-1)

        cross_entropy_pro_skill = bce_loss(pro_skill_logits, pro_skill_targets.reshape(-1))
        cross_entropy_pro_pro = bce_loss(pro_pro_logits, pro_pro_targets.reshape(-1))
        cross_entropy_skill_skill = bce_loss(skill_skill_logits, skill_skill_targets.reshape(-1))

        skill_sum = torch.sum(pro_skill_targets, dim=1, keepdim=True).clamp_min(1e-8)
        skill_embed = torch.matmul(pro_skill_targets, skill_embed_matrix) / skill_sum
        _, p = model.pnn_forward([pro_embed, skill_embed, diff_feat_embed])
        mse = mse_loss(p, auxiliary_targets)

        loss = mse + cross_entropy_pro_skill + cross_entropy_pro_pro + cross_entropy_skill_skill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= train_steps
    print('epoch %d, loss %.4f' % (i, train_loss))

    if i + 1 in [50, 100, 200, 500, 1000, 1500, 2000]:
        torch.save(model.state_dict(), os.path.join(saved_model_folder, 'pebg_%d.pt' % (i + 1)))

print('finish training')

# store pretrained pro skill embedding
model.eval()
with torch.no_grad():
    pro_repre = model.pro_embedding.weight.detach().cpu().numpy()
    skill_repre = model.skill_embedding.weight.detach().cpu().numpy()
    print(pro_repre.shape, skill_repre.shape)

    all_pro = torch.arange(pro_num, dtype=torch.long, device=device)
    all_diff_feat = torch.from_numpy(pro_feat[:, :-1].astype(np.float32)).to(device)
    all_pro_skill_targets = torch.from_numpy(pro_skill_dense).to(device)

    all_pro_embed = model.pro_embedding(all_pro)
    all_skill_embed = torch.matmul(all_pro_skill_targets, model.skill_embedding.weight) / \
        torch.sum(all_pro_skill_targets, dim=1, keepdim=True).clamp_min(1e-8)
    all_diff_embed = torch.matmul(all_diff_feat, model.diff_embedding)
    pro_final_repre, _ = model.pnn_forward([all_pro_embed, all_skill_embed, all_diff_embed])
    pro_final_repre = pro_final_repre.detach().cpu().numpy()
    print(pro_final_repre.shape)

with open(os.path.join(data_folder, 'skill_id_dict.txt'), 'r') as f:
    skill_id_dict = eval(f.read())
join_skill_num = len(skill_id_dict)
print('original skill number %d, joint skill number %d' % (skill_num, join_skill_num))

skill_repre_new = np.zeros([join_skill_num, skill_repre.shape[1]])
skill_repre_new[:skill_num, :] = skill_repre
for s in skill_id_dict.keys():
    if con_sym in str(s):
        tmp_skill_id = skill_id_dict[s]
        tmp_skills = [skill_id_dict[ele] for ele in s.split(con_sym)]
        skill_repre_new[tmp_skill_id, :] = np.mean(skill_repre[tmp_skills], axis=0)

np.savez(os.path.join(data_folder, 'embedding_%d.npz' % epochs),
         pro_repre=pro_repre, skill_repre=skill_repre_new, pro_final_repre=pro_final_repre)
