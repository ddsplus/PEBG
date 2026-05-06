import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm


def train_test_split(data, split=0.8):
    n_samples = data[0].shape[0]
    split_point = int(n_samples * split)
    train_data, test_data = [], []
    for d in data:
        train_data.append(d[:split_point])
        test_data.append(d[split_point:])
    return train_data, test_data


# data
data_folder = 'assist09'
data = np.load(os.path.join(data_folder, data_folder + '.npz'))
y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']
skill_num, pro_num = data['skill_num'], data['problem_num']
print('problem number %d, skill number %d' % (pro_num, skill_num))

# divide train test set
train_data, test_data = train_test_split([y, skill, problem, real_len])   # [y, skill, pro, real_len]
train_y, train_skill, train_problem, train_real_len = train_data[0], train_data[1], train_data[2], train_data[3]
test_y, test_skill, test_problem, test_real_len = test_data[0], test_data[1], test_data[2], test_data[3]

# embed data, used for initialize
embed_data = np.load(os.path.join(data_folder, 'embedding_200.npz'))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print(pre_pro_embed.shape, pre_pro_embed.dtype)


# hyper-params
epochs = 200
bs = 128
embed_dim = pre_pro_embed.shape[1]
hidden_dim = 128
lr = 0.001
use_pretrain = True
train_embed = False
train_flag = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PEBGDKT(nn.Module):
    def __init__(self, pro_num, embed_dim, hidden_dim, use_pretrain, pre_pro_embed, train_embed):
        super().__init__()
        self.embed_dim = embed_dim

        self.pro_embedding = nn.Embedding(pro_num + 1, embed_dim, padding_idx=0)
        if use_pretrain:
            print('use pretrain embedding matrix')
            self.pro_embedding.weight.data[1:] = torch.from_numpy(pre_pro_embed).float()
        else:
            print('use random init embedding matrix')
            nn.init.trunc_normal_(self.pro_embedding.weight[1:], std=0.1)

        self.pro_embedding.weight.requires_grad = train_embed
        self.lstm = nn.LSTM(input_size=embed_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + embed_dim, 1)

    def forward(self, pro, y, real_seq_len):
        all_pro_embed = self.pro_embedding(pro)
        pro_embed = all_pro_embed[:, :-1, :]
        next_pro_embed = all_pro_embed[:, 1:, :]

        zero_vec = torch.zeros_like(pro_embed)
        y_prev = y[:, :-1].unsqueeze(-1)
        # correct: [embed, 0], wrong: [0, embed]
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


model = PEBGDKT(pro_num, embed_dim, hidden_dim, use_pretrain, pre_pro_embed, train_embed).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
criterion = nn.BCEWithLogitsLoss()


def get_filtered(logits, y_batch):
    targets = y_batch[:, 1:].reshape(-1)
    mask = targets != -1
    filtered_targets = targets[mask]
    filtered_logits = logits[mask]
    return filtered_targets, filtered_logits


if train_flag:
    train_steps = int(math.ceil(train_skill.shape[0] / float(bs)))
    test_steps = int(math.ceil(test_skill.shape[0] / float(bs)))

    best_auc = best_acc = 0
    for i in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0

        for j in range(train_steps):
            batch_y = torch.from_numpy(train_y[j * bs:(j + 1) * bs, :].astype(np.float32)).to(device)
            batch_pro = torch.from_numpy(train_problem[j * bs:(j + 1) * bs, :].astype(np.int64)).to(device)
            batch_real_len = torch.from_numpy((train_real_len[j * bs:(j + 1) * bs] - 1).astype(np.int64)).to(device)

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

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for j in range(test_steps):
                test_y_ = torch.from_numpy(test_y[j * bs:(j + 1) * bs, :].astype(np.float32)).to(device)
                test_pro_ = torch.from_numpy(test_problem[j * bs:(j + 1) * bs, :].astype(np.int64)).to(device)
                test_real_len_ = torch.from_numpy((test_real_len[j * bs:(j + 1) * bs] - 1).astype(np.int64)).to(device)

                logits = model(test_pro_, test_y_, test_real_len_)
                targets_, logits_ = get_filtered(logits, test_y_)
                if targets_.numel() == 0:
                    continue
                preds_ = torch.sigmoid(logits_)

                test_preds.append(preds_.cpu().numpy())
                test_targets.append(targets_.cpu().numpy())

        if len(test_preds) == 0:
            print('Epoch %d/%d, no valid test samples after filtering.' % (i + 1, epochs))
            continue

        test_preds = np.concatenate(test_preds, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        test_auc = metrics.roc_auc_score(test_targets, test_preds)
        test_binary = (test_preds > 0.5).astype(np.float32)
        test_acc = metrics.accuracy_score(test_targets, test_binary)

        records = 'Epoch %d/%d, train loss:%3.5f, test auc:%f, test acc:%3.5f' % \
                  (i + 1, epochs, train_loss, test_auc, test_acc)
        print(records)

        # if best_auc < test_auc:
        #     best_auc = test_auc
        #     torch.save(model.state_dict(), os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.pt'))
else:
    ckpt = os.path.join(data_folder, 'bekt_dkt_model/dkt_pretrain.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    pro_embed_trained = model.pro_embedding.weight.detach().cpu().numpy()[1:]
    np.savez(os.path.join(data_folder, 'bekt_dkt_model/pro_embed_bekt_dkt.npz'), pro_final_repre=pro_embed_trained)
