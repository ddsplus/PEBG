import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


def read_three_line_sequences(path: str) -> List[Tuple[str, List[int], List[int]]]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f]

    i = 0
    while i + 2 < len(lines):
        user_id = lines[i]
        seq_line = lines[i + 1]
        ans_line = lines[i + 2]
        i += 3

        if not seq_line or not ans_line:
            continue

        seq = [int(x) for x in seq_line.split(',') if x != '']
        ans = [int(x) for x in ans_line.split(',') if x != '']
        L = min(len(seq), len(ans))
        if L == 0:
            continue
        records.append((user_id, seq[:L], ans[:L]))
    return records


def build_qs_map(path: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    if 'problem_id' not in cols or 'skill_id' not in cols:
        raise ValueError(f'ques_skill.csv must contain problem_id, skill_id, got {cols}')
    out = {}
    for _, r in df.iterrows():
        out[int(r['problem_id'])] = int(r['skill_id'])
    return out


def pad_sequences(records: List[Tuple[str, List[int], List[int]]], max_len: int, min_len: int):
    y, problem, real_len = [], [], []
    for _, pro_seq, ans_seq in records:
        if len(pro_seq) < min_len:
            continue
        pro_seq = pro_seq[:max_len]
        ans_seq = ans_seq[:max_len]
        rl = len(pro_seq)

        pro_pad = pro_seq + [0] * (max_len - rl)
        ans_pad = ans_seq + [-1] * (max_len - rl)

        problem.append(pro_pad)
        y.append(ans_pad)
        real_len.append(rl)

    return (
        np.asarray(problem, dtype=np.int32),
        np.asarray(y, dtype=np.float32),
        np.asarray(real_len, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description='Build PEBG-compatible dataset from DiffuQKT-style files.')
    parser.add_argument('--input-dir', required=True, help='Directory containing train/test_question.txt and ques_skill.csv')
    parser.add_argument('--output-dir', required=True, help='Output dataset directory')
    parser.add_argument('--dataset-name', required=True, help='Output npz name, e.g. assist17')
    parser.add_argument('--max-len', type=int, default=200)
    parser.add_argument('--min-len', type=int, default=3)
    parser.add_argument('--split', choices=['train', 'all'], default='all', help='Use only train split or train+test')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_q_path = os.path.join(args.input_dir, 'train_question.txt')
    test_q_path = os.path.join(args.input_dir, 'test_question.txt')
    qs_map_path = os.path.join(args.input_dir, 'ques_skill.csv')

    train_records = read_three_line_sequences(train_q_path)
    test_records = read_three_line_sequences(test_q_path) if os.path.exists(test_q_path) else []
    records = train_records if args.split == 'train' else train_records + test_records

    if len(records) == 0:
        raise ValueError('No usable records found.')

    qs_map = build_qs_map(qs_map_path)
    all_problems = set(qs_map.keys())
    all_skills = set(qs_map.values())

    # Add UNK bucket if question appears in sequences but not in mapping.
    for _, pro_seq, _ in records:
        all_problems.update(pro_seq)

    pro_num = max(all_problems) + 1
    skill_num = max(all_skills) + 1 if all_skills else 1

    # Build pro-skill sparse matrix.
    rows, cols, data = [], [], []
    for p in range(pro_num):
        s = qs_map.get(p, None)
        if s is None:
            continue
        rows.append(p)
        cols.append(s)
        data.append(1.0)
    pro_skill_sparse = sparse.coo_matrix((np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))), shape=(pro_num, skill_num))
    sparse.save_npz(os.path.join(args.output_dir, 'pro_skill_sparse.npz'), pro_skill_sparse)

    # Problem features: [difficulty_like_feature, auxiliary_target]
    # We use (1-correct_rate, correct_rate) from sequence interactions.
    correct_sum = np.zeros(pro_num, dtype=np.float64)
    cnt = np.zeros(pro_num, dtype=np.float64)
    for _, pro_seq, ans_seq in records:
        for p, a in zip(pro_seq, ans_seq):
            if a < 0:
                continue
            correct_sum[p] += a
            cnt[p] += 1

    rate = np.zeros(pro_num, dtype=np.float32)
    seen = cnt > 0
    rate[seen] = (correct_sum[seen] / cnt[seen]).astype(np.float32)
    rate[~seen] = 0.5

    diff = (1.0 - rate).astype(np.float32)
    pro_feat = np.stack([diff, rate], axis=1)
    np.savez(os.path.join(args.output_dir, 'pro_feat.npz'), pro_feat=pro_feat)

    # Build sequence npz for DKT.
    problem, y, real_len = pad_sequences(records, max_len=args.max_len, min_len=args.min_len)

    # Build skill sequence (aligned with question sequence).
    skill = np.zeros_like(problem, dtype=np.int32)
    for i in range(problem.shape[0]):
        for t in range(problem.shape[1]):
            p = int(problem[i, t])
            if p == 0 and (t >= real_len[i]):
                skill[i, t] = 0
                continue
            s = qs_map.get(p, 0)
            skill[i, t] = int(s)

    np.savez(
        os.path.join(args.output_dir, f'{args.dataset_name}.npz'),
        problem=problem,
        y=y,
        skill=skill,
        real_len=real_len,
        skill_num=np.int64(skill_num),
        problem_num=np.int64(pro_num),
    )

    # Save skill id dict for compatibility with pebg export step.
    skill_id_dict = {str(i): int(i) for i in range(skill_num)}
    with open(os.path.join(args.output_dir, 'skill_id_dict.txt'), 'w', encoding='utf-8') as f:
        f.write(str(skill_id_dict))

    print(f'Built dataset at {args.output_dir}')
    print(f'users={problem.shape[0]}, pro_num={pro_num}, skill_num={skill_num}')


if __name__ == '__main__':
    main()
