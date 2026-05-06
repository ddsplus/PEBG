#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess ASSIST2009 dataset.

Input:
  - Data/ASSIST2009/skill_builder_data.csv

Output:
  - Data/data/ASSIST09/ques_skill.csv
  - Data/data/ASSIST09/train_question.txt
  - Data/data/ASSIST09/test_question.txt
  - Data/data/ASSIST09/train_skill.txt
  - Data/data/ASSIST09/test_skill.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_process_data(csv_path):
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, encoding='latin1')
    print(f"Raw shape: {df.shape}")

    required_cols = ['user_id', 'order_id', 'problem_id', 'skill_id', 'correct']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols).copy()
    for col in required_cols:
        df[col] = df[col].astype(int)

    df = df.sort_values(['user_id', 'order_id']).reset_index(drop=True)
    print(f"Processed shape: {df.shape}")
    return df


def create_ques_skill_mapping(df):
    qsm = df[['problem_id', 'skill_id']].drop_duplicates().sort_values('problem_id')
    print(f"Unique problems: {qsm['problem_id'].nunique()}")
    print(f"Unique skills: {qsm['skill_id'].nunique()}")
    return qsm


def group_by_user(df):
    groups = {}
    for uid, g in df.groupby('user_id'):
        groups[uid] = g.sort_values('order_id').reset_index(drop=True)
    print(f"Total users: {len(groups)}")
    return groups


def generate_dataset_files(user_groups, ques_skill_map, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    q2s = dict(zip(ques_skill_map['problem_id'], ques_skill_map['skill_id'].astype(int)))

    user_ids = list(user_groups.keys())
    train_ids, test_ids = train_test_split(user_ids, test_size=0.2, random_state=42)

    stats = {'train': {'total': 0, 'filtered': 0}, 'test': {'total': 0, 'filtered': 0}}
    for split_name, uid_list in [('train', train_ids), ('test', test_ids)]:
        qf = os.path.join(output_dir, f'{split_name}_question.txt')
        sf = os.path.join(output_dir, f'{split_name}_skill.txt')

        with open(qf, 'w', encoding='utf-8') as fq, open(sf, 'w', encoding='utf-8') as fs:
            for uid in uid_list:
                g = user_groups[uid]
                problem_seq = g['problem_id'].astype(int).tolist()
                skill_seq = [int(q2s.get(p, -1)) for p in problem_seq]
                ans_seq = g['correct'].astype(int).tolist()

                if -1 in skill_seq:
                    stats[split_name]['filtered'] += 1
                    continue

                stats[split_name]['total'] += 1

                fq.write(f"{uid}\n")
                fq.write(','.join(map(str, problem_seq)) + '\n')
                fq.write(','.join(map(str, ans_seq)) + '\n')

                fs.write(f"{uid}\n")
                fs.write(','.join(map(str, skill_seq)) + '\n')
                fs.write(','.join(map(str, ans_seq)) + '\n')

        print(f"{split_name}: kept={stats[split_name]['total']}, filtered={stats[split_name]['filtered']}")

    return stats


def save_ques_skill_mapping(ques_skill_map, output_dir):
    out = os.path.join(output_dir, 'ques_skill.csv')
    ques_skill_map.to_csv(out, index=False)
    print(f"Saved: {out}")


def main():
    csv_path = os.path.join(SCRIPT_DIR, 'ASSIST2009', 'skill_builder_data.csv')
    output_dir = os.path.join(SCRIPT_DIR, 'data', 'ASSIST09')

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return

    df = load_and_process_data(csv_path)
    qsm = create_ques_skill_mapping(df)
    user_groups = group_by_user(df)
    stats = generate_dataset_files(user_groups, qsm, output_dir)
    save_ques_skill_mapping(qsm, output_dir)

    total = stats['train']['total'] + stats['test']['total']
    filtered = stats['train']['filtered'] + stats['test']['filtered']
    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'Users kept: {total}, filtered: {filtered}')


if __name__ == '__main__':
    main()
