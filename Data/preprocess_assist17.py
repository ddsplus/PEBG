#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess ASSIST2017 dataset.

Input:
  - Data/ASSIST2017/anonymized_full_release_competition_dataset.csv

Output:
  - Data/data/ASSIST17/ques_skill.csv
  - Data/data/ASSIST17/train_question.txt
  - Data/data/ASSIST17/test_question.txt
  - Data/data/ASSIST17/train_skill.txt
  - Data/data/ASSIST17/test_skill.txt
  - Data/data/ASSIST17/problem_map.txt
  - Data/data/ASSIST17/skill_map.txt
"""

import os
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_process_data(csv_path):
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"Raw shape: {df.shape}")

    required = ['studentId', 'startTime', 'problemId', 'skill', 'correct']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(['studentId', 'startTime']).reset_index(drop=True)
    return df


def create_ques_skill_mapping(df):
    qsm = df[['problemId', 'skill']].drop_duplicates().sort_values('problemId')
    print(f"Unique problems: {qsm['problemId'].nunique()}")
    print(f"Unique skills: {qsm['skill'].nunique()}")
    return qsm


def group_by_user(df):
    user_interactions = defaultdict(list)
    for _, row in df.iterrows():
        uid = int(row['studentId'])
        user_interactions[uid].append(
            {
                'problem_id': int(row['problemId']),
                'skill': str(row['skill']).strip(),
                'correct': int(row['correct']),
            }
        )
    print(f"Total users: {len(user_interactions)}")
    return user_interactions


def generate_dataset_files(output_dir, user_interactions, ques_skill_map, train_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)

    unique_problems = ques_skill_map['problemId'].unique()
    problem_to_idx = {pid: idx for idx, pid in enumerate(sorted(unique_problems))}

    unique_skills = ques_skill_map['skill'].unique()
    skill_to_idx = {sk: idx for idx, sk in enumerate(sorted(unique_skills))}

    user_ids = list(user_interactions.keys())
    train_users, test_users = train_test_split(user_ids, test_size=1 - train_ratio, random_state=42)

    with open(os.path.join(output_dir, 'train_question.txt'), 'w', encoding='utf-8') as f_train_q, \
         open(os.path.join(output_dir, 'train_skill.txt'), 'w', encoding='utf-8') as f_train_s, \
         open(os.path.join(output_dir, 'test_question.txt'), 'w', encoding='utf-8') as f_test_q, \
         open(os.path.join(output_dir, 'test_skill.txt'), 'w', encoding='utf-8') as f_test_s:

        for uid in train_users:
            interactions = user_interactions[uid]
            if len(interactions) < 3:
                continue

            p = [str(problem_to_idx[it['problem_id']]) for it in interactions]
            s = [str(skill_to_idx[it['skill']]) for it in interactions]
            a = [str(it['correct']) for it in interactions]

            f_train_q.write('\n')
            f_train_q.write(','.join(p) + '\n')
            f_train_q.write(','.join(a) + '\n')

            f_train_s.write('\n')
            f_train_s.write(','.join(s) + '\n')
            f_train_s.write(','.join(a) + '\n')

        for uid in test_users:
            interactions = user_interactions[uid]
            if len(interactions) < 3:
                continue

            p = [str(problem_to_idx[it['problem_id']]) for it in interactions]
            s = [str(skill_to_idx[it['skill']]) for it in interactions]
            a = [str(it['correct']) for it in interactions]

            f_test_q.write('\n')
            f_test_q.write(','.join(p) + '\n')
            f_test_q.write(','.join(a) + '\n')

            f_test_s.write('\n')
            f_test_s.write(','.join(s) + '\n')
            f_test_s.write(','.join(a) + '\n')

    with open(os.path.join(output_dir, 'ques_skill.csv'), 'w', encoding='utf-8') as f:
        f.write('problem_id,skill_id\n')
        for orig_p in sorted(problem_to_idx.keys()):
            pid = problem_to_idx[orig_p]
            skills = ques_skill_map[ques_skill_map['problemId'] == orig_p]['skill'].unique()
            if len(skills) == 0:
                continue
            sid = skill_to_idx[str(skills[0])]
            f.write(f'{pid},{sid}\n')

    with open(os.path.join(output_dir, 'problem_map.txt'), 'w', encoding='utf-8') as f:
        f.write('problem_id,original_problem_id\n')
        for orig_id, new_id in sorted(problem_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{new_id},{orig_id}\n')

    with open(os.path.join(output_dir, 'skill_map.txt'), 'w', encoding='utf-8') as f:
        f.write('skill_id,original_skill\n')
        for orig_skill, new_id in sorted(skill_to_idx.items(), key=lambda x: x[1]):
            f.write(f'{new_id},{orig_skill}\n')

    print(f"Saved dataset files to: {output_dir}")
    return len(problem_to_idx), len(skill_to_idx)


def main():
    csv_path = os.path.join(SCRIPT_DIR, 'ASSIST2017', 'anonymized_full_release_competition_dataset.csv')
    output_dir = os.path.join(SCRIPT_DIR, 'data', 'ASSIST17')

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return

    df = load_and_process_data(csv_path)
    qsm = create_ques_skill_mapping(df)
    user_interactions = group_by_user(df)
    pro_max, skill_max = generate_dataset_files(output_dir, user_interactions, qsm)

    print('Done.')
    print(f'Output dir: {output_dir}')
    print(f'Problems: {pro_max}, Skills: {skill_max}')


if __name__ == '__main__':
    main()
