import argparse
import os
import numpy as np
from scipy import sparse


def extract_pro_pro_sim(pro_skill_coo):
    pro_num, skill_num = pro_skill_coo.shape
    pro_skill_csc = pro_skill_coo.tocsc()
    pro_skill_csr = pro_skill_coo.tocsr()

    pro_pro_adj = []
    for p in range(pro_num):
        tmp_skills = pro_skill_csr.getrow(p).indices
        if tmp_skills.size == 0:
            continue
        similar_pros = pro_skill_csc[:, tmp_skills].indices
        pro_pro_adj.extend(zip([p] * similar_pros.shape[0], similar_pros))

    if len(pro_pro_adj) == 0:
        return sparse.coo_matrix((pro_num, pro_num), dtype=np.float32)

    pro_pro_adj = np.array(list(set(pro_pro_adj)), dtype=np.int32)
    data = np.ones(pro_pro_adj.shape[0], dtype=np.float32)
    return sparse.coo_matrix((data, (pro_pro_adj[:, 0], pro_pro_adj[:, 1])), shape=(pro_num, pro_num))


def extract_skill_skill_sim(pro_skill_coo):
    pro_num, skill_num = pro_skill_coo.shape
    pro_skill_csc = pro_skill_coo.tocsc()
    pro_skill_csr = pro_skill_coo.tocsr()

    skill_skill_adj = []
    for s in range(skill_num):
        tmp_pros = pro_skill_csc.getcol(s).indices
        if tmp_pros.size == 0:
            continue
        similar_skills = pro_skill_csr[tmp_pros, :].indices
        skill_skill_adj.extend(zip([s] * similar_skills.shape[0], similar_skills))

    if len(skill_skill_adj) == 0:
        return sparse.coo_matrix((skill_num, skill_num), dtype=np.float32)

    skill_skill_adj = np.array(list(set(skill_skill_adj)), dtype=np.int32)
    data = np.ones(skill_skill_adj.shape[0], dtype=np.float32)
    return sparse.coo_matrix((data, (skill_skill_adj[:, 0], skill_skill_adj[:, 1])), shape=(skill_num, skill_num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()

    pro_skill_path = os.path.join(args.data_dir, 'pro_skill_sparse.npz')
    pro_skill_coo = sparse.load_npz(pro_skill_path)
    pro_num, skill_num = pro_skill_coo.shape
    print(f'problem number {pro_num}, skill number {skill_num}')

    pro_pro = extract_pro_pro_sim(pro_skill_coo)
    skill_skill = extract_skill_skill_sim(pro_skill_coo)

    sparse.save_npz(os.path.join(args.data_dir, 'pro_pro_sparse.npz'), pro_pro)
    sparse.save_npz(os.path.join(args.data_dir, 'skill_skill_sparse.npz'), skill_skill)
    print('Saved pro_pro_sparse.npz and skill_skill_sparse.npz')


if __name__ == '__main__':
    main()
