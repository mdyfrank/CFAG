"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
import sys
import heapq
from concurrent.futures import ThreadPoolExecutor


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def precision(rank, ground_truth):
    hits = [1 if user in ground_truth else 0 for user in rank]
    result = np.cumsum(hits, dtype=np.float) / np.arange(1, len(rank) + 1)
    return result


def recall(rank, ground_truth):
    hits = [1 if user in ground_truth else 0 for user in rank]
    result = np.cumsum(hits, dtype=np.float) / len(ground_truth)
    return result


def map_rank(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if user in ground_truth else 0 for idx, user in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    gt_len = len(ground_truth)
    # len_rank = np.array([min(i, gt_len) for i in range(1, len(rank)+1)])
    result = sum_pre / gt_len
    return result


def ndcg(rank, ground_truth):
    len_rank = len(rank)
    len_gt = len(ground_truth)
    idcg_len = min(len_gt, len_rank)

    # calculate idcg
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]

    # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    dcg = np.cumsum([1.0 / np.log2(idx + 2) if user in ground_truth else 0.0 for idx, user in enumerate(rank)])
    result = dcg / idcg
    return result


def mrr(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, user in enumerate(rank):
        if user in ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0 / (last_idx + 1)
    return result


def eval_score_matrix_foldout(score_matrix, test_users, top_k=50, log_file = None, thread_num=None):
    with open(log_file, 'w') as f:
        pass

    def _eval_one_group(idx):
        scores = score_matrix[idx]  # all scores of the test group
        test_user = test_users[idx]  # all test users of the test group

        ranking = argmax_top_k(scores, top_k)  # Top-K users
        with open(log_file, 'a') as f:
            line = str(idx) + ' ' + ','.join(map(str, ranking.tolist())) + \
                   ' ' + ','.join(map(str,ndcg(ranking, test_user)))  + '\n'
                   # ' ' + ','.join(map(str,recall(ranking, test_user))) + '\n'
            f.write(line)
        result = []
        result.extend(precision(ranking, test_user))
        result.extend(recall(ranking, test_user))
        result.extend(map_rank(ranking, test_user))
        result.extend(ndcg(ranking, test_user))
        result.extend(mrr(ranking, test_user))

        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_group, range(len(test_users)))

    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray
