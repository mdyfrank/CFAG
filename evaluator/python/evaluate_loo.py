"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import heapq
def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def hit(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, user in enumerate(rank):
        if user == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def ndcg(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, user in enumerate(rank):
        if user == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/np.log2(last_idx+2)
    return result


def mrr(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, user in enumerate(rank):
        if user == ground_truth:
            last_idx = idx
            break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/(last_idx+1)
    return result


def eval_score_matrix_loo(score_matrix, test_users, top_k=50, thread_num=None):
    def _eval_one_group(idx):
        scores = score_matrix[idx]  # all scores of the test group
        test_user = test_users[idx]  # all test users of the test group

        ranking = argmax_top_k(scores, top_k)  # Top-K users
        result = []
        result.extend(hit(ranking, test_user))
        result.extend(ndcg(ranking, test_user))
        result.extend(mrr(ranking, test_user))

        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_group, range(len(test_users)))

    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray
