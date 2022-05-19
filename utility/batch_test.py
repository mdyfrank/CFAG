'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import sys

from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
from evaluator.python.evaluate_foldout import eval_score_matrix_foldout as eval_py
import multiprocessing
import heapq
import numpy as np
cores = multiprocessing.cpu_count() // 2

args = parse_args()

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, user_NUM = data_generator.n_groups, data_generator.n_users
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size

embedding_path = args.embedding_path
def save_embedding(sess, model, groups_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of users
    top_show = np.sort(model.Ks)

    u_batch_size = 12000

    test_groups = groups_to_test
    n_test_groups = len(test_groups)
    n_group_batchs = n_test_groups // u_batch_size + 1

    user_batch = range(user_NUM)
    for u_batch_id in range(n_group_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        group_batch = test_groups[start: end]
        if drop_flag == False:
            rate_batch = sess.run(model.saved_embedding, {model.groups: group_batch,
                                                        model.pos_users: user_batch})
            print(rate_batch[2])
            print(type(rate_batch[0]),type(rate_batch[2]))
            print(rate_batch[0].shape,rate_batch[1].shape,rate_batch[2].shape)
            np.save(embedding_path+'user_embedding',rate_batch[0])
            np.save(embedding_path+'group_embedding',rate_batch[1])
            np.save(embedding_path+'inter_group_embedding',rate_batch[2])


def test_case_study(sess, model, groups_to_test, drop_flag=False, train_set_flag=0, log_file = './CaseStudy/social_group.txt'):
    # B: batch size
    # N: the number of users
    top_show = np.sort(model.Ks)
    max_top = max(top_show)

    u_batch_size = BATCH_SIZE

    test_groups = groups_to_test
    n_test_groups = len(test_groups)
    n_group_batchs = n_test_groups // u_batch_size + 1

    user_batch = range(user_NUM)
    for u_batch_id in range(n_group_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        group_batch = test_groups[start: end]

        # print(len(user_batch),len(group_batch))
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.groups: group_batch,
                                                        model.pos_users: user_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.groups: group_batch,
                                                        model.pos_users: user_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)  # (B, N)
        test_users = []
        if train_set_flag == 0:
            for group in group_batch:
                test_users.append(data_generator.test_set[group])  # (B, #test_users)

            # set the ranking scores of training users to -inf,
            # then the training users will be sorted at the end of the ranking list.
            for idx, group in enumerate(group_batch):
                train_users_off = data_generator.train_users[group]
                rate_batch[idx][train_users_off] = -np.inf
        else:
            for group in group_batch:
                test_users.append(data_generator.train_users[group])
        # print(len(rate_batch),len(test_users))
        print(rate_batch.shape, len(test_users), max_top) # (B,k*metric_num), max_top= 20
        eval_py(rate_batch, test_users, max_top,log_file = log_file)
        # print(argmax_top_k(rate_batch,max_top))
        print('Case study done!')

def test(sess, model, groups_to_test, drop_flag=False, train_set_flag=0):
    # B: batch size
    # N: the number of users
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {'precision': np.zeros(len(model.Ks)), 'recall': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks))}

    u_batch_size = BATCH_SIZE


    test_groups = groups_to_test
    # print(test_groups)
    n_test_groups = len(test_groups)
    # print(n_test_groups)
    # sys.exit(0)
    n_group_batchs = n_test_groups // u_batch_size + 1
    
    count = 0
    all_result = []
    user_batch = range(user_NUM)
    for u_batch_id in range(n_group_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        group_batch = test_groups[start: end]

        # print(len(user_batch),len(group_batch))
        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.groups: group_batch,
                                                        model.pos_users: user_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.groups: group_batch,
                                                        model.pos_users: user_batch,
                                                        model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                        model.mess_dropout: [0.] * len(eval(args.layer_size))})
        rate_batch = np.array(rate_batch)# (B, N)
        test_users = []
        if train_set_flag == 0:
            for group in group_batch:
                test_users.append(data_generator.test_set[group])# (B, #test_users)
                
            # set the ranking scores of training users to -inf,
            # then the training users will be sorted at the end of the ranking list.    
            for idx, group in enumerate(group_batch):
                    train_users_off = data_generator.train_users[group]
                    rate_batch[idx][train_users_off] = -np.inf
        else:
            for group in group_batch:
                test_users.append(data_generator.train_users[group])
        batch_result = eval_score_matrix_foldout(rate_batch, test_users, max_top)#(B,k*metric_num), max_top= 20
        # print(argmax_top_k(rate_batch,max_top))
        count += len(batch_result)
        all_result.append(batch_result)
        
    
    assert count == n_test_groups
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result
               
            








