'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import sys

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        group_item_file = path + '/groupItemTrain.txt'
        user_item_file = path + '/userItemTrain.txt'

        self.n_groups, self.n_users, self.n_items = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_groups = []
        self.groups_list, self.users_list = [], []
        with open(group_item_file) as f_gi:
            with open(user_item_file) as f_ui:
                for l in f_gi.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        self.n_items = max(self.n_items, max(items))
                for l in f_ui.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        self.n_items = max(self.n_items, max(items))

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    users = [int(i) for i in l[1:]]
                    gid = int(l[0])
                    self.exist_groups.append(gid)
                    self.n_users = max(self.n_users, max(users))
                    self.n_groups = max(self.n_groups, gid)
                    self.n_train += len(users)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        users = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_users = max(self.n_users, max(users))
                    self.n_test += len(users)

        self.n_users += 1
        self.n_groups += 1
        self.n_items += 1

        self.print_statistics()

        self.R_gi = sp.dok_matrix((self.n_groups, self.n_items), dtype=np.float32)
        self.train_group_item = {}
        with open(group_item_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                gid, train_items = items[0], items[1:]
                for i in train_items:
                    self.R_gi[gid, i] = 1.

                self.train_group_item[gid] = train_items

        self.R_ui = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_user_item = {}
        with open(user_item_file) as f_train:
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]

                for i in train_items:
                    self.R_ui[uid, i] = 1.

                self.train_user_item[uid] = train_items

        self.R = sp.dok_matrix((self.n_groups, self.n_users), dtype=np.float32)
        self.train_users, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    users = [int(i) for i in l.split(' ')]
                    gid, train_users = users[0], users[1:]

                    for u in train_users:
                        self.R[gid, u] = 1.

                    self.train_users[gid] = train_users

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        users = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    gid, test_users = users[0], users[1:]
                    self.test_set[gid] = test_users


        for gid, users in self.train_users.items():
            for user in users:
                self.users_list.append(user)
                self.groups_list.append(gid)
        # (users_np, items_np) = self.train_matrix.nonzero()

    def get_pre_adj_mat(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        # print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
        return pre_adj_mat

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

            adj_mat_gi = sp.load_npz(self.path + '/s_adj_mat_gi.npz')
            norm_adj_mat_gi = sp.load_npz(self.path + '/s_norm_adj_mat_gi.npz')
            mean_adj_mat_gi = sp.load_npz(self.path + '/s_mean_adj_mat_gi.npz')
            print('already load adj matrix group_item', adj_mat_gi.shape, time() - t1)

            adj_mat_ui = sp.load_npz(self.path + '/s_adj_mat_ui.npz')
            norm_adj_mat_ui = sp.load_npz(self.path + '/s_norm_adj_mat_ui.npz')
            mean_adj_mat_ui = sp.load_npz(self.path + '/s_mean_adj_mat_ui.npz')
            print('already load adj matrix user_item', adj_mat_ui.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.n_groups, self.n_users, self.R)
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

            adj_mat_gi, norm_adj_mat_gi, mean_adj_mat_gi = self.create_adj_mat(self.n_groups, self.n_items,
                                                                               self.R_gi)
            sp.save_npz(self.path + '/s_adj_mat_gi.npz', adj_mat_gi)
            sp.save_npz(self.path + '/s_norm_adj_mat_gi.npz', norm_adj_mat_gi)
            sp.save_npz(self.path + '/s_mean_adj_mat_gi.npz', mean_adj_mat_gi)

            adj_mat_ui, norm_adj_mat_ui, mean_adj_mat_ui = self.create_adj_mat(self.n_users, self.n_items,
                                                                               self.R_ui)
            sp.save_npz(self.path + '/s_adj_mat_ui.npz', adj_mat_ui)
            sp.save_npz(self.path + '/s_norm_adj_mat_ui.npz', norm_adj_mat_ui)
            sp.save_npz(self.path + '/s_mean_adj_mat_ui.npz', mean_adj_mat_ui)

        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            pre_adj_mat_gi = sp.load_npz(self.path + '/s_pre_adj_mat_gi.npz')
            pre_adj_mat_ui = sp.load_npz(self.path + '/s_pre_adj_mat_ui.npz')
        except Exception:
            adj_mat = adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            pre_adj_mat_gi = self.get_pre_adj_mat(adj_mat_gi)
            pre_adj_mat_ui = self.get_pre_adj_mat(adj_mat_ui)

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat, adj_mat_gi, norm_adj_mat_gi, mean_adj_mat_gi, pre_adj_mat_gi, adj_mat_ui, norm_adj_mat_ui, mean_adj_mat_ui, pre_adj_mat_ui

    def create_adj_mat(self, n_groups, n_users, R):
        t1 = time()
        adj_mat = sp.dok_matrix((n_groups + n_users, n_groups + n_users), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()
        # prevent memory from overflowing
        for i in range(5):
            adj_mat[int(n_groups * i / 5.0):int(n_groups * (i + 1.0) / 5), n_groups:] = \
                R[int(n_groups * i / 5.0):int(n_groups * (i + 1.0) / 5)]
            adj_mat[n_groups:, int(n_groups * i / 5.0):int(n_groups * (i + 1.0) / 5)] = \
                R[int(n_groups * i / 5.0):int(n_groups * (i + 1.0) / 5)].T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_users.keys():
            neg_users = list(set(range(self.n_users)) - set(self.train_users[u]))
            pools = [rd.choice(neg_users) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_groups:
            # print(self.n_groups,len(self.exist_groups), self.batch_size)
            groups = rd.sample(self.exist_groups, self.batch_size)
        else:
            groups = [rd.choice(self.exist_groups) for _ in range(self.batch_size)]

        def sample_pos_users_for_u(u, num):
            pos_users = self.train_users[u]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_i_id = pos_users[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_users_for_u(u, num):
            neg_users = []
            while True:
                if len(neg_users) == num: break
                neg_id = np.random.randint(low=0, high=self.n_users, size=1)[0]
                if neg_id not in self.train_users[u] and neg_id not in neg_users:
                    neg_users.append(neg_id)
            return neg_users

        def sample_neg_users_for_u_from_pools(u, num):
            neg_users = list(set(self.neg_pools[u]) - set(self.train_users[u]))
            return rd.sample(neg_users, num)

        pos_users, neg_users = [], []
        for u in groups:
            pos_users += sample_pos_users_for_u(u, 1)
            neg_users += sample_neg_users_for_u(u, 1)
        return groups, pos_users, neg_users

    def sample_test(self):
        if self.batch_size <= self.n_groups:
            groups = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            groups = [rd.choice(self.exist_groups) for _ in range(self.batch_size)]

        def sample_pos_users_for_u(u, num):
            pos_users = self.test_set[u]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_i_id = pos_users[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_users_for_u(u, num):
            neg_users = []
            while True:
                if len(neg_users) == num: break
                neg_id = np.random.randint(low=0, high=self.n_users, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_users[u]) and neg_id not in neg_users:
                    neg_users.append(neg_id)
            return neg_users

        def sample_neg_users_for_u_from_pools(u, num):
            neg_users = list(set(self.neg_pools[u]) - set(self.train_users[u]))
            return rd.sample(neg_users, num)

        pos_users, neg_users = [], []
        for u in groups:
            pos_users += sample_pos_users_for_u(u, 1)
            neg_users += sample_neg_users_for_u(u, 1)

        return groups, pos_users, neg_users

    def get_num_groups_users(self):
        return self.n_groups, self.n_users

    def print_statistics(self):
        print('n_groups=%d, n_users=%d, n_items=%d' % (self.n_groups, self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_groups * self.n_users)))

    def get_sparsity_split(self):
        try:
            split_gids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_gids.append([int(gid) for gid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_gids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(gid) for gid in split_gids[idx]]) + '\n')
            print('create sparsity split.')

        return split_gids, split_state

    def create_sparsity_split(self):
        all_groups_to_test = list(self.test_set.keys())
        group_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of gid).
        for gid in all_groups_to_test:
            train_iids = self.train_users[gid]
            test_iids = self.test_set[gid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in group_n_iid.keys():
                group_n_iid[n_iids] = [gid]
            else:
                group_n_iid[n_iids].append(gid)
        split_gids = list()

        # split the whole group set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(group_n_iid)):
            temp += group_n_iid[n_iids]
            n_rates += n_iids * len(group_n_iid[n_iids])
            n_count -= n_iids * len(group_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_gids.append(temp)

                state = '#inter per group<=[%d], #groups=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(group_n_iid.keys()) - 1 or n_count == 0:
                split_gids.append(temp)

                state = '#inter per group<=[%d], #groups=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_gids, split_state
