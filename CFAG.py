'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
version:
Parallelized sampling on CPU
C++ evaluation for top-k recommendation
'''

import os
import sys
import threading
import tensorflow as tf
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.attention import *
from utility.batch_test import *
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']


class LightGCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'LightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_groups = data_config['n_groups']
        self.n_users = data_config['n_users']

        self.aug_type = args.aug_type
        self.item_side = args.item_side
        self.ssl_mode = args.ssl_mode
        self.ssl_temp = args.ssl_temp
        self.n_items = data_config['n_items']
        self.ssl_reg = args.ssl_reg
        self.ssl_ratio = args.ssl_ratio
        self.n_fold = 1
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.att_ver = args.att_ver
        self.att_coef = args.att_coef
        self.gat_type = args.gat_type
        self.R = data_config['R']
        self.R_gi = data_config['R_gi']
        self.norm_adj_gi = data_config['norm_adj_gi']
        self.n_nonzero_elems_gi = self.norm_adj.count_nonzero()
        self.norm_adj_ui = data_config['norm_adj_ui']
        self.n_nonzero_elems_ui = self.norm_adj_ui.count_nonzero()
        if self.aug_type != -1:
            self.sgl_norm_adj = self.create_adj_mat(False)
        # print(self.norm_adj.toarray().shape)
        # print(self.norm_adj_gi.toarray().shape)
        # print(self.norm_adj_ui.toarray().shape)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.intra_emb_dim = args.intra_emb_dim
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.gi_side = args.gi_side
        self.att_concat = args.att_concat
        self.ebd_save = args.ebd_save
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.train = tf.placeholder(tf.bool)
        if self.train == tf.constant(True):
            self.groups = tf.placeholder(tf.int32, shape=(self.batch_size,))
            self.pos_users = tf.placeholder(tf.int32, shape=(self.batch_size,))
            self.neg_users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        else:
            self.groups = tf.placeholder(tf.int32, shape=(None,))
            self.pos_users = tf.placeholder(tf.int32, shape=(None,))
            self.neg_users = tf.placeholder(tf.int32, shape=(None,))

        att_initializer = tf.random_normal_initializer(stddev=0.01)
        if self.att_ver in [1]:
            self.gu_attention = tf.Variable(
                tf.constant_initializer(1.)(shape=[self.n_users, self.n_users], dtype=tf.float32))
            self.gi_attention = tf.Variable(
                tf.constant_initializer(1.)(shape=[self.n_items, self.n_items], dtype=tf.float32))
        elif self.att_ver == 2:
            (gu_indices_row, gu_indices_col) = self.R.nonzero()
            (gi_indices_row, gi_indices_col) = self.R_gi.nonzero()
            gu_atten_w = tf.Variable(tf.constant_initializer(1.)(shape=[gu_indices_row.shape[0], ], dtype=tf.float32))
            gi_atten_w = tf.Variable(tf.constant_initializer(1.)(shape=[gi_indices_row.shape[0], ], dtype=tf.float32))
            self.gu_attention_v2 = tf.sparse_softmax(
                tf.SparseTensor(indices=np.stack((gu_indices_row, gu_indices_col), axis=1),
                                values=gu_atten_w, dense_shape=self.R.shape))
            self.gi_attention_v2 = tf.sparse_softmax(
                tf.SparseTensor(indices=np.stack((gi_indices_row, gi_indices_col), axis=1),
                                values=gi_atten_w, dense_shape=self.R_gi.shape))
        elif self.att_ver == 3:
            self.gu_attention = tf.Variable(
                att_initializer(shape=[self.n_groups, self.n_users], dtype=tf.float32))
            self.gi_attention = tf.Variable(
                att_initializer(shape=[self.n_groups, self.n_items], dtype=tf.float32))
        elif self.att_ver == 4:
            self.gu_attention_vert = tf.Variable(
                att_initializer(shape=[self.n_users, self.intra_emb_dim], dtype=tf.float32))
            self.gu_attention_hori = tf.Variable(
                att_initializer(shape=[self.intra_emb_dim, self.n_users], dtype=tf.float32))
            self.gi_attention_vert = tf.Variable(
                att_initializer(shape=[self.n_items, self.intra_emb_dim], dtype=tf.float32))
            self.gi_attention_hori = tf.Variable(
                att_initializer(shape=[self.intra_emb_dim, self.n_items], dtype=tf.float32))
        elif self.att_ver in [5, 6]:
            self.gu_attention_vert = tf.Variable(
                att_initializer(shape=[self.n_users, self.intra_emb_dim], dtype=tf.float32), name='gu_attention')
            self.gi_attention_vert = tf.Variable(
                att_initializer(shape=[self.n_items, self.intra_emb_dim], dtype=tf.float32), name='gi_attention')

        self.items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
            self.train_ssl_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ssl_loss', self.train_ssl_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))

        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            # record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            # record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
            self.test_ssl_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ssl_loss', self.test_ssl_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all groups & users via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            if self.item_side == 0:
                self.ga_embeddings, self.ua_embeddings = self._create_lightgcn_embed()
            elif self.item_side == 1:
                self.ga_embeddings, self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed_general()
            print('embedding created')
            if self.aug_type != -1:
                self.ga_embeddings_sub1, self.ga_embeddings_sub2, self.ua_embeddings_sub1, self.ua_embeddings_sub2 = self._create_lightgcn_embed_sgl()
        elif self.alg_type in ['ngcf']:
            self.ga_embeddings, self.ua_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ga_embeddings, self.ua_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ga_embeddings, self.ua_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for group-user pairs in batch.
        """
        self.g_g_embeddings = tf.nn.embedding_lookup(self.ga_embeddings, self.groups)
        self.pos_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.pos_users)
        self.neg_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.neg_users)
        self.g_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['group_embedding'], self.groups)
        self.pos_u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.pos_users)
        self.neg_u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.neg_users)

        """
        *********************************************************
        Inference for the testing phase.
        """
        if self.att_ver == 5 and self.ebd_save == 1:
            # self.gu_atten_embeddings = tf.nn.embedding_lookup(self.gu_attention_vert, self.pos_users)
            # self.saved_embedding = [self.g_g_embeddings, self.pos_u_g_embeddings, self.gu_atten_embeddings]
            self.gi_atten_embeddings = self.gi_attention_vert
            self.saved_embedding = [self.g_g_embeddings, self.ia_embeddings, self.gi_atten_embeddings]
        self.batch_ratings = tf.matmul(self.g_g_embeddings, self.pos_u_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.g_g_embeddings,
                                                                          self.pos_u_g_embeddings,
                                                                          self.neg_u_g_embeddings)
        self.ssl_loss = self.calc_ssl_loss_v2(self.ga_embeddings_sub1, self.ga_embeddings_sub2, self.ua_embeddings_sub1,
                                              self.ua_embeddings_sub2) if self.aug_type != -1 else tf.constant(0.0,
                                                                                                               tf.float32)

        self.loss = self.mf_loss + self.emb_loss + self.ssl_loss
        # self.loss = self.mf_loss + self.emb_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_model_str(self):
        log_dir = '/' + self.alg_type + '/layers_' + str(self.n_layers) + '/dim_' + str(self.emb_dim)
        log_dir += '/' + args.dataset + '/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.n_groups + self.n_users
        self.training_group = data_generator.groups_list
        self.training_user = data_generator.users_list
        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            # data augmentation type --- 0: Node Dropout; 1: Edge Dropout; 2: Random Walk
            if aug_type == 0:
                drop_user_idx = randint_choice(self.n_groups, size=int(self.n_groups * self.ssl_ratio), replace=False)
                drop_item_idx = randint_choice(self.n_users, size=int(self.n_users * self.ssl_ratio), replace=False)
                indicator_user = np.ones(self.n_groups, dtype=np.float32)
                indicator_item = np.ones(self.n_users, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = data_generator.R
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.n_groups)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in [1, 2]:
                keep_idx = randint_choice(len(self.training_group),
                                          size=int(len(self.training_group) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(self.training_group)[keep_idx]
                item_np = np.array(self.training_user)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_groups)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(self.training_group)
            item_np = np.array(self.training_user)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_groups)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        # print('use the pre adjcency matrix')
        return adj_matrix

    def _init_weights(self):
        if self.aug_type != -1:
            self.sub_mat = {}
            if self.aug_type in [0, 1]:
                self.sub_mat['adj_values_sub1'] = tf.placeholder(tf.float32)
                self.sub_mat['adj_indices_sub1'] = tf.placeholder(tf.int64)
                self.sub_mat['adj_shape_sub1'] = tf.placeholder(tf.int64)

                self.sub_mat['adj_values_sub2'] = tf.placeholder(tf.float32)
                self.sub_mat['adj_indices_sub2'] = tf.placeholder(tf.int64)
                self.sub_mat['adj_shape_sub2'] = tf.placeholder(tf.int64)
            else:
                for k in range(1, self.n_layers + 1):
                    self.sub_mat['adj_values_sub1%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub1%d' % k)
                    self.sub_mat['adj_indices_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub1%d' % k)
                    self.sub_mat['adj_shape_sub1%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub1%d' % k)

                    self.sub_mat['adj_values_sub2%d' % k] = tf.placeholder(tf.float32, name='adj_values_sub2%d' % k)
                    self.sub_mat['adj_indices_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_indices_sub2%d' % k)
                    self.sub_mat['adj_shape_sub2%d' % k] = tf.placeholder(tf.int64, name='adj_shape_sub2%d' % k)

        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['group_embedding'] = tf.Variable(initializer([self.n_groups, self.emb_dim]),
                                                         name='group_embedding')
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')

            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            # all_weights['gu_atten_embedding'] = tf.Variable(initializer([self.n_users, self.intra_emb_dim]),
            #                                             name='gu_atten_embedding')
            print('using random initialization')  # print('using xavier initialization')
        else:
            all_weights['group_embedding'] = tf.Variable(initial_value=self.pretrain_data['group_embed'],
                                                         trainable=True,
                                                         name='group_embedding', dtype=tf.float32)
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)

            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)

            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_groups + self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_groups + self.n_users
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_general(self, n_groups, n_users, norm_adj):
        # print(norm_adj.toarray().shape)
        A_fold_hat = []
        fold_len = (n_groups + n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_groups + n_users
            else:
                end = (i_fold + 1) * fold_len
            # print(start, end)
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(norm_adj[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout_general(self, n_groups, n_users, norm_adj):
        A_fold_hat = []
        fold_len = (n_groups + n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_groups + n_users
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(norm_adj[start:end])
            n_nonzero_temp = norm_adj[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        fold_len = (self.n_groups + self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_groups + self.n_users
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed_sgl(self):
        for k in range(1, self.n_layers + 1):
            if self.aug_type in [0, 1]:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1'],
                    self.sub_mat['adj_values_sub1'],
                    self.sub_mat['adj_shape_sub1'])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2'],
                    self.sub_mat['adj_values_sub2'],
                    self.sub_mat['adj_shape_sub2'])
            else:
                self.sub_mat['sub_mat_1%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub1%d' % k],
                    self.sub_mat['adj_values_sub1%d' % k],
                    self.sub_mat['adj_shape_sub1%d' % k])
                self.sub_mat['sub_mat_2%d' % k] = tf.SparseTensor(
                    self.sub_mat['adj_indices_sub2%d' % k],
                    self.sub_mat['adj_values_sub2%d' % k],
                    self.sub_mat['adj_shape_sub2%d' % k])
        # adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj_sgl)
        ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        for k in range(1, self.n_layers + 1):
            # ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")
            # all_embeddings += [ego_embeddings]

            ego_embeddings_sub1 = tf.sparse_tensor_dense_matmul(
                self.sub_mat['sub_mat_1%d' % k],
                ego_embeddings_sub1, name="sparse_dense_sub1%d" % k)
            all_embeddings_sub1 += [ego_embeddings_sub1]

            ego_embeddings_sub2 = tf.sparse_tensor_dense_matmul(
                self.sub_mat['sub_mat_2%d' % k],
                ego_embeddings_sub2, name="sparse_dense_sub2%d" % k)
            all_embeddings_sub2 += [ego_embeddings_sub2]

        # all_embeddings = tf.stack(all_embeddings, 1)
        # all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        all_embeddings_sub1 = tf.stack(all_embeddings_sub1, 1)
        all_embeddings_sub1 = tf.reduce_mean(all_embeddings_sub1, axis=1, keepdims=False)
        g_g_embeddings_sub1, u_g_embeddings_sub1 = tf.split(all_embeddings_sub1, [self.n_groups, self.n_users], 0)

        all_embeddings_sub2 = tf.stack(all_embeddings_sub2, 1)
        all_embeddings_sub2 = tf.reduce_mean(all_embeddings_sub2, axis=1, keepdims=False)
        g_g_embeddings_sub2, u_g_embeddings_sub2 = tf.split(all_embeddings_sub2, [self.n_groups, self.n_users], 0)

        return g_g_embeddings_sub1, u_g_embeddings_sub1, g_g_embeddings_sub2, u_g_embeddings_sub2

    def calc_ssl_loss_v2(self, ga_embeddings_sub1, ga_embeddings_sub2, ua_embeddings_sub1, ua_embeddings_sub2):
        '''
        The denominator is summing over all the group or user nodes in the whole grpah
        '''
        if self.ssl_mode in ['group_side', 'both_side']:
            group_emb1 = tf.nn.embedding_lookup(ga_embeddings_sub1, self.groups)
            group_emb2 = tf.nn.embedding_lookup(ga_embeddings_sub2, self.groups)

            normalize_group_emb1 = tf.nn.l2_normalize(group_emb1, 1)
            normalize_group_emb2 = tf.nn.l2_normalize(group_emb2, 1)
            normalize_all_group_emb2 = tf.nn.l2_normalize(ga_embeddings_sub2, 1)
            pos_score_group = tf.reduce_sum(tf.multiply(normalize_group_emb1, normalize_group_emb2), axis=1)
            ttl_score_group = tf.matmul(normalize_group_emb1, normalize_all_group_emb2, transpose_a=False,
                                        transpose_b=True)

            pos_score_group = tf.exp(pos_score_group / self.ssl_temp)
            ttl_score_group = tf.reduce_sum(tf.exp(ttl_score_group / self.ssl_temp), axis=1)

            ssl_loss_group = -tf.reduce_sum(tf.log(pos_score_group / ttl_score_group))

        if self.ssl_mode in ['user_side', 'both_side']:
            user_emb1 = tf.nn.embedding_lookup(ua_embeddings_sub1, self.pos_users)
            user_emb2 = tf.nn.embedding_lookup(ua_embeddings_sub2, self.pos_users)

            normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
            normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
            normalize_all_user_emb2 = tf.nn.l2_normalize(ua_embeddings_sub2, 1)
            pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
            ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False,
                                       transpose_b=True)

            pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
            ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)

            ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        if self.ssl_mode == 'group_side':
            ssl_loss = self.ssl_reg * ssl_loss_group
        elif self.ssl_mode == 'user_side':
            ssl_loss = self.ssl_reg * ssl_loss_user
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_group + ssl_loss_user)

        return ssl_loss

    def graph_atten(self, user_embeddings, R, gu_attention):
        R = self._convert_sp_mat_to_sp_tensor(R)
        atten_mat = tf.nn.softmax(tf.sparse_tensor_dense_matmul(R, gu_attention), axis=1)
        # atten_mat = tf.nn.softmax(tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(R, gu_attention)), axis=1)
        # print(self.R.shape)
        print(atten_mat)
        g_atten_embeddings = tf.matmul(atten_mat, user_embeddings)
        return g_atten_embeddings

    def graph_atten_v4(self, user_embeddings, R, gu_attention_vert, gu_attention_hori):
        print('v4 used.')
        R = self._convert_sp_mat_to_sp_tensor(R)
        # gu_attention = tf.matmul(gu_attention_vert, gu_attention_hori)
        # gu_attention = tf.nn.sigmoid(tf.matmul(gu_attention_vert, gu_attention_hori))
        # gu_attention = tf.nn.softmax(tf.matmul(tf.nn.softmax(gu_attention_vert,axis = 1), tf.nn.softmax(gu_attention_hori,axis = 0)),axis=0)
        if args.att_norm == 0:
            gu_attention = tf.nn.softmax(tf.matmul(gu_attention_vert, gu_attention_hori), axis=0)
        else:
            gu_attention = tf.matmul(gu_attention_vert, gu_attention_hori)
        # gu_attention = tf.nn.tanh(tf.matmul(gu_attention_vert, gu_attention_hori))
        # atten_mat = tf.nn.softmax(tf.nn.relu(tf.sparse_tensor_dense_matmul(R, gu_attention)), axis=1)
        # atten_mat = tf.nn.softmax(tf.sparse_tensor_dense_matmul(R, gu_attention), axis=1)
        atten_mat = tf.nn.softmax(tf.nn.leaky_relu(tf.sparse_tensor_dense_matmul(R, gu_attention), alpha=0.2), axis=1)
        # print(self.R.shape)
        print(atten_mat)
        g_atten_embeddings = tf.matmul(atten_mat, user_embeddings)
        return g_atten_embeddings

    def graph_atten_v2(self, user_embeddings, gu_attention):
        '''
        :param user_embeddings: UxD
        :param R: GxU
        :param gu_attention:UxU
        :return: GxD
        '''
        return tf.sparse_tensor_dense_matmul(gu_attention, user_embeddings)

    def graph_atten_v3(self, user_embeddings, gu_attention):
        print('v3 used')
        g_atten_embeddings = tf.matmul(gu_attention, user_embeddings)
        return g_atten_embeddings

    def graph_conv(self, A_fold_hat, n_layers, ego_embeddings):
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                # print(A_fold_hat[f].shape, ego_embeddings.shape)
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        # print(all_embeddings.shape)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        # print(all_embeddings.shape)
        # sys.exit(0)
        return all_embeddings

    def _create_lightgcn_embed_general(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout_general(self.n_groups, self.n_users, self.norm_adj)
            A_fold_hat_gi = self._split_A_hat_node_dropout_general(self.n_groups, self.n_items, self.norm_adj_gi)
            if self.gi_side == 1:
                A_fold_hat_ui = self._split_A_hat_node_dropout_general(self.n_users, self.n_items, self.norm_adj_ui)
        else:
            A_fold_hat = self._split_A_hat_general(self.n_groups, self.n_users, self.norm_adj)
            A_fold_hat_gi = self._split_A_hat_general(self.n_groups, self.n_items, self.norm_adj_gi)
            if self.gi_side == 1:
                A_fold_hat_ui = self._split_A_hat_general(self.n_users, self.n_items, self.norm_adj_ui)

        if self.gat_type in ['user_side', 'both_side']:
            if self.att_ver == 1:  # one matrix
                atten_embedding = self.graph_atten(self.weights['user_embedding'], self.R, self.gu_attention)
            elif self.att_ver == 2:  # GAT_mode
                atten_embedding = self.graph_atten_v2(self.weights['user_embedding'], self.gu_attention_v2)
            elif self.att_ver == 3:
                atten_embedding = self.graph_atten_v3(self.weights['user_embedding'], self.gu_attention)
            elif self.att_ver == 4:
                atten_embedding = self.graph_atten_v4(self.weights['user_embedding'], self.R, self.gu_attention_vert,
                                                      self.gu_attention_hori)
            elif self.att_ver == 5:  # two embedding. M2 is transpose of M1
                atten_embedding = self.graph_atten_v4(self.weights['user_embedding'], self.R, self.gu_attention_vert,
                                                      tf.transpose(self.gu_attention_vert))
            elif self.att_ver == 6:
                # print(self.gu_attention_vert.shape)
                # print(self.weights['user_embedding'].shape)
                # sys.exit(0)
                atten_embedding = self.graph_atten_v4(self.weights['user_embedding'], self.R,
                                                      self.weights['user_embedding'],
                                                      tf.transpose(self.weights['user_embedding']))

            if self.att_concat:
                group_embedding = tf.concat([self.weights['group_embedding'], atten_embedding], axis=1)
                all_embeddings = self.graph_conv(A_fold_hat, self.n_layers,
                                                 tf.concat([group_embedding, tf.concat(
                                                     [self.weights['user_embedding'], self.weights['user_embedding']],
                                                     axis=1)], axis=0))
            else:
                if args.att_norm == 0:
                    group_embedding = self.weights['group_embedding'] + self.att_coef * atten_embedding
                else:
                    group_embedding = tf.nn.l2_normalize(self.weights['group_embedding'] + self.att_coef * atten_embedding,axis=1)

                all_embeddings = self.graph_conv(A_fold_hat, self.n_layers,
                                                 tf.concat([group_embedding, self.weights['user_embedding']], axis=0))
        else:
            all_embeddings = self.graph_conv(A_fold_hat, self.n_layers,
                                             tf.concat(
                                                 [self.weights['group_embedding'], self.weights['user_embedding']],
                                                 axis=0))
        g_u_embeddings, u_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_users], 0)
        # print(g_u_embeddings.shape, u_g_embeddings.shape)
        if self.gat_type in ['item_side', 'both_side']:
            if self.att_ver == 1:
                atten_embedding_gi = self.graph_atten(self.weights['item_embedding'], self.R_gi, self.gi_attention)
            elif self.att_ver == 2:
                atten_embedding_gi = self.graph_atten_v2(self.weights['item_embedding'], self.gi_attention_v2)
            elif self.att_ver == 3:
                atten_embedding_gi = self.graph_atten_v3(self.weights['item_embedding'], self.gi_attention)
            elif self.att_ver == 4:
                atten_embedding_gi = self.graph_atten_v4(self.weights['item_embedding'], self.R_gi,
                                                         self.gi_attention_vert, self.gi_attention_hori)
            elif self.att_ver == 5:
                atten_embedding_gi = self.graph_atten_v4(self.weights['item_embedding'], self.R_gi,
                                                         self.gi_attention_vert, tf.transpose(self.gi_attention_vert))
            elif self.att_ver == 6:
                atten_embedding_gi = self.graph_atten_v4(self.weights['item_embedding'], self.R_gi,
                                                         self.weights['item_embedding'],
                                                         tf.transpose(self.weights['item_embedding']))

            if self.att_concat:
                group_embedding_gi = tf.concat([self.weights['group_embedding'], atten_embedding_gi], axis=1)
                all_embeddings_gi = self.graph_conv(A_fold_hat_gi, self.n_layers,
                                                    tf.concat([group_embedding_gi, tf.concat(
                                                        [self.weights['item_embedding'],
                                                         self.weights['item_embedding']], axis=1)],
                                                              axis=0))
            else:
                if args.att_norm == 0:
                    group_embedding_gi = self.weights['group_embedding'] + self.att_coef * atten_embedding_gi
                else:
                    group_embedding_gi = tf.nn.l2_normalize(self.weights['group_embedding'] + self.att_coef * atten_embedding_gi,axis=1)
                all_embeddings_gi = self.graph_conv(A_fold_hat_gi, self.n_layers,
                                                    tf.concat([group_embedding_gi, self.weights['item_embedding']],
                                                              axis=0))
        else:
            all_embeddings_gi = self.graph_conv(A_fold_hat_gi, self.n_layers,
                                                tf.concat(
                                                    [self.weights['group_embedding'], self.weights['item_embedding']],
                                                    axis=0))
        g_i_embeddings, i_g_embeddings = tf.split(all_embeddings_gi, [self.n_groups, self.n_items], 0)
        # print(g_i_embeddings.shape, i_g_embeddings.shape)
        if self.gi_side == 1:
            if self.att_concat:
                all_embeddings_ui = self.graph_conv(A_fold_hat_ui, self.n_layers,
                                                    tf.concat([tf.concat([self.weights['user_embedding'],
                                                                          self.weights['user_embedding']], axis=1),
                                                               tf.concat([self.weights['item_embedding'],
                                                                          self.weights['item_embedding']], axis=1)],
                                                              axis=0))
            else:
                all_embeddings_ui = self.graph_conv(A_fold_hat_ui, self.n_layers,
                                                    tf.concat([self.weights['user_embedding'],
                                                               self.weights['item_embedding']],
                                                              axis=0))
            u_i_embeddings, i_u_embeddings = tf.split(all_embeddings_ui, [self.n_users, self.n_items], 0)
            u_embeddings = tf.concat([u_g_embeddings, u_i_embeddings], 1)
            g_embeddings = tf.concat([g_u_embeddings, g_i_embeddings], 1)
            i_embeddings = tf.concat([i_g_embeddings, i_u_embeddings], 1)
        elif self.gi_side == 0:
            u_embeddings = tf.concat([u_g_embeddings, tf.ones_like(u_g_embeddings, dtype=tf.float32)], 1)
            g_embeddings = tf.concat([g_u_embeddings, g_i_embeddings], 1)
            i_embeddings = tf.concat([i_g_embeddings, tf.ones_like(i_g_embeddings, dtype=tf.float32)], 1)
        # print(g_embeddings.shape, u_embeddings.shape)
        # sys.exit(0)
        return g_embeddings, u_embeddings, i_embeddings

    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        g_g_embeddings, u_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_users], 0)
        return g_g_embeddings, u_g_embeddings

    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        g_g_embeddings, u_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_users], 0)
        return g_g_embeddings, u_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        g_g_embeddings, u_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_users], 0)
        return g_g_embeddings, u_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['group_embedding'], self.weights['user_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k]
            # mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        g_g_embeddings, u_g_embeddings = tf.split(all_embeddings, [self.n_groups, self.n_users], 0)
        return g_g_embeddings, u_g_embeddings

    def create_bpr_loss(self, groups, pos_users, neg_users):
        # print(groups.shape,pos_users.shape, neg_users.shape ,items.shape)
        # groups = tf.nn.l2_normalize(groups, 1)
        # pos_users = tf.nn.l2_normalize(pos_users, 1)
        # neg_users = tf.nn.l2_normalize(neg_users, 1)

        pos_scores = tf.reduce_sum(tf.multiply(groups, pos_users), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(groups, neg_users), axis=1)

        # pos_item_scores = tf.reduce_sum(tf.multiply(tf.multiply(groups, pos_users),items), axis=1)

        regularizer = tf.nn.l2_loss(self.g_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_u_g_embeddings_pre) + tf.nn.l2_loss(self.neg_u_g_embeddings_pre)
        # if self.aug_type == 5:
        #     regularizer = tf.nn.l2_loss(self.g_g_embeddings_pre) + tf.nn.l2_loss(
        #         self.pos_u_g_embeddings_pre) + tf.nn.l2_loss(self.neg_u_g_embeddings_pre) \
        #                   + tf.nn.l2_loss(self.gu_atten_embeddings) + tf.nn.l2_loss(self.gi_atten_embeddings)
        regularizer = regularizer / self.batch_size
        if self.aug_type != -1:
            mf_loss = tf.reduce_mean(tf.nn.sigmoid(-(pos_scores - neg_scores)))
        else:
            mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        # print(coo.toarray().shape)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


# parallelized sampling on CPU
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()


class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample_test()


# training on GPU
class train_thread(threading.Thread):
    def __init__(self, model, sess, sample, sub_mat=None):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
        self.sub_mat = sub_mat

    def run(self):
        groups, pos_users, neg_users = self.sample.data
        feed_dict = {model.groups: groups, model.pos_users: pos_users,
                     model.node_dropout: eval(args.node_dropout),
                     model.mess_dropout: eval(args.mess_dropout),
                     model.neg_users: neg_users,
                     model.train: True}
        if self.model.aug_type != -1:
            if self.model.aug_type in [0, 1]:
                feed_dict.update({
                    self.model.sub_mat['adj_values_sub1']: self.sub_mat['adj_values_sub1'],
                    self.model.sub_mat['adj_indices_sub1']: self.sub_mat['adj_indices_sub1'],
                    self.model.sub_mat['adj_shape_sub1']: self.sub_mat['adj_shape_sub1'],
                    self.model.sub_mat['adj_values_sub2']: self.sub_mat['adj_values_sub2'],
                    self.model.sub_mat['adj_indices_sub2']: self.sub_mat['adj_indices_sub2'],
                    self.model.sub_mat['adj_shape_sub2']: self.sub_mat['adj_shape_sub2']
                })
            else:
                for k in range(1, self.model.n_layers + 1):
                    feed_dict.update({
                        self.model.sub_mat['adj_values_sub1%d' % k]: self.sub_mat['adj_values_sub1%d' % k],
                        self.model.sub_mat['adj_indices_sub1%d' % k]: self.sub_mat['adj_indices_sub1%d' % k],
                        self.model.sub_mat['adj_shape_sub1%d' % k]: self.sub_mat['adj_shape_sub1%d' % k],
                        self.model.sub_mat['adj_values_sub2%d' % k]: self.sub_mat['adj_values_sub2%d' % k],
                        self.model.sub_mat['adj_indices_sub2%d' % k]: self.sub_mat['adj_indices_sub2%d' % k],
                        self.model.sub_mat['adj_shape_sub2%d' % k]: self.sub_mat['adj_shape_sub2%d' % k]
                    })
        self.data = sess.run(
            [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss,
             self.model.ssl_loss],
            feed_dict=feed_dict)


class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample, sub_mat=None):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
        self.sub_mat = sub_mat

    def run(self):
        groups, pos_users, neg_users = self.sample.data
        feed_dict = {model.groups: groups, model.pos_users: pos_users,
                     model.node_dropout: eval(args.node_dropout),
                     model.mess_dropout: eval(args.mess_dropout),
                     model.neg_users: neg_users,
                     model.train: False}
        if self.model.aug_type != -1:
            if self.model.aug_type in [0, 1]:
                feed_dict.update({
                    self.model.sub_mat['adj_values_sub1']: self.sub_mat['adj_values_sub1'],
                    self.model.sub_mat['adj_indices_sub1']: self.sub_mat['adj_indices_sub1'],
                    self.model.sub_mat['adj_shape_sub1']: self.sub_mat['adj_shape_sub1'],
                    self.model.sub_mat['adj_values_sub2']: self.sub_mat['adj_values_sub2'],
                    self.model.sub_mat['adj_indices_sub2']: self.sub_mat['adj_indices_sub2'],
                    self.model.sub_mat['adj_shape_sub2']: self.sub_mat['adj_shape_sub2']
                })
            else:
                for k in range(1, self.model.n_layers + 1):
                    feed_dict.update({
                        self.model.sub_mat['adj_values_sub1%d' % k]: self.sub_mat['adj_values_sub1%d' % k],
                        self.model.sub_mat['adj_indices_sub1%d' % k]: self.sub_mat['adj_indices_sub1%d' % k],
                        self.model.sub_mat['adj_shape_sub1%d' % k]: self.sub_mat['adj_shape_sub1%d' % k],
                        self.model.sub_mat['adj_values_sub2%d' % k]: self.sub_mat['adj_values_sub2%d' % k],
                        self.model.sub_mat['adj_indices_sub2%d' % k]: self.sub_mat['adj_indices_sub2%d' % k],
                        self.model.sub_mat['adj_shape_sub2%d' % k]: self.sub_mat['adj_shape_sub2%d' % k]
                    })
        self.data = sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.ssl_loss],
                             feed_dict=feed_dict)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()

    config = dict()
    config['n_groups'] = data_generator.n_groups
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['R'] = data_generator.R
    config['R_gi'] = data_generator.R_gi
    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj, plain_adj_gi, norm_adj_gi, mean_adj_gi, pre_adj_gi, plain_adj_ui, norm_adj_ui, mean_adj_ui, pre_adj_ui = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        config['norm_adj_gi'] = plain_adj_gi
        config['norm_adj_ui'] = plain_adj_ui
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        config['norm_adj_gi'] = norm_adj_gi
        config['norm_adj_ui'] = norm_adj_ui
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        config['norm_adj_gi'] = mean_adj_gi
        config['norm_adj_ui'] = mean_adj_ui
        print('use the gcmc adjacency matrix')
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        config['norm_adj_gi'] = pre_adj_gi
        config['norm_adj_ui'] = pre_adj_ui
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = LightGCN(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                groups_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, groups_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]
                cur_best_pre_1 = ret['ndcg'][0]

                pretrain_ret = 'pretrained model recall=[%s], precision=[%s], ' \
                               'ndcg=[%s]' % \
                               (', '.join(['%.5f' % r for r in ret['recall']]),
                                ', '.join(['%.5f' % r for r in ret['precision']]),
                                ', '.join(['%.5f' % r for r in ret['ndcg']]))
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            cur_best_pre_1 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        cur_best_pre_1 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        groups_to_test_list, split_state = data_generator.get_sparsity_split()
        groups_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, groups_to_test in enumerate(groups_to_test_list):
            ret = test(sess, model, groups_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                         (', '.join(['%.5f' % r for r in ret['recall']]),
                          ', '.join(['%.5f' % r for r in ret['precision']]),
                          ', '.join(['%.5f' % r for r in ret['ndcg']]))

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir + '/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path + model.log_dir + '/run_' + str(run_time), sess.graph)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(1, args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss, ssl_loss = 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        loss_test, mf_loss_test, emb_loss_test, reg_loss_test, ssl_loss_test = 0., 0., 0., 0., 0.
        if model.aug_type != -1:
            sub_mat = {}
            if model.aug_type in [0, 1]:
                sub_mat['adj_indices_sub1'], sub_mat['adj_values_sub1'], sub_mat[
                    'adj_shape_sub1'] = model._convert_csr_to_sparse_tensor_inputs(
                    model.create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
                sub_mat['adj_indices_sub2'], sub_mat['adj_values_sub2'], sub_mat[
                    'adj_shape_sub2'] = model._convert_csr_to_sparse_tensor_inputs(
                    model.create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
            else:
                for k in range(1, model.n_layers + 1):
                    sub_mat['adj_indices_sub1%d' % k], sub_mat['adj_values_sub1%d' % k], sub_mat[
                        'adj_shape_sub1%d' % k] = model._convert_csr_to_sparse_tensor_inputs(
                        model.create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
                    sub_mat['adj_indices_sub2%d' % k], sub_mat['adj_values_sub2%d' % k], sub_mat[
                        'adj_shape_sub2%d' % k] = model._convert_csr_to_sparse_tensor_inputs(
                        model.create_adj_mat(is_subgraph=True, aug_type=model.aug_type))
        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last = sample_thread()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread(model, sess, sample_last, sub_mat) if model.aug_type != -1 else train_thread(model,
                                                                                                                  sess,
                                                                                                                  sample_last)
            sample_next = sample_thread()

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            groups, pos_users, neg_users = sample_last.data
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss, batch_ssl_loss = train_cur.data
            sample_last = sample_next

            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
            ssl_loss += batch_ssl_loss / n_batch
        summary_train_loss = sess.run(model.merged_train_loss,
                                      feed_dict={model.train_loss: loss, model.train_mf_loss: mf_loss,
                                                 model.train_emb_loss: emb_loss, model.train_reg_loss: reg_loss,
                                                 model.train_ssl_loss: ssl_loss})
        train_writer.add_summary(summary_train_loss, epoch)
        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch % 20) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f+ %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, ssl_loss)
                print(perf_str)
            continue
        groups_to_test = list(data_generator.train_users.keys())
        # save_embedding(sess, model, groups_to_test, drop_flag=False, train_set_flag=1)
        ret = test(sess, model, groups_to_test, drop_flag=True, train_set_flag=1)
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss, reg_loss,
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str)
        summary_train_acc = sess.run(model.merged_train_acc, feed_dict={model.train_rec_first: ret['recall'][0],
                                                                        model.train_rec_last: ret['recall'][-1],
                                                                        model.train_ndcg_first: ret['ndcg'][0],
                                                                        model.train_ndcg_last: ret['ndcg'][-1]})
        train_writer.add_summary(summary_train_acc, epoch // 20)

        '''
        *********************************************************
        parallelized sampling
        '''
        sample_last = sample_thread_test()
        sample_last.start()
        sample_last.join()
        for idx in range(n_batch):
            train_cur = train_thread_test(model, sess, sample_last,
                                          sub_mat) if model.aug_type != -1 else train_thread_test(model, sess,
                                                                                                  sample_last)
            sample_next = sample_thread_test()

            train_cur.start()
            sample_next.start()

            sample_next.join()
            train_cur.join()

            groups, pos_users, neg_users = sample_last.data
            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test, batch_ssl_loss_test = train_cur.data
            sample_last = sample_next

            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch
            ssl_loss_test += batch_ssl_loss_test / n_batch

        summary_test_loss = sess.run(model.merged_test_loss,
                                     feed_dict={model.test_loss: loss_test, model.test_mf_loss: mf_loss_test,
                                                model.test_emb_loss: emb_loss_test, model.test_reg_loss: reg_loss_test,
                                                model.test_ssl_loss: ssl_loss_test})
        train_writer.add_summary(summary_test_loss, epoch // 20)
        t2 = time()
        groups_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, groups_to_test, drop_flag=True)
        summary_test_acc = sess.run(model.merged_test_acc,
                                    feed_dict={model.test_rec_first: ret['recall'][0],
                                               model.test_rec_last: ret['recall'][-1],
                                               model.test_ndcg_first: ret['ndcg'][0],
                                               model.test_ndcg_last: ret['ndcg'][-1]})
        train_writer.add_summary(summary_test_acc, epoch // 20)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f+ %.5f], recall=[%s], ' \
                       'precision=[%s], ndcg=[%s]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test, ssl_loss_test,
                        ', '.join(['%.5f' % r for r in ret['recall']]),
                        ', '.join(['%.5f' % r for r in ret['precision']]),
                        ', '.join(['%.5f' % r for r in ret['ndcg']]))
            print(perf_str)

        if args.es_type == 0:
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break

            # *********************************************************
            # save the group & user embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)
        elif args.es_type == 1:
            cur_best_pre_1, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_1,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=5)

            # *********************************************************
            # early stopping when cur_best_pre_1 is decreasing for ten successive steps.
            if should_stop == True:
                break

            # *********************************************************
            # save the group & user embeddings for pretraining.
            if ret['ndcg'][0] == cur_best_pre_1 and args.save_flag == 1:
                save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)
    # print(groups_to_test)
    if args.case_study == 1:
        if args.item_side == 1:
            test_case_study(sess, model, groups_to_test, drop_flag=False, log_file = './CaseStudy/social_group.txt')
        else:
            test_case_study(sess, model, groups_to_test, drop_flag=False, log_file = './CaseStudy/lightgcn.txt')
    groups_to_test = list(data_generator.train_users.keys())
    if args.ebd_save == 1 and args.att_ver == 5:
        save_embedding(sess, model, groups_to_test, drop_flag=False, train_set_flag=1)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'top_k=%s, embed_size=%d, lr=%.4f, layer_size=%s, batch_size=%d, att_ver=%d, item_side=%d aug_type=%d, gat_type=%s, regs=%s, intra_emb_dim=%d,att_coef=%.3f\n\t%s\n'
        % (args.Ks, args.embed_size, args.lr, args.layer_size, args.batch_size, args.att_ver, args.item_side,
           args.aug_type,
           args.gat_type, args.regs,
           args.intra_emb_dim,args.att_coef, final_perf))
    f.close()
