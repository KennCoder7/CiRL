#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import random
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from tqdm import tqdm
from .processor import Processor
import torch.backends.cudnn as cudnn
import random
# Turn off benchmark mode when not needed
cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        self.mask_loss = lambda x, y: (-(F.log_softmax(x, dim=1) * y).sum(1) / y.sum(1)).mean()
        seed = self.arg.seed
        # Set the random seed manually for reproducibility.
        torch.manual_seed(seed)
        # Set the random seed manually for reproducibility.
        np.random.seed(seed)
        # Set the random seed manually for reproducibility.
        random.seed(seed)

        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        warmup_epoch = self.arg.warmup_epoch
        if self.arg.lr_decay_type == 'step':
            if self.meta_info['epoch'] < warmup_epoch:
                self.warmup(warmup_epoch)
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
        elif self.arg.lr_decay_type == 'cosine':
            if self.meta_info['epoch'] < warmup_epoch:
                self.warmup(warmup_epoch)
            else:
                lr = self.cosine_annealing(self.arg.base_lr, eta_min=0.0001 * self.arg.base_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.lr = lr
        elif self.arg.lr_decay_type == 'constant':
            self.lr = self.arg.base_lr
        else:
            raise ValueError()

    def cosine_annealing(self, x, eta_min=0.):
        """Cosine annealing scheduler
        """
        return eta_min + (x - eta_min) * (1. + np.cos(np.pi * self.meta_info['epoch'] / self.arg.num_epoch)) / 2

    def warmup(self, warmup_epoch=10):
        lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / warmup_epoch
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def show_topk(self, k):
        # self.result is a dict
        for key in self.result.keys():
            rank = self.result[key].argsort()
            hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
            accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
            self.io.print_log(f'{key}' + '\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_cls_value = []
        loss_supcl_value = []
        loss_cl1_value = []
        loss_cl2_value = []

        for [data1, data2], label in tqdm(loader):
            if type(label) == list:
                label = label[0]
            # get data
            data1 = data1.float().to(self.dev)
            data2 = data2.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            if self.meta_info['epoch'] < self.arg.npnm_epoch:
                cl_logits, cl_labels, logits, logits_cl1, logits_cl2 = self.model(data1, data2, label)
                loss_cls = self.loss(logits, label)
                loss_supcl = self.loss(cl_logits, cl_labels)
                loss_cl1 = self.loss(logits_cl1, label)
                loss_cl2 = self.loss(logits_cl2, label)
                loss = self.arg.alpha * loss_cls + self.arg.beta * loss_supcl + self.arg.gamma * loss_cl1 + self.arg.delta * loss_cl2
            else:
                cl_logits, cl_labels, logits, logits_cl1, logits_cl2 = self.model(data1, data2, label, NPNM=True)
                loss_cls = self.loss(logits, label)
                loss_supcl = self.mask_loss(cl_logits, cl_labels)
                loss_cl1 = self.loss(logits_cl1, label)
                loss_cl2 = self.loss(logits_cl2, label)
                loss = self.arg.alpha * loss_cls + self.arg.beta * loss_supcl + self.arg.gamma * loss_cl1 + self.arg.delta * loss_cl2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_cls'] = self.arg.alpha * loss_cls.data.item()
            self.iter_info['loss_supcl'] = self.arg.beta * loss_supcl.data.item()
            self.iter_info['loss_cl1'] = self.arg.gamma * loss_cl1.data.item()
            self.iter_info['loss_cl2'] = self.arg.delta * loss_cl2.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_cls_value.append(self.iter_info['loss_cls'])
            loss_supcl_value.append(self.iter_info['loss_supcl'])
            loss_cl1_value.append(self.iter_info['loss_cl1'])
            loss_cl2_value.append(self.iter_info['loss_cl2'])
            # self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss_cls'] = np.mean(loss_cls_value)
        self.epoch_info['mean_loss_supcl'] = np.mean(loss_supcl_value)
        self.epoch_info['mean_loss_cl1'] = np.mean(loss_cl1_value)
        self.epoch_info['mean_loss_cl2'] = np.mean(loss_cl2_value)
        self.epoch_info['lr'] = self.lr
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        loss1_value = []
        loss2_value = []
        result_frag = []
        result1_frag = []
        result2_frag = []
        label_frag = []
        max_acc = 0.

        for data, label in tqdm(loader):
            if type(data) is list:
                data = data[0]
            if type(label) == list:
                label = label[0]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output, output1, output2 = self.model(data)
            result_frag.append(output.data.cpu().numpy())
            result1_frag.append(output1.data.cpu().numpy())
            result2_frag.append(output2.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss1 = self.loss(output1, label)
                loss2 = self.loss(output2, label)
                loss_value.append(loss.item())
                loss1_value.append(loss1.item())
                loss2_value.append(loss2.item())
                label_frag.append(label.data.cpu().numpy())

        self.result['logits'] = np.concatenate(result_frag)
        self.result['logits_cl1'] = np.concatenate(result1_frag)
        self.result['logits_cl2'] = np.concatenate(result2_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss_cls'] = np.mean(loss_value)
            self.epoch_info['mean_loss_supcl'] = -1.
            self.epoch_info['mean_loss_cl1'] = np.mean(loss1_value)
            self.epoch_info['mean_loss_cl2'] = np.mean(loss2_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
        save_path = os.path.join(self.arg.work_dir, 'result.npy')
        np.save(save_path, self.result['logits'])
        save_path = os.path.join(self.arg.work_dir, 'result_cl1.npy')
        np.save(save_path, self.result['logits_cl1'])
        save_path = os.path.join(self.arg.work_dir, 'result_cl2.npy')
        np.save(save_path, self.result['logits_cl2'])

    def visualize(self):

        self.model.eval()
        loader = self.data_loader['test']

        max_n = 50000
        sample_level_representation = np.zeros((max_n, 256))
        sample_level_label = np.zeros((max_n))
        object_level_representation = np.zeros((max_n, 5, 256))
        object_level_label = np.zeros((max_n, 5))
        queries = np.zeros((60, 256))
        ptr = 0

        for data, label in loader:
            if type(data) is list:
                data = data[0]
            assert type(label) == list
            label, label_lst = label[0], label[1]
            # get data
            data = data.float().to(self.dev)
            bs = data.size(0)

            # inference
            with torch.no_grad():
                local_features, global_features, learned_queries = self.model.visualization(data)
            sample_level_representation[ptr:ptr+bs] = global_features.data.cpu().numpy()
            sample_level_label[ptr:ptr+bs] = label
            object_level_representation[ptr:ptr+bs] = local_features.data.cpu().numpy()
            object_level_label[ptr:ptr+bs] = np.array(label_lst)
            ptr += bs
            # sample_level_label.append(label)
            # sample_level_representation.append(global_features.data.cpu().numpy())
            # object_level_label.append(label_lst)
            # object_level_representation.append(local_features.data.cpu().numpy())
            queries = learned_queries.data.cpu().numpy()

            self.show_iter_info()
            self.meta_info['iter'] += 1

        # create the folder to save the features
        path = os.path.join(self.arg.work_dir, self.arg.features_path)
        if not os.path.exists(path):
            os.makedirs(path)

        # save the sample-level representation
        sample_level_representation = sample_level_representation[:ptr]
        sample_level_label = sample_level_label[:ptr]
        print(sample_level_representation.shape)
        print(sample_level_label.shape)
        np.save(os.path.join(path, 'sample_level_representation.npy'), sample_level_representation)
        np.save(os.path.join(path, 'sample_level_label.npy'), sample_level_label)

        # save the object-level representation
        object_level_representation = object_level_representation[:ptr]
        object_level_label = object_level_label[:ptr]
        print(object_level_representation.shape)
        print(object_level_label.shape)
        np.save(os.path.join(path, 'object_level_representation.npy'), object_level_representation)
        np.save(os.path.join(path, 'object_level_label.npy'), object_level_label)

        # save the learned queries
        print(queries.shape)
        np.save(os.path.join(path, 'learned_queries.npy'), queries)

    def save_att_map(self):

        self.model.eval()
        loader = self.data_loader['test']

        att_map = []
        label_list = []

        for data, label in tqdm(loader):
            if type(data) is list:
                data = data[0]
            if type(label) == list:
                label = label[0]
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                _, att = self.model(data, return_att=True)
            att_map.append(att.data.cpu().numpy())
            label_list.append(label.data.cpu().numpy())

            self.show_iter_info()
            self.meta_info['iter'] += 1

        # create the folder to save the features
        path = os.path.join(self.arg.work_dir, 'att_map')
        if not os.path.exists(path):
            os.makedirs(path)

        # save the sample-level representation
        np.save(os.path.join(path, 'att_map.npy'), att_map)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay_type', type=str, default='step', help='lr_decay_type')
        parser.add_argument('--alpha', type=float, default=1.0, help='weight for classification loss')
        parser.add_argument('--beta', type=float, default=1.0, help='weight for contrastive loss')
        parser.add_argument('--gamma', type=float, default=1.0, help='weight for contrastive1 loss')
        parser.add_argument('--delta', type=float, default=1.0, help='weight for contrastive2 loss')
        parser.add_argument('--npnm_epoch', type=int, default=10, help='npnm_epoch')
        parser.add_argument('--features_path', type=str, default='features', help='features_path')
        parser.add_argument('--seed', type=int, default=337, help='seed')
        parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup_epoch')

        # endregion yapf: enable

        return parser
