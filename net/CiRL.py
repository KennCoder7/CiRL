import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from net.attention_block import CrossAttention, QueriesEmbedding


class Model(nn.Module):
    """ NPNM: Nearest Positive Neighbor Mining """

    def __init__(self, base_encoder, encoder_args, n_classes, d_features,
                 encoder_pretrained='',
                 queue_size=32768, cls_frozen=False, mask_out=True,
                 topk=1,
                 momentum=0.999, Temperature=0.07, mlp=True,
                 cross_attention_args=None,
                 contrastive_intra_temperature=0.07,
                 queries_emb_layers=1,
                 ):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)

        self.K = queue_size
        self.m = momentum
        self.T = Temperature

        self.encoder_q = base_encoder(**encoder_args)
        if encoder_pretrained is not '':
            self.encoder_q.load_state_dict(torch.load(encoder_pretrained))
        del self.encoder_q.fcn  # remove the final fc layer in stgcn
        self.encoder_k = base_encoder(**encoder_args)
        del self.encoder_k.fcn

        if mlp:  # hack: brute-force replacement
            dim_mlp = d_features
            self.fc_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                      nn.ReLU(),
                                      nn.Linear(dim_mlp, dim_mlp))
            self.fc_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                      nn.ReLU(),
                                      nn.Linear(dim_mlp, dim_mlp))
            for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(d_features, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_label", torch.full([queue_size], -1, dtype=torch.long))

        self.cls_head = nn.Linear(d_features, n_classes)
        self.cls_head.weight.data.normal_(mean=0.0, std=0.01)
        self.cls_head.bias.data.zero_()
        self.cls_frozen = cls_frozen
        self.mask_out = mask_out
        self.topk = topk

        self.contrastive_intra_temperature = contrastive_intra_temperature
        self.cross_attention = CrossAttention(**cross_attention_args)
        self.n_queries = n_classes
        self.learned_queries = nn.Embedding(self.n_queries, d_features)
        self.learned_queries.weight.data.normal_(mean=0.0, std=0.01)
        self.queries_emb_layers = queries_emb_layers
        if queries_emb_layers > 0:
            self.queries_embedding = QueriesEmbedding(n_queries=self.n_queries,
                                                      d_queries_in=d_features,
                                                      d_queries_out=d_features,
                                                      n_layers=queries_emb_layers)
        self.apply(weights_init)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def copy_params(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.fc_q.parameters(), self.fc_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
        self.queue_label[(ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = labels
        self.update_ptr(batch_size)

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, labels=None, NPNM=False, att_map=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        # compute query features
        local_features = self.encoder_q.extract_feature_person(im_q)  # queries: N, M, C
        global_features = local_features.mean(dim=1)  # N, C
        if self.cls_frozen:
            logits = self.cls_head(global_features.clone().detach())
        else:
            logits = self.cls_head(global_features.clone())
        # stage 1  build learned_queries
        # learned_queries = self.learned_queries.weight.unsqueeze(0).tile([logits.size(0), 1, 1])  # N, L, C
        # if self.queries_emb_layers > 0:
        #     learned_queries = self.queries_embedding(learned_queries)
        learned_queries = self.learned_queries.weight  # L, C
        learned_queries = learned_queries.unsqueeze(0)  # 1, L, C
        if self.queries_emb_layers > 0:
            learned_queries = self.queries_embedding(learned_queries)  # 1, L, C
        learned_queries = learned_queries.tile([logits.size(0), 1, 1])  # N, L, C
        logits_contrastive_1 = self.contrastive_intra(features=global_features,
                                                      learned_queries=learned_queries)

        # stage 2  cross attention
        if att_map:
            learned_queries_ca, att_map = self.cross_attention(cross_ft=local_features, ft=learned_queries, return_att_map=True)
            return learned_queries_ca, att_map
        learned_queries_ca = self.cross_attention(cross_ft=local_features, ft=learned_queries)  # N, L, C
        logits_contrastive_2 = self.contrastive_sim(learned_queries=learned_queries,
                                                    learned_queries_ca=learned_queries_ca)

        if im_k is None:
            return logits, logits_contrastive_1, logits_contrastive_2

        q = self.fc_q(global_features)
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k.extract_feature_person(im_k)  # keys: NxC
            k = k.mean(dim=1)  # N, C
            k = self.fc_k(k)
            k = F.normalize(k, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # NPNM
        if NPNM:
            mask = torch.eq(labels.unsqueeze(-1), self.queue_label.clone().detach()).float()  # NxK
            l_neg_positives = l_neg * mask
            _, topk_idx = torch.topk(l_neg_positives, self.topk, dim=1)
            topk_onehot = torch.zeros_like(l_neg_positives).scatter_(1, topk_idx, 1)
            mask.scatter_(1, topk_idx, 0.)
            cl_labels = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
        else:
            mask = torch.eq(labels.unsqueeze(-1), self.queue_label.clone().detach()).float()  # NxK
            cl_labels = torch.zeros(l_neg.size(0), dtype=torch.long).cuda()

        if self.mask_out:  # mask-out positive samples which are regarded as negative
            l_neg = l_neg - mask * 1e9
        # logits: Nx(1+K)
        cl_logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        cl_logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return cl_logits, cl_labels, logits, logits_contrastive_1, logits_contrastive_2

    def contrastive_intra(self, features, learned_queries):
        # features: N, C
        # learned_queries: N, L, C
        features = F.normalize(features, dim=1)  # N, C
        learned_queries = F.normalize(learned_queries, dim=2)  # N, L, C
        logits = torch.einsum('nc,nlc->nl', features, learned_queries)  # N, L
        logits = logits / self.contrastive_intra_temperature
        return logits

    def contrastive_sim(self, learned_queries, learned_queries_ca):
        # learned_queries: N, L, C
        # learned_queries_ca: N, L, C
        learned_queries = F.normalize(learned_queries, dim=2)  # N, L, C
        learned_queries_ca = F.normalize(learned_queries_ca, dim=2)  # N, L, C
        logits = torch.einsum('nlc,nlc->nl', learned_queries, learned_queries_ca)  # N, L
        logits = logits / self.contrastive_intra_temperature
        return logits


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