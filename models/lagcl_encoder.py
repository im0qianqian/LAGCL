import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from torch.nn.parameter import Parameter
from utils.sparse_mat_interface import GraphSparseMatInterface
from utils.spmm_utils import SpecialSpmm


def eliminate_zeros(x):
    indices = x.coalesce().indices()
    values = x.coalesce().values()

    mask = values.nonzero()
    nv = values.index_select(0, mask.view(-1))
    ni = indices.index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, x.shape)


class Discriminator(nn.Module):

    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.d = nn.Linear(in_features, in_features, bias=True)
        self.wd = nn.Linear(in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ft):
        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)

        return prob


class Relation(nn.Module):

    def __init__(self, in_features, ablation):
        super(Relation, self).__init__()

        self.gamma_1 = nn.Linear(in_features, in_features, bias=False)
        self.gamma_2 = nn.Linear(in_features, in_features, bias=False)

        self.beta_1 = nn.Linear(in_features, in_features, bias=False)
        self.beta_2 = nn.Linear(in_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        # self.m_r = nn.Linear(in_features, in_features, bias=False)

        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        self.ablation = ablation

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, ft, neighbor):
        # todo: r 要干好自己的事情，别让梯度从 ft、neighbor 传上去
        ft = ft.detach()
        neighbor = neighbor.detach()
        if self.ablation == 3:
            m = ft + self.r - neighbor
        else:
            gamma = self.gamma_1(ft) + self.gamma_2(neighbor)
            gamma = self.lrelu(gamma) + 1.0

            beta = self.beta_1(ft) + self.beta_2(neighbor)
            beta = self.lrelu(beta)

            r_v = gamma * self.r + beta

            # transE
            m = ft + r_v - neighbor

            # transR
            # ft = self.m_r(ft)
            # neighbor = self.m_r(neighbor)
            # self.m = ft + self.r_v - neighbor

            # self.m = self.gamma_1(ft) + self.r_v - self.gamma_2(neighbor)

            # transH
            # norm = F.normalize(self.r_v)
            # h_ft = ft - norm * torch.sum((norm * ft), dim=1, keepdim=True)
            # h_neighbor = neighbor - norm * torch.sum((norm * neighbor), dim=1, keepdim=True)
            # self.m = h_ft - h_neighbor

        return m  # F.normalize(self.m)


class RelationF(nn.Module):

    def __init__(self, nfeat):
        super(RelationF, self).__init__()
        self.fc1 = nn.Linear(nfeat * 4, nfeat)
        self.fc2 = nn.Linear(nfeat, nfeat)
        # self.self_attn = SelfAttention(nfeat, num_attention_heads=1, output_attentions=True)

    def forward(self, x, neighbor, masked_neighbor):
        # todo: 保证 fc12 只干自己的事情，不能把梯度传给输入，且 output 需过滤为仅 user 的部分
        x = x.detach()
        neighbor = neighbor.detach()
        masked_neighbor = masked_neighbor.detach()

        ngb_seq = torch.stack(
            [x, neighbor, neighbor * x, (neighbor + x) / 2.0], dim=1)
        # missing_info = self.self_attn(ngb_seq)[0][:, 0, :]
        # missing_info = self.self_attn(ngb_seq)[0][:, 1, :] - neighbor   # 邻居聚合 x 的信息，产出 neighbor_plus，减去 neighbor 为邻居确实信息
        missing_info = self.fc1(ngb_seq.reshape(len(ngb_seq), -1))
        missing_info = F.relu(missing_info)
        missing_info = self.fc2(missing_info)
        support_out = missing_info - masked_neighbor
        return missing_info, support_out


class LightTailGCN(nn.Module):

    def __init__(self, nfeat, global_r=None, use_relation_rf=False):
        super(LightTailGCN, self).__init__()
        self.use_relation_rf = use_relation_rf
        if self.use_relation_rf:
            self.trans_relation = RelationF(nfeat)
        else:
            self.trans_relation = Relation(nfeat, 0)
        if global_r is not None:
            self.trans_relation = global_r
        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj, adj_norm, adj_node_degree, adj_with_loop,
                adj_with_loop_norm, adj_with_loop_norm_plus_1, head, res_adj,
                res_adj_norm):
        # 可观测邻居 pooling 表征，这里 adj 与 res_adj 均不含自环，因此 neighbor 没有自身信息
        neighbor = self.special_spmm(adj_norm, x)

        # 预测邻域缺失信息
        if self.use_relation_rf:
            masked_neighbor = torch.sparse.mm(res_adj_norm, x)
            missing_info, output = self.trans_relation(x, neighbor,
                                                       masked_neighbor)
        else:
            missing_info = self.trans_relation(x, neighbor)
            output = missing_info

        if head:
            h_k = self.special_spmm(adj_with_loop_norm, x)
        else:
            h_s = missing_info
            h_k = self.special_spmm(
                adj_with_loop_norm_plus_1,
                x) + h_s / (adj_node_degree + 2).reshape(-1, 1)
        return h_k, output


class LAGCLEncoder(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_hops: int,
        tail_k_threshold: int = 5,
        cl_rate: float = 0.2,
        cl_tau: float = 0.2,
        m_h_eps: float = 0.01,
        kl_eps: float = 10.0,
        noise_eps: float = 0.1,
        agg_function: int = 0,
        neighbor_norm_type: str = 'left',
        use_relation_rf=False,
        agg_w_exp_scale=20,
    ):
        """
        Args:
            in_dim: 输入给 LAGCLEncoder 的节点表征维度，若与 hidden_dim 不同则会新增一层 Linear 进行转换
            hidden_dim: GNN Layer 中处理的节点表征维度
            n_hops: GNN 层数
            tail_k_threshold: 长尾图表征根据节点度区分头尾节点的阈值 k，设定为 -1 则退化为 LightGCN
            cl_rate: 对节点表征通过噪声扰动的方式产出 view1, view2 并应用对比学习的损失权重
            cl_tau: 对比学习 InfoNCE 中的 temperature 设置
            m_h_eps: 知识迁移模块中，学习节点邻域缺失信息的损失权重
            kl_eps: 为 pseudo head node 与 real head node 计算信息损失 KL 散度的权重
            noise_eps: 噪声扰动的幅度
            agg_function: 0 代表计算 n1 n2 节点间相似度时基于网络参数自动学习，1 代表直接使用余弦相似度进行计算
            neighbor_norm_type: 对邻居信息做聚合时的 norm 方式，默认为 left norm
            use_relation_rf: Knowledge Transfer Module 中的邻域缺失信息预测模块类型，设置为 True 使用基于 MLP 的方式
            agg_w_exp_scale: 在做完 Auto Drop 后将邻接矩阵放缩至 0/1 两端的程度，值越大则放缩程度越强
        """
        super().__init__()
        self.neighbor_norm_type = neighbor_norm_type
        self.tail_k_threshold = tail_k_threshold
        self.cl_rate = cl_rate
        self.cl_tau = cl_tau
        self.m_h_eps = m_h_eps
        self.kl_eps = kl_eps
        self.noise_eps = noise_eps
        self.agg_function = agg_function
        self.use_relation_rf = use_relation_rf
        self.agg_w_exp_scale = agg_w_exp_scale

        self.lin = nn.Linear(
            in_dim, hidden_dim) if in_dim != hidden_dim else lambda x: x

        self.hidden_dim = hidden_dim

        self.n_layers = n_hops
        self.x_weights = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # self.global_r = Relation(emb_size, ablation=3)
        self.rel_layers = nn.ModuleList([
            LightTailGCN(hidden_dim,
                         global_r=None,
                         use_relation_rf=self.use_relation_rf)
            for _ in range(self.n_layers)
        ])

    def encode(self, subgraph: dgl.DGLGraph, x: torch.Tensor):
        """
        Args:
            subgraph: dgl.DGLGraph
            x: initialized node feat

        Returns:
            embeddings or other information you need
        """
        ego_embeddings = self.lin(x)
        edge_index = torch.stack(subgraph.edges())
        node_degrees = subgraph.ndata['node_degree']
        node_types = subgraph.ndata['node_type']
        drop_node_mask = node_types == 0  # trick 只给 user 节点做 drop
        node_is_head_mask = node_degrees > self.tail_k_threshold
        node_is_head_mask[
            ~drop_node_mask] = True  # 对 drop_node_mask 以外的内容全部认为是 head

        # 去除已有的自环（For LightGCN）
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        n_nodes = len(ego_embeddings)
        n_edges = len(edge_index[0])

        # 构建 LAGCL 所需的各种邻接矩阵
        adj = torch.sparse.FloatTensor(edge_index,
                                       torch.ones(n_edges,
                                                  device=edge_index.device),
                                       size=(n_nodes, n_nodes)).coalesce()
        adj_norm, adj_node_degree = GraphSparseMatInterface.normalize_graph_mat(
            adj, norm=self.neighbor_norm_type, return_node_degree=True)
        adj_with_loop = adj + torch.sparse.FloatTensor(
            torch.arange(len(adj)).repeat(2, 1), torch.ones(len(adj))).to(
                adj.device)
        adj_with_loop_norm = GraphSparseMatInterface.normalize_graph_mat(
            adj_with_loop)
        adj_with_loop_norm_plus_1 = GraphSparseMatInterface.normalize_graph_mat(
            adj_with_loop, add_virtual_node_num=1)
        res_adj = torch.sparse.FloatTensor(
            torch.tensor([[0], [0]], dtype=torch.int64), torch.tensor([0.0]),
            adj.shape).to(adj.device)
        res_adj_norm = GraphSparseMatInterface.normalize_graph_mat(res_adj)

        enable_lagcl = self.tail_k_threshold >= 0

        # 对所有节点不做 auto drop，直接做聚合产出表征
        emb_h, support_h, _ = self.gather_pre(ego_embeddings,
                                              adj,
                                              adj_norm,
                                              adj_node_degree,
                                              adj_with_loop,
                                              adj_with_loop_norm,
                                              adj_with_loop_norm_plus_1,
                                              res_adj,
                                              res_adj_norm,
                                              head=True,
                                              use_auto_drop=False,
                                              drop_node_mask=None,
                                              node_is_head_mask=None,
                                              add_other_status=False)

        if enable_lagcl:
            # 如果启用了 LAGCL，则为尾部节点做增强
            emb_t, support_t, emb_nt = self.gather_pre(
                ego_embeddings,
                adj,
                adj_norm,
                adj_node_degree,
                adj_with_loop,
                adj_with_loop_norm,
                adj_with_loop_norm_plus_1,
                res_adj,
                res_adj_norm,
                head=False,
                use_auto_drop=True if self.training else False,
                drop_node_mask=drop_node_mask,
                node_is_head_mask=node_is_head_mask,
                add_other_status=True)

            # 整合构建最终的 embedding
            node_emb = emb_h * node_is_head_mask.long().reshape(
                -1, 1) + emb_t * (1 - node_is_head_mask.long().reshape(-1, 1))
        else:
            node_emb = emb_h
            emb_nt = emb_t = emb_h
            support_t = support_h

        other_embs_dict = {
            # 不补邻居、不裁剪
            'head_true_drop_false': emb_h,
            # 不补邻居、裁剪
            'head_true_drop_true': emb_nt,
            # 补邻居、裁剪
            'head_false_drop_true': emb_t,
            'support_h': support_h,
            'support_t': support_t,
        }

        return node_emb, other_embs_dict

    def gather_pre(self,
                   ego_embeddings,
                   adj,
                   adj_norm,
                   adj_node_degree,
                   adj_with_loop,
                   adj_with_loop_norm,
                   adj_with_loop_norm_plus_1,
                   res_adj,
                   res_adj_norm,
                   head,
                   use_auto_drop,
                   drop_node_mask,
                   node_is_head_mask,
                   add_other_status=False):
        """
        add_other_status 若 head 是 True，增加 head 是 False 时的表征，反之亦然
        """
        tail_k_threshold = self.tail_k_threshold
        assert tail_k_threshold != 0
        if use_auto_drop and tail_k_threshold > 0:
            indices = adj.indices()
            node_need_drop = drop_node_mask[indices[0]]
            indices = indices.t()[node_need_drop].t()
            if self.agg_function == 0:
                # 自定义学习 attention 的方式
                ego_norm = torch.max(
                    ego_embeddings.norm(dim=1)[:, None],
                    torch.zeros(
                        len(ego_embeddings), 1, device=ego_embeddings.device) +
                    1e-8)
                normd_emb = ego_embeddings / ego_norm

                agg_w = (self.x_weights.weight[0] * normd_emb[indices[0]] *
                         normd_emb[indices[1]]).sum(dim=1)

                agg_w = torch.nn.Softsign()(agg_w)
                agg_w = torch.nn.Softsign()(torch.exp(agg_w *
                                                      self.agg_w_exp_scale))
            else:
                # 余弦相似度
                sims = F.cosine_similarity(ego_embeddings[indices[0]],
                                           ego_embeddings[indices[1]])
                sims = torch.nn.Softsign()(torch.exp(sims *
                                                     self.agg_w_exp_scale))
                agg_w = sims

            head_to_tail_sample_type = 'top-k'  # topk or mantail-k
            if head_to_tail_sample_type == 'top-k':
                drop_node_is_head_mask = node_is_head_mask[drop_node_mask]
                # 所有节点邻居采样数目最多是 k 个，具体数目随机生成
                # 大致思路，先给每个节点随机分配一个 1-k 的随机整数，再将该节点的出边都设置成此类型，采样时不同类型边采样不同的邻居
                k = {i: i for i in range(1, tail_k_threshold + 1)}

                # 方案1（当前）：在采样产出 node_type 时随机等频采样，这会让伪标签中每个 degree 的数量均匀
                # 方案2：在采样产出 node_type 时根据真实尾部节点数量分布采样，这会让伪标签中每个 degree 的数量与真实尾部的数量相近
                node_type = torch.randint(1,
                                          tail_k_threshold + 1,
                                          (len(ego_embeddings), ),
                                          device=indices.device)

                node_type[drop_node_mask][
                    ~drop_node_is_head_mask] = tail_k_threshold  # 本身是尾部的点不做裁剪，保持原邻接数量
                edge_type = node_type[indices[0]]
                data_dict = {}
                edata = {}
                for edge_type_idx in k.keys():
                    select_mask = edge_type == edge_type_idx
                    nen_type = ('node', edge_type_idx, 'node')
                    data_dict[nen_type] = (indices[0][select_mask],
                                           indices[1][select_mask])
                    edata[nen_type] = agg_w[select_mask]
                g = dgl.heterograph(data_dict)
                g.edata['weight'] = edata

                sampled_g = dgl.sampling.sample_neighbors(
                    g,
                    nodes=g.nodes(),
                    fanout=k,
                    edge_dir='out',
                    prob='weight',
                    output_device=g.device)
                all_edges = []
                all_agg_w = []
                for etype in k.keys():
                    all_edges.append(torch.stack(sampled_g.edges(etype=etype)))
                    all_agg_w.append(sampled_g.edata['weight'][('node', etype,
                                                                'node')])
                all_edges = torch.cat(all_edges, dim=1)
                all_agg_w = torch.cat(all_agg_w)
            elif head_to_tail_sample_type == 'mantail-k':
                # 所有节点都采样最多 k 个邻居的模式
                g = dgl.graph((indices[0], indices[1]))
                g.edata['weight'] = agg_w
                # sampled_g = dgl.sampling.select_topk(g.to('cpu'), tail_k_threshold, 'weight', edge_dir='out')
                sampled_g = dgl.sampling.sample_neighbors(
                    g,
                    nodes=g.nodes(),
                    fanout=tail_k_threshold,
                    edge_dir='out',
                    prob='weight')
                all_edges = sampled_g.edges()
                all_agg_w = sampled_g.edata['weight']

            # todo: 对于对称矩阵来说，这里必须要加，对于 mini-batch 这里不需要做反向边
            tail_indices = torch.stack([
                torch.cat([all_edges[0], all_edges[1]]),
                torch.cat([all_edges[1], all_edges[0]])
            ])
            tail_values = torch.cat([all_agg_w, all_agg_w])

            tail_adj = torch.sparse.FloatTensor(
                tail_indices, tail_values, adj.shape).coalesce().to(adj.device)
            tail_adj_norm, tail_adj_node_degree = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj,
                norm=self.neighbor_norm_type,
                return_node_degree=True)
            tail_adj_with_loop = tail_adj + torch.sparse.FloatTensor(
                torch.arange(len(tail_adj)).repeat(2, 1),
                torch.ones(len(tail_adj))).to(adj.device)
            tail_adj_with_loop_norm = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj_with_loop)
            tail_adj_with_loop_norm_plus_1 = GraphSparseMatInterface.normalize_graph_mat(
                tail_adj_with_loop, add_virtual_node_num=1)
            tail_res_adj = eliminate_zeros(adj - tail_adj)
            tail_res_adj_norm = GraphSparseMatInterface.normalize_graph_mat(
                tail_res_adj)

            adj, adj_norm, adj_node_degree = tail_adj, tail_adj_norm, tail_adj_node_degree
            adj_with_loop, adj_with_loop_norm = tail_adj_with_loop, tail_adj_with_loop_norm
            adj_with_loop_norm_plus_1 = tail_adj_with_loop_norm_plus_1
            res_adj, res_adj_norm = tail_res_adj, tail_res_adj_norm

        all_status_embeddings = {True: [], False: []}
        all_status_support_outs = {True: [], False: []}
        ego_embeddings1 = ego_embeddings2 = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings1, output1 = self.rel_layers[k](
                ego_embeddings1, adj, adj_norm, adj_node_degree, adj_with_loop,
                adj_with_loop_norm, adj_with_loop_norm_plus_1, head, res_adj,
                res_adj_norm)
            if add_other_status:
                ego_embeddings2, output2 = self.rel_layers[k](
                    ego_embeddings2, adj, adj_norm, adj_node_degree,
                    adj_with_loop, adj_with_loop_norm,
                    adj_with_loop_norm_plus_1, not head, res_adj, res_adj_norm)
            else:
                ego_embeddings2, output2 = ego_embeddings1, output1
            all_status_embeddings[head].append(ego_embeddings1)
            all_status_embeddings[not head].append(ego_embeddings2)
            all_status_support_outs[head].append(output1)
            all_status_support_outs[not head].append(output2)

        def agg_all_layers_out(all_embeddings, backbone_name='lightgcn'):
            if backbone_name == 'lightgcn':
                # LightGCN 模型输出为各个层的平均
                all_embeddings = torch.stack(all_embeddings, dim=1)
                all_embeddings = torch.mean(all_embeddings, dim=1)
            elif backbone_name == 'gcn':
                # GCN 模型的输出为最后一层
                all_embeddings = all_embeddings[-1]
            return all_embeddings

        all_embeddings = agg_all_layers_out(all_status_embeddings[head])
        all_embeddings_other = agg_all_layers_out(
            all_status_embeddings[not head])

        return all_embeddings, all_status_support_outs[
            head], all_embeddings_other

    def forward(
        self,
        subgraph: dgl.DGLGraph,
        x: torch.Tensor,
    ):
        return self.encode(subgraph, x)