import torch
import torch.nn as nn
import copy
from typing import Dict, Set, List
from DataSet import DataSet


class LightGCN(nn.Module):
    def __init__(self, origin_data: DataSet, hidden_dim: int, n_layers: int,alpha_k:List[float], load_weight:bool):
        super().__init__()
        # 邻接表
        self.user_item_map = origin_data.user_item_map
        # 邻接矩阵
        self.graph = origin_data.norm_adj_matrix
        # 超参数
        self.hidden_dim = hidden_dim
        self.user_num = origin_data.user_num
        self.item_num = origin_data.item_num
        self.n_layers = n_layers
        self.alpha_k=alpha_k
        # 模型参数
        self.user_embedding = nn.Embedding(self.user_num, hidden_dim)
        self.item_embedding = nn.Embedding(self.item_num, hidden_dim)
        
        if load_weight==False:
            # nn.init.normal_(self.user_embedding.weight, std=0.1)
            # nn.init.normal_(self.item_embedding.weight, std=0.1)
            # print('use NORMAL distribution initilizer')

            nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
            nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
            print('use xavier initilizer')

    def LightGCNConvolute(self):
        new_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        sum_embedding = self.alpha_k[0]*new_embedding
        for layer in range(1,self.n_layers):
            next_embedding=torch.sparse.mm(self.graph,new_embedding)
            sum_embedding+=next_embedding*self.alpha_k[layer]
            new_embedding=next_embedding
        return torch.split(sum_embedding, [self.user_num, self.item_num])
        #return sum_embedding[:self.user_num],sum_embedding[self.user_num:]

    def forward(self, user, item):
        self.train()
        all_user_embedding, all_item_embedding = self.LightGCNConvolute()
        user_embedding, item_embedding = all_user_embedding[user], all_item_embedding[item]
        inner_pro = torch.mul(user_embedding, item_embedding)
        gamma = torch.sum(inner_pro, dim=1)
        output = torch.sigmoid(gamma)
        return output

    def bpr_loss(self, users, pos, neg):
        self.train()
        all_user_embedding, all_item_embedding = self.LightGCNConvolute()
        users_emb, pos_emb, neg_emb = all_user_embedding[users], all_item_embedding[pos], all_item_embedding[neg]
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss
    
    def getUsersRating(self, users):
        self.eval()
        with torch.no_grad():  # 不求导
            all_users, all_items = self.LightGCNConvolute()
            users_emb = all_users[users]
            items_emb = all_items
            rating = torch.sigmoid(torch.matmul(users_emb, items_emb.t()))
            return rating
    