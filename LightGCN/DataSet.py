import torch
import math
from torch import Tensor
from typing import Dict, Set, List


def get_user_item_map(test_data_path: str) -> Dict[int, Set[int]]:
    user_item_map: Dict[int, Set[int]] = dict()  # 邻接表
    with open(test_data_path, "r") as fp:
        for line in fp.readlines():
            line_list = line.split(' ')
            item_set = set()
            for i in range(1, len(line_list)):
                item_id = int(line_list[i])
                item_set.add(item_id)
            user_id = int(line_list[0])
            user_item_map[user_id] = item_set
    return user_item_map


class DataSet():
    def __init__(self, data_path: str, device: str):
        self.user_item_map: Dict[int, Set[int]] = dict()  # 邻接表
        self.item_num: int = 0
        self.user_num: int = 0

        # 构造邻接表
        train_data_path = data_path
        with open(train_data_path, "r") as fp:
            for line in fp.readlines():
                line_list = line.split(' ')
                item_set = set()
                for i in range(1, len(line_list)):
                    item_id = int(line_list[i])
                    item_set.add(item_id)
                    self.item_num = max(self.item_num, item_id)

                user_id = int(line_list[0])
                self.user_num = max(self.user_num, user_id)
                self.user_item_map[user_id] = item_set

        self.item_num += 1  # 最小序号是0
        self.user_num += 1  # 最小序号是0

        self.item_degree: List[int] = [0] * self.item_num  # 物品热门度
        self.user_degree: List[int] = [0] * self.user_num  # 用户交互数
        for user, items in self.user_item_map.items():
            self.user_degree[user] = len(items)
            for item in items:
                self.item_degree[item] += 1

        exclude_index = []
        exclude_items = []
        for user_id, items in self.user_item_map.items():
            exclude_index.extend([user_id] * len(items))
            exclude_items.extend(items)
        self.interactions_num = len(exclude_items)
        self.exclude_index = torch.LongTensor(exclude_index).to(device)
        self.exclude_items = torch.LongTensor(exclude_items).to(device)

        adj_matrix = self.__init_adj_matrix()
        L2_degree_matrix = self.__init_L2_degree_matrix()
        self.norm_adj_matrix = torch.sparse.mm(torch.sparse.mm(
            L2_degree_matrix, adj_matrix), L2_degree_matrix).to(device)

    def __init_L2_degree_matrix(self):
        rows = []
        cols = []
        values = []
        for i in range(self.user_num):
            rows.append(i)
            cols.append(i)
            values.append(1/math.sqrt(self.user_degree[i]))
        for i in range(self.item_num):
            rows.append(i+self.user_num)
            cols.append(i+self.user_num)
            values.append(1/math.sqrt(self.item_degree[i]))

        # 将行、列和值列表转换为 PyTorch 张量
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        values = torch.FloatTensor(values)

        # 构建稀疏 COO 张量
        sparse_L2_degree_matrix = torch.sparse_coo_tensor(indices=torch.stack(
            [rows, cols]), values=values, size=(self.user_num+self.item_num, self.item_num+self.user_num), dtype=torch.float32)

        return sparse_L2_degree_matrix

    def __init_adj_matrix(self):
        """
        build a graph in torch.sparse.IntTensor
        A =
            |0,   R|
            |R^T, 0|
        """
        # 列表用于存储稀疏矩阵非零位置及其值
        rows = []
        cols = []
        values = []

        # 遍历邻接表（user_item_map），添加非零值的位置和值
        for user_id, items in self.user_item_map.items():
            for item_id in items:
                rows.append(user_id)
                cols.append(item_id+self.user_num)
                values.append(1)  # 邻接矩阵中所有的值都是 1

                rows.append(self.user_num+item_id)
                cols.append(user_id)
                values.append(1)

        # 将行、列和值列表转换为 PyTorch 张量
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        values = torch.FloatTensor(values)

        # 构建稀疏 COO 张量
        sparse_adj_matrix = torch.sparse_coo_tensor(indices=torch.stack(
            [rows, cols]), values=values, size=(self.user_num+self.item_num, self.item_num+self.user_num), dtype=torch.float32)

        return sparse_adj_matrix
