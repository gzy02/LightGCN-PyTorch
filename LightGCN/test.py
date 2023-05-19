# 测试并保存模型
import math
from config import config
from LightGCN import LightGCN, LR_LightGCN
import torch
from torch import Tensor
from typing import Dict, Set, Tuple
from DataSet import DataSet
from typing import Optional, Union


def Test(model: Union[LightGCN, LR_LightGCN], batch_users_gpu: Tensor, origin_train_data: DataSet, test_user_item_map: Dict[int, Set[int]]) -> Tuple[float, float, float, float]:
    """测试当前模型"""
    model.eval()
    with torch.no_grad():
        precision = 0
        recall = 0
        nDCG = 0
        # 测试开始
        rating = model.getUsersRating(batch_users_gpu)  # 模型预测的用户-物品交互矩阵
        # 去除用户交互过的物品
        rating[origin_train_data.exclude_index,
               origin_train_data.exclude_items] = 0.0  # 因为sigmoid输出是(0,1)
        _, topk_indices = torch.topk(
            rating, k=config.topk)  # 降序排列的top-k个待推荐物品矩阵
        for user_id, items in test_user_item_map.items():
            rec_list = topk_indices[user_id].tolist()  # 模型的推荐列表
            intersect = items.intersection(set(rec_list))  # 模型推荐与真实交互的交集
            if len(intersect) == 0:  # 此时precision, recall, nDCG均为0
                continue
            precision += len(intersect)/config.topk
            recall += len(intersect)/len(items)
            # 交互了：rel_i=1 否则：rel_i=0
            DCG = sum([1 / math.log2(index + 2)
                       for index, item in enumerate(rec_list) if item in items])
            # 推荐系统返回的最好结果：实际喜欢数和topk的最小值
            IDCG = sum([1 / math.log2(i + 2)
                        for i in range(min(config.topk, len(items)))])
            nDCG += DCG/IDCG

        precision /= len(test_user_item_map)
        recall /= len(test_user_item_map)
        nDCG /= len(test_user_item_map)
        F1 = 0.0 if precision == 0 or recall == 0 else 2 * \
            precision*recall/(precision+recall)
        return precision, recall, nDCG, F1
