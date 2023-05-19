# 仅限book数据集生成提交文件使用
import torch
from DataSet import DataSet
from test import Test
from LightGCN import LightGCN
import pandas as pd
epoch = 45
hidden_dim = 64
n_layers = 3
topk = 10
alpha_k = [1/(layer+1) for layer in range(1+n_layers)]
lr = 0.001
decay = 1e-06

data_path = "./data/book/"
test_user_data_path = data_path+'test_dataset.csv'
load_path = data_path + \
    f"model/{epoch}_{hidden_dim}_{n_layers}_{lr}_{decay}.pth"
submission_path = data_path + \
    f'bce_{epoch}_{hidden_dim}_{n_layers}_{lr}_{decay}.csv'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
origin_train_data = DataSet(data_path + 'train.txt', device)

model = LightGCN(origin_train_data, hidden_dim,
                 n_layers, alpha_k, True)
model.load_state_dict(torch.load(load_path))
model = model.to(device)

df = pd.read_csv(test_user_data_path)
user_for_test = df['user_id'].to_list()
batch_users_gpu = torch.arange(
    0, origin_train_data.user_num, dtype=torch.int32).to(device)
model.eval()
with torch.no_grad():
    rating = model.getUsersRating(batch_users_gpu)
    exclude_index = []
    exclude_items = []
    for user_id, items in origin_train_data.user_item_map.items():
        exclude_index.extend([user_id] * len(items))
        exclude_items.extend(items)
    rating[exclude_index, exclude_items] = 0  # 因为sigmoid输出是(0,1)
    _, topk_indices = torch.topk(
        rating, k=topk)  # 降序排列的top-k个待推荐物品矩阵

    with open(submission_path, "w", encoding="utf8") as fp:
        fp.write("user_id,item_id\n")
        for user_id in user_for_test:
            rec_list = topk_indices[user_id].tolist()  # 模型的推荐列表
            for item_id in rec_list:
                fp.write(f"{user_id},{item_id}\n")
