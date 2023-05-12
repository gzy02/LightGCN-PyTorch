# %% import
import torch

from config import config
from LightGCN import LightGCN
from test import Test
from rand import setup_seed
from SampleData import get_trainloader
from DataSet import DataSet, get_user_item_map

# %% 打印全局超参数
config.display()

# %% 设置随机数种子
setup_seed(config.seed)

# %% 选gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 初始化原始训练数据
origin_train_data = DataSet(config.data_path + 'train.txt', device)
print("用户数：", origin_train_data.user_num)
print("物品数：", origin_train_data.item_num)
print("用户节点度数最大值：", max(origin_train_data.user_degree))
print("用户节点度数最小值：", min(origin_train_data.user_degree))
print("物品节点度数最大值：", max(origin_train_data.item_degree))
print("物品节点度数最小值：", min(origin_train_data.item_degree))

# %% 负采样得到完整训练数据
trainloader = get_trainloader(origin_train_data, config.data_path+'sample.pkl')

# 将 DataLoader 中的全部数据集载入GPU，防止反复迁移数据
user_tensor = []
item_tensor = []
label_tensor = []
for data in trainloader:
    user, item, label = data
    user_tensor.append(user.to(device))
    item_tensor.append(item.to(device))
    label_tensor.append(label.float().to(device))

num_data_points = len(label_tensor)

# %% 初始化测试数据
test_user_item_map = get_user_item_map(config.data_path+'test.txt')
batch_users_gpu = torch.arange(
    0, len(test_user_item_map), dtype=torch.int64).to(device)

# %% 模型训练
model = LightGCN(origin_train_data, config.hidden_dim,
                 config.n_layers, config.load_weight)
if config.load_weight:
    model.load_state_dict(torch.load(config.load_path))
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=config.decay)

crit = torch.nn.BCELoss()  # 损失函数：BCELoss

for epoch in range(1+config.load_epoch, config.epochs+1+config.load_epoch):
    # 在训练集上训练
    losses = []
    # 提前将所有训练数据移入GPU，会快些，但每次epoch都不会shuffle
    for i in range(num_data_points):
        y_ = model(user_tensor[i], item_tensor[i])
        loss = crit(y_, label_tensor[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())

    # 也可以用trainloader，每次epoch都会shuffle，但会慢
    # for data in trainloader:
    #     user, item, label = data
    #     y_ = model(user.to(device), item.to(device))
    #     loss = crit(y_, label.to(device).float())
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.detach().cpu().item())

    print('Epoch {} finished, average loss {}'.format(
        epoch, sum(losses) / len(losses)))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), config.data_path+'model/{}_{}_{}_{}_{}.pth'.format(epoch,
                   config.hidden_dim, config.n_layers, config.lr, config.decay))
        precision, recall, nDCG, F1 = Test(
            model, batch_users_gpu, origin_train_data, test_user_item_map)
        print('Epoch {} : Precision@{} {}, Recall@{} {}, F1@{} {}, nDCG@{} {}.'.format(
            epoch, config.topk, precision, config.topk, recall, config.topk, F1, config.topk, nDCG))
