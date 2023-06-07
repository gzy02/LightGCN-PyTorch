# %% import
import os
import torch

from config import config
from LightGCN import LightGCN, LR_LightGCN
from test import Test, Test_batch
from rand import setup_seed
from SampleData import get_trainloader, get_trainloaderPair
from DataSet import DataSet, get_user_item_map

# %% 打印全局超参数
pid = os.getpid()
print("pid =", pid)
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
print("用户-物品交互数：", origin_train_data.interactions_num)

# %% 初始化原始测试数据
test_user_item_map = get_user_item_map(config.data_path+'test.txt')

# %% 模型与优化器
model = LightGCN(origin_train_data, config.hidden_dim,
                 config.n_layers, config.alpha_k, config.load_weight)
if config.load_weight:
    model.load_state_dict(torch.load(config.load_path))
    print(f"Load model {config.load_path}")
model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=config.decay)

# %% 负采样得到完整训练数据后开始训练
if config.useBCELoss:
    # trainloader = get_trainloader(origin_train_data)
    crit = torch.nn.BCELoss()  # 损失函数：BCELoss

    for epoch in range(1+config.load_epoch, config.epochs+1+config.load_epoch):
        # 在训练集上训练
        losses = []
        trainloader = get_trainloader(origin_train_data, config.seed+epoch)
        for data in trainloader:
            user, item, label = data
            y_ = model(user.to(device), item.to(device))
            loss = crit(y_, label.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

        print('Epoch {} finished, average loss {}'.format(
            epoch, sum(losses) / len(losses)))
        precision, recall, nDCG, F1 = Test_batch(
            model, origin_train_data, test_user_item_map, device)
        print('Epoch {} : Precision@{} {}, Recall@{} {}, F1@{} {}, nDCG@{} {}'.format(
            epoch, config.topk, precision, config.topk, recall, config.topk, F1, config.topk, nDCG))

        # if epoch % 10 == 0:
        #    torch.save(model.state_dict(), config.data_path+'model/{}_{}_{}_{}_{}.pth'.format(
        #        epoch, config.hidden_dim, config.n_layers, config.lr, config.decay))

else:

    # trainloader = get_trainloaderPair(origin_train_data)
    for epoch in range(1+config.load_epoch, config.epochs+1+config.load_epoch):
        # 在训练集上训练
        losses = []

        trainloader = get_trainloaderPair(origin_train_data, config.seed+epoch)
        for data in trainloader:
            user, item, neg = data
            loss = model.bpr_loss(
                user.to(device), item.to(device), neg.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

        print('Epoch {} finished, average loss {}'.format(
            epoch, sum(losses) / len(losses)))
        precision, recall, nDCG, F1 = Test_batch(
            model, origin_train_data, test_user_item_map, device)
        print('Epoch {} : Precision@{} {}, Recall@{} {}, F1@{} {}, nDCG@{} {}'.format(
            epoch, config.topk, precision, config.topk, recall, config.topk, F1, config.topk, nDCG))

        # if epoch % 10 == 0:
        #    torch.save(model.state_dict(), config.data_path+'model/{}_{}_{}_{}_{}.pth'.format(
        #        epoch, config.hidden_dim, config.n_layers, config.lr, config.decay))
