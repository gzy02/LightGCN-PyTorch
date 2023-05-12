import os
import pickle
import random
import tqdm
import numpy as np
from config import config
from DataSet import DataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter


class SampleData(Dataset):
    def __init__(self, origin_data: DataSet, fake_num: int):
        super().__init__()
        self.user_item_map = origin_data.user_item_map
        self.item_num = origin_data.item_num
        self.fake_num = fake_num
        self.item_weight = np.array(origin_data.item_degree, dtype=float)
        self.item_weight /= self.item_weight.sum()
        self.Xs = []
        self.__init_dataset()

    def __init_dataset(self):
        item_indices = np.arange(self.item_num)
        for user, items in tqdm.tqdm(self.user_item_map.items()):
            for item in items:
                self.Xs.append((user, item, 1))  # 有交互则为1

            need = len(items)*self.fake_num
            items_set = items
            neg_samples = Counter()
            while need > 0:
                # 根据 item_weight 抽取 fake_num 个样本
                neg_items = Counter(
                    np.random.choice(item_indices, size=need*2, p=self.item_weight, replace=True))

                # 只保留用户未交互过的物品
                for item in items_set:
                    if item in neg_items:
                        del neg_items[item]
                neg_items_len = sum(neg_items.values())
                if neg_items_len <= need:
                    neg_samples.update(neg_items)
                else:
                    neg_samples.update(
                        Counter(random.sample(list(neg_items.elements()), need)))
                need -= neg_items_len

            # assert len(items)*self.fake_num == sum(neg_samples.values())
            # 处理负样本，label 设为 0
            for neg_sample in neg_samples.elements():
                self.Xs.append((user, neg_sample, 0))

    def __getitem__(self, index):
        user_id, item_id, label = self.Xs[index]
        return user_id, item_id, label

    def __len__(self):
        return len(self.Xs)


def get_trainloader(origin_train_data: DataSet, trainloader_path: str) -> DataLoader:
    """获取供BCELoss"""
    if not os.path.exists(trainloader_path):
        sample_train_data = SampleData(origin_train_data, config.fake_num)
        trainloader = DataLoader(sample_train_data,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=False,
                                 num_workers=8)
        with open(trainloader_path, "wb") as fp:
            pickle.dump(trainloader, fp)
    else:
        with open(trainloader_path, "rb") as fp:
            trainloader = pickle.load(fp)
    return trainloader


class SampleDataPair(Dataset):
    def __init__(self, origin_data: DataSet):
        super().__init__()
        self.user_item_map = origin_data.user_item_map
        self.item_num = origin_data.item_num
        self.item_weight = np.array(origin_data.item_degree, dtype=float)
        self.item_weight /= self.item_weight.sum()
        self.Xs = []
        self.__init_dataset()

    def __init_dataset(self):
        item_indices = np.arange(self.item_num)
        for user, items in tqdm.tqdm(self.user_item_map.items()):
            for item in items:
                self.Xs.append((user, item, 1))  # 有交互则为1

            need = len(items)
            items_set = items
            neg_samples = Counter()
            while need > 0:
                # 根据 item_weight 抽取 fake_num 个样本
                neg_items = Counter(
                    np.random.choice(item_indices, size=need*2, p=self.item_weight, replace=True))

                # 只保留用户未交互过的物品
                for item in items_set:
                    if item in neg_items:
                        del neg_items[item]
                neg_items_len = sum(neg_items.values())
                if neg_items_len <= need:
                    neg_samples.update(neg_items)
                else:
                    neg_samples.update(
                        Counter(random.sample(list(neg_items.elements()), need)))
                need -= neg_items_len

            # assert len(items_set) == sum(neg_samples.values())
            zipped = zip(items_set, neg_samples.elements())
            for item, neg in zipped:
                self.Xs.append((user, item, neg))

    def __getitem__(self, index):
        user_id, pos, neg = self.Xs[index]
        return user_id, pos, neg

    def __len__(self):
        return len(self.Xs)


def get_trainloaderPair(origin_train_data: DataSet, trainloader_path: str) -> DataLoader:
    """获取供BCELoss"""
    if not os.path.exists(trainloader_path):
        sample_train_data = SampleDataPair(origin_train_data)
        trainloader = DataLoader(sample_train_data,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=False,
                                 num_workers=8)
        with open(trainloader_path, "wb") as fp:
            pickle.dump(trainloader, fp)
    else:
        with open(trainloader_path, "rb") as fp:
            trainloader = pickle.load(fp)
    return trainloader
