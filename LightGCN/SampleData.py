import my_sample
from config import config
from DataSet import DataSet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, Set, List, Tuple


class SampleDataBCE(Dataset):
    def __init__(self, origin_data: DataSet, fake_num: int, seed: int):
        super().__init__()
        my_sample.set_seed(seed)
        self.Xs: List[Tuple[int, int, int]] = my_sample.sample_weightBCE(
            origin_data.user_item_map, origin_data.item_degree, fake_num)

    def __getitem__(self, index):
        user_id, item_id, label = self.Xs[index]
        return user_id, item_id, label

    def __len__(self):
        return len(self.Xs)


def get_trainloader(origin_train_data: DataSet, seed: int) -> DataLoader:
    """获取供BCELoss"""
    sample_train_data = SampleDataBCE(origin_train_data, config.fake_num, seed)
    trainloader = DataLoader(sample_train_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False,
                             num_workers=0)
    return trainloader


class SampleDataPair(Dataset):
    def __init__(self, origin_data: DataSet, seed: int):
        super().__init__()
        my_sample.set_seed(seed)
        # self.Xs: List[Tuple[int, int, int]] = my_sample.sample_weightBPR(
        #    origin_data.user_item_map, origin_data.item_degree)
        self.Xs: List[Tuple[int, int, int]] = my_sample.sample_randomBPR(
            origin_data.user_item_map, origin_data.item_num)

    def __getitem__(self, index):
        user_id, pos, neg = self.Xs[index]
        return user_id, pos, neg

    def __len__(self):
        return len(self.Xs)


def get_trainloaderPair(origin_train_data: DataSet, seed: int) -> DataLoader:
    """获取供BCELoss"""
    sample_train_data = SampleDataPair(origin_train_data, seed)
    trainloader = DataLoader(sample_train_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False,
                             num_workers=0)
    return trainloader
