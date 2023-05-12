# 定义超参
class config:
    seed = 2023
    dataSetName = 'gowalla'
    # 可选列表: ['book','amazon-book','gowalla','yelp2018']
    data_path = "./data/{}/".format(dataSetName)

    # 取样
    fake_num = 300
    batch_size = 4096

    # 模型
    hidden_dim = 1
    n_layers = 0

    load_weight = False
    load_epoch = 0
    load_path = data_path+""

    # 训练
    epochs = 1000
    lr = 1e-3
    decay = 0
    lambda_BCELoss = 0.2

    # 测试
    topk = 20

    @classmethod
    def display(cls):
        """打印参数"""
        print("-----------Configuration-----------")
        # 除去以双下划线开头的属性和可调用的属性后，打印出每个属性及其值
        for key, value in cls.__dict__.items():
            if not key.startswith('__') and not callable(value) and key != cls.display.__name__:
                print(f"{key}: {value}")
        print("----------------End----------------")


if __name__ == '__main__':
    config.display()
