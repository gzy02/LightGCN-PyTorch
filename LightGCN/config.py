# 定义超参
class config:
    seed = 2023
    dataSetName = 'gowalla'
    # 可选列表: ['book','amazon-book','gowalla','yelp2018']
    data_path = "./data/{}/".format(dataSetName)

    # 取样
    useBCELoss = True
    fake_num = 8  # BCELoss使用
    batch_size = 4096

    # 模型
    hidden_dim = 64
    n_layers = 1
    alpha_k = [1/(layer+1) for layer in range(n_layers+1)]

    load_weight = False
    load_epoch = 0
    load_path = data_path+f"./model/{load_epoch}_64_3_0.001_0.0001.pth"

    # 训练
    epochs = 1000
    lr = 1e-3
    decay = 1e-5

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
