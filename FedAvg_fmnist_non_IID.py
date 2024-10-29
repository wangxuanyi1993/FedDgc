import os.path
import pickle
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math, random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
from fedlab.contrib.dataset import PartitionedFMNIST
from torch.utils.data import Dataset, DataLoader
# from util.Models import Mnist_CNN,Mnist_2NN


class Net(nn.Module):
    def __init__(self, in_channel, batch_size, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.dropout = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.dropout = nn.Dropout(0.25)

        self.linner1 = nn.Linear(6272, 512)
        self.dropout = nn.Dropout(0.25)
        self.linner2 = nn.Linear(512, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(self.relu(x))

        x = self.conv2(x)
        x = self.maxPool1(self.batchnorm2(self.relu(x)))

        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchnorm3(self.relu(x))

        x = self.conv4(x)
        x = self.maxPool2(self.batchnorm4(self.relu(x)))

        x = self.dropout(x)

        x = self.flatten(x)

        x = self.linner1(x)
        x = self.dropout(x)
        x = self.linner2(x)

        output = F.log_softmax(x, dim=1)

        return output

# 定义热身训练函数
def warm_train(model, device, train_loader, optimizer):
    model.train()
    # 分批次读入数据
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 优化器梯度置零
        optimizer.zero_grad()
        # 数据通过网络
        output = model(data)
        # 计算损失
        loss = torch.nn.functional.cross_entropy(output,target)
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()

# 定义训练函数
def train(model, device, train_loader, optimizer):
    model.train()
    # 分批次读入数据
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # 优化器梯度置零
        optimizer.zero_grad()
        # 数据通过网络
        output = model(data)
        # 计算损失
        loss = torch.nn.functional.cross_entropy(output,target)
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()


# 定义测试函数
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    # 不进行计算图的构建，即没有grad_fn属性
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output,target,reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nEpoch : {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        epoch,test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

    write.writerow({'EPOCH': epoch, 'ACC': 100. * correct / len(test_loader.dataset), 'LOSS': test_loss})
    # writer.add_scalar('Loss/test', test_loss, epoch)
    # writer.add_scalar('Accuracy/test', 100. * correct / len(test_loader.dataset), epoch)

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # csvfile = open('result/FedAvg_fmnist_unbalance_noiid_uniform_nowarm.csv', mode='w', newline='')
    csvfile = open('Invalid_results/FedAvg_fmnist_unbalance_noiid_ascend_nowarm_338.csv', mode='a', newline='')
    fieldnames = ['EPOCH', 'ACC', 'LOSS']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)
    write.writeheader()

    # writer = SummaryWriter(log_dir='FedAvg/Cifar10/Unbalance_IID',filename_suffix="zhangwenjv")

    EPOCHS = 300
    LR = 0.01
    SEED = 338
    CLIENT_NUM = 100
    DIR = 'datasets'
    CLASSES = 10
    WEIGHT_DECAY = 0.0001
    RHO = 0.01
    ALPHA = 0.6
    MAX_DELAY = 1
    BATCH_SIZE = 10
    ITERATIONS = 1
    # 自适应参数类型 none power hige
    ALPHA_TYPE = 'power'
    ALPHA_ADAPTIVE = 0.6
    ALPHA_ADAPTIVE2 = 1
    INTERVAL = 10
    TRAIN_NUM = 10
    GLOBAL_EPOCH = 300

    epochs = EPOCHS
    lr = LR
    seed = SEED
    client_num = CLIENT_NUM
    datasets_dir = DIR
    classes = CLASSES
    weight_decay = WEIGHT_DECAY
    rho = RHO
    alpha = ALPHA
    max_delay = MAX_DELAY
    batch_size = BATCH_SIZE
    iterations = ITERATIONS
    alpha_type = ALPHA_TYPE
    alpha_adaptive = ALPHA_ADAPTIVE
    alpha_adaptive2 = ALPHA_ADAPTIVE2
    interval = INTERVAL
    train_num = TRAIN_NUM
    global_epoch = GLOBAL_EPOCH

    # preprocess=False不在进行数据切割
    hetero = PartitionedFMNIST(
        root='.',
        path='datasets/fmnist_unbalance_dirichlet',
        num_clients=100,
        download=True,
        preprocess=False,
        partition="noniid-labeldir",
        verbose=True,
        seed=336,
        dir_alpha=0.3,
        transform=transforms.Compose([
            transforms.ToTensor()]
        ),
        target_transform=transforms.ToTensor()
    )

    # 测试集
    test_data = datasets.FashionMNIST(
        root=".",
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()]
        )
    )
    test_set = DataLoader(test_data, batch_size=1000, shuffle=False)


    # 设置种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 模型初始化
    model = Net(1, 10, 10).to(device)
    # model = Mnist_CNN().to(device)
    # model = Mnist_2NN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 热身训练
    # 加载热身训练数据
    train_data = hetero.get_dataloader(0,batch_size=batch_size)
    # for i in range(1):
    #     warm_train(model, device, train_data, optimizer)
    test(model, device, test_set, 0)

    # 保存模型
    params_prev = [param.clone() for param in model.parameters()]
    params_prev_list = [params_prev]


    model_cache = []


    # 全局训练
    for epoch in range(global_epoch):

        # 上升
        train_num = math.ceil((epoch + 1) / 15)
        # 统一
        # train_num = 10
        # 下降
        # train_num = 21 - math.ceil((epoch + 1) / 10)

        for i in range(train_num):

            # 读取全局模型
            params_prev = params_prev_list[0]

            # 加载选择模型
            for param, param_prev in zip(model.named_parameters(), params_prev):
                name = param[0]
                model.state_dict()[name].copy_(param_prev.clone())

            # 客户端训练
            client_id = random.randint(0,client_num-1)
            train_data = hetero.get_dataloader(client_id, batch_size=batch_size)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            for local_epoch in range(iterations):
                train(model, device, train_data, optimizer)

            # 模型存储至缓存
            params_prev = [param.clone() for param in model.parameters()]
            model_cache.append(params_prev)

            if len(model_cache) == train_num:

                # 模型聚合
                agg_model = [torch.zeros_like(param_prev) for param_prev in params_prev]

                for local_model in model_cache:
                    for x, y in zip(agg_model, local_model):
                        x += y

                for layer in agg_model:
                    layer[:] = layer / train_num

                # push
                params_prev_list.append(agg_model)
                # pop
                del params_prev_list[0]
                # 清除缓存
                model_cache.clear()

        # 模型测试
        if epoch > 0:
            if epoch % interval == 0 or epoch == epochs - 1:
                # 装载模型
                for param, param_prev in zip(model.named_parameters(), params_prev_list[-1]):
                    name = param[0]
                    model.state_dict()[name].copy_(param_prev.clone())
                # 测试集获取准确率
                test(model, device, test_set, epoch)

