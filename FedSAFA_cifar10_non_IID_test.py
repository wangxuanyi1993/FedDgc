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
from torchvision import transforms,datasets
from fedlab.contrib.dataset import PartitionedCIFAR10
from torch.utils.data import Dataset, DataLoader


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

        self.linner1 = nn.Linear(8192, 512)
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
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break

# 定义训练函数
def train(model, device, train_loader, optimizer, rho, params_prev):
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
        # 添加正则化
        if rho > 0:
            for param, param_prev in zip(model.named_parameters(),params_prev):
                name = param[0]
                layer = param[1]
                update_layer = layer * (1 - rho) + param_prev
                model.state_dict()[name].copy_(update_layer.clone())
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

        epoch ,test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))
    write.writerow({'EPOCH':epoch,'ACC': 100. * correct / len(test_loader.dataset), 'LOSS': test_loss})

def calculate_alpha_factor(worker_delay,alpha_type='power',alpha_adaptive=0.6,alpha_adaptive2=1):
    if alpha_type == 'none':
        alpha_factor = 1
    elif alpha_type == 'power':
        if alpha_adaptive > 0:
            alpha_factor = 1 / math.pow(worker_delay + 1.0, alpha_adaptive)
        else:
            alpha_factor = 1
    elif alpha_type == 'exp':
        alpha_factor = math.exp(-worker_delay * alpha_adaptive)
    elif alpha_type == 'sigmoid':
        # a soft-gated function in the range of (1/a, 1]
        # maximum value
        a = alpha_adaptive2
        # slope
        c = alpha_adaptive
        b = math.exp(- c * worker_delay)
        alpha_factor = (1 - b) / a + b
    elif alpha_type == 'hinge':
        if worker_delay <= alpha_adaptive2:
            alpha_factor = 1
        else:
            alpha_factor = 1.0 / ((worker_delay - alpha_adaptive2) * alpha_adaptive + 1)
            # alpha_factor = math.exp(- (worker_delay-args.alpha_adaptive2) * args.alpha_adaptive)
    else:
        alpha_factor = 1
    return alpha_factor

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csvfile = open('result/FedSAFA_cifar10_unbalance_noiid_nowarm_1_norho_24.csv', mode='a', newline='')
    fieldnames = ['EPOCH', 'ACC', 'LOSS']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)
    write.writeheader()

    EPOCHS = 2000
    LR = 0.01

    SEED = 336
    CLIENT_NUM = 100
    DIR = 'datasets'
    CLASSES = 10
    WEIGHT_DECAY = 0.0001
    # 修改：去除正则化
    RHO = 0
    # 修改：去除模型更新权重部分
    # 修改2：收敛太快，尝试降速
    # 去除alpha影响
    ALPHA = 1
    MAX_DELAY = 24
    BATCH_SIZE = 10
    ITERATIONS = 1
    # 自适应参数类型 none power hige
    ALPHA_TYPE = 'none'
    ALPHA_ADAPTIVE = 0.6
    ALPHA_ADAPTIVE2 = 1
    INTERVAL = 10
    # TRAIN_NUM = 10
    GLOBAL_EPOCH = 2000

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
    # train_num = TRAIN_NUM
    global_epoch = GLOBAL_EPOCH

    # preprocess=False不在进行数据切割
    hetero = PartitionedCIFAR10(
        root='.',
        path='datasets/cifar10_unbalance_dirichlet',
        dataname="cifar10",
        num_clients=100,
        download=False,
        preprocess=False,
        balance=True,
        partition="dirichlet",
        seed=seed,
        dir_alpha=0.3,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])]
        ),
        target_transform=transforms.ToTensor()
    )

    # 测试集
    test_data = datasets.CIFAR10(
        root=".",
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])]
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
    model = Net(3, 10, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 热身训练
    # 加载热身训练数据
    train_data = hetero.get_dataloader(0, batch_size=batch_size)
    # for i in range(1):
    #     warm_train(model, device, train_data, optimizer)
    test(model, device, test_set, 0)

    # 保存模型
    params_prev = [param.clone() for param in model.parameters()]
    params_prev_list = [params_prev]
    server_ts = 0

    # 设置时间戳列表
    ts_list = [server_ts]
    sum_delay = 0

    alpha_factor_cache = []
    model_cache = []
    worker_delay_cache = []
    data_len_cache = []

    # 全局训练
    for epoch in range(global_epoch):

        #train_num = min(math.ceil((epoch + 1) / 100), 10)
        # 上升
        # train_num = math.ceil((epoch + 1) / 100)
        # 统一
        train_num = 10
        # 下降
        # train_num = 21 - math.ceil((epoch + 1) / 100)

        for i in range(train_num):

            # 选择模型
            model_idx = random.randint(0, len(params_prev_list) - 1)
            params_prev = params_prev_list[model_idx]

            # 加载选择模型
            for param, param_prev in zip(model.named_parameters(), params_prev):
                name = param[0]
                model.state_dict()[name].copy_(param_prev.clone())

            params_prev = [param.clone() for param in model.parameters()]
            if rho > 0:
                for param_prev in params_prev:
                    param_prev[:] = param_prev * rho


            # 记录选取的模型
            worker_ts = ts_list[model_idx]
            print("worker_ts")
            print(worker_ts)

            # 客户端训练
            client_id = random.randint(0, client_num - 1)
            train_data = hetero.get_dataloader(client_id, batch_size=batch_size)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            for local_epoch in range(iterations):
                train(model, device, train_data, optimizer, rho, params_prev)

            # 计算模型延迟
            worker_delay = epoch - worker_ts
            sum_delay += worker_delay
            alpha_factor = calculate_alpha_factor(worker_delay, alpha_type)
            # alpha_scaled = alpha * alpha_factor

            # 计算数据量
            data_len = len(hetero.get_dataset(client_id))

            # 模型存储至缓存
            params_prev = [param.clone() for param in model.parameters()]
            for param_prev in params_prev:
                param_prev[:] = param_prev * alpha_factor * data_len

            model_cache.append(params_prev)
            alpha_factor_cache.append(alpha_factor)
            worker_delay_cache.append(worker_delay)
            data_len_cache.append(data_len)

            if len(model_cache) == train_num:
                print("模型聚合")
                print(epoch)
                print(alpha_factor_cache)
                print(worker_delay_cache)
                print(data_len_cache)

                # worker_delay聚合
                sum_worker_delay = 0
                for worker_delay in worker_delay_cache:
                    sum_worker_delay += worker_delay
                avg_worker_delay = sum_worker_delay / train_num
                avg_alpha_factor = calculate_alpha_factor(avg_worker_delay,alpha_type)
                # 源代码如下
                # alpha_scaled = alpha * avg_alpha_factor
                # 取消alpha修正
                alpha_scaled = alpha


                # alpha_factor聚合
                # sum_alpha_factor = 0
                # for alpha_factor in alpha_factor_cache:
                #     sum_alpha_factor += alpha_factor

                sum_alpha_factor = 0
                for alpha_f, data_l in zip(alpha_factor_cache, data_len_cache):
                    sum_alpha_factor += alpha_f * data_l

                # 模型聚合
                agg_model = [torch.zeros_like(param_prev) for param_prev in params_prev]
                # agg_model = []
                # for param_prev in model_cache[0]:
                #     x = torch.zeros_like(param_prev)
                #     agg_model.append(x)
                # print('--------------------------------')
                # print('生成')
                # print(agg_model[0][0][0][0])

                for local_model in model_cache:
                    for x, y in zip(agg_model, local_model):
                        x += y
                # print('--------------------------------')
                # print('聚合')
                # print(agg_model[0][0][0][0])
                for layer in agg_model:
                    layer[:] = layer / sum_alpha_factor
                # print('--------------------------------')
                # print('平均')
                # print(agg_model[0][0][0][0])

                for param, param_server in zip(agg_model, params_prev_list[-1]):
                    param[:] = param_server * (1 - alpha_scaled) + param * alpha_scaled
                # print('--------------------------------')
                # print('加和')
                # print(agg_model[0][0][0][0])

                # push
                params_prev_list.append(agg_model)
                ts_list.append(epoch + 1)
                # pop
                if len(params_prev_list) > max_delay:
                    del params_prev_list[0]
                    del ts_list[0]
                # 清除缓存
                alpha_factor_cache.clear()
                model_cache.clear()
                worker_delay_cache.clear()
                data_len_cache.clear()

        # 模型测试
        if epoch > 0:
            if epoch % interval == 0 or epoch == epochs - 1:
                # 装载模型
                for param, param_prev in zip(model.named_parameters(), params_prev_list[-1]):
                    name = param[0]
                    model.state_dict()[name].copy_(param_prev.clone())
                # 测试集获取准确率
                test(model, device, test_set, epoch)
