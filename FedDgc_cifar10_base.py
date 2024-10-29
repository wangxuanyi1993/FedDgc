import os.path
import pickle
import numpy as np
import argparse, time, logging, os, math, random
from os import listdir
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 读取自定义数据集
class CustomedDataSet(Dataset):
    def __init__(self, train=True, train_x = None, train_y = None, test_x = None, test_y = None, val = False, transform = None):
        self.train = train
        self.val = val
        self.transform = transform
        if self.train:
            self.dataset=train_x
            self.labels=train_y
        elif val:
            self.dataset=test_x
            self.labels=test_y
        else:
            self.dataset= test_x

    def __getitem__(self, index):
        if self.train:
            #return torch.Tensor(self.dataset[index]).to(device), self.labels[index].to(device)
            return self.dataset[index].to(device), self.labels[index].to(device)
        elif self.val:
            return torch.Tensor(self.dataset[index]).to(device), self.labels[index].to(device)
        else:
            return torch.Tensor(self.dataset[index].astype(float)).to(device)

    def __len__(self):
        return self.dataset.shape[0]

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

        self.linner1 = nn.Linear(4608, 512)
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
        loss = torch.nn.functional.cross_entropy(output,target.squeeze(dim=1))
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
        loss = torch.nn.functional.cross_entropy(output,target.squeeze(dim=1))
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
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 不进行计算图的构建，即没有grad_fn属性
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target.squeeze(dim=1).long(),
                                                           reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

def calculate_alpha_factor(worker_delay,alpha_type='power',alpha_adaptive=0.6,alpha_adaptive2=1):
    if alpha_type == 'power':
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

    LOG = 'dongtai.txt'
    EPOCHS = 2000
    LR = 0.01
    SEED = 336
    CLIENT_NUM = 100
    DIR = 'datasets'
    CLASSES = 10
    WEIGHT_DECAY = 0.0001
    RHO = 0.01
    ALPHA = 0.6
    MAX_DELAY = 4
    BATCH_SIZE = 10
    ITERATIONS = 1
    # 自适应参数类型 none power hige
    ALPHA_TYPE = 'power'
    ALPHA_ADAPTIVE = 0.6
    ALPHA_ADAPTIVE2 = 1
    INTERVAL = 10
    # TRAIN_NUM = 10
    GLOBAL_EPOCH = 2000

    log = LOG
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

    # # 实现
    # # 输出日志到文件
    # filehandler = logging.FileHandler(log)
    # # 输出日志到屏幕和日志文件
    # streamhandler = logging.StreamHandler()
    #
    # # 定义日志设置
    # logger = logging.getLogger('')
    # logger.setLevel(logging.INFO)
    # logger.addHandler(filehandler)
    # logger.addHandler(streamhandler)

    # 设置种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置路径
    data_dir = os.path.join('datasets', 'cifar10_dataset_split_{}'.format(client_num))
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 加载训练数据
    training_files = []
    for filename in sorted(listdir(train_dir)):
        absolute_filename = os.path.join(train_dir, filename)
        training_files.append(absolute_filename)


    # 定义读取数据函数
    def get_train_batch(train_filename):
        with open(train_filename, "rb") as f:
            B, L = pickle.load(f)
        return torch.tensor(B.transpose(0, 3, 1, 2)), torch.tensor(L)
        # return B.transpose(0, 3, 1, 2), L


    def get_val_train_batch(data_dir):
        test_filename = os.path.join(data_dir, 'train_data.pkl')
        with open(test_filename, "rb") as f:
            B, L = pickle.load(f)
        return torch.tensor(B.transpose(0, 3, 1, 2)), torch.tensor(L)


    def get_val_val_batch(data_dir):
        test_filename = os.path.join(data_dir, 'val_data.pkl')
        with open(test_filename, "rb") as f:
            B, L = pickle.load(f)
        return torch.tensor(B.transpose(0, 3, 1, 2)), torch.tensor(L)


    # 模型初始化
    model = Net(3, 10, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 热身训练
    # 加载热身训练数据
    [train_X, train_Y] = get_train_batch(training_files[0])
    train_set = CustomedDataSet(train_x=train_X, train_y=train_Y)
    train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    for i in range(5):
        warm_train(model, device, train_data, optimizer)

    # 保存模型
    params_prev = [param.clone() for param in model.parameters()]
    params_prev_list = [params_prev]
    server_ts = 0

    # 设置时间戳列表
    ts_list = [server_ts]
    # print(len())


    # 加载训练数据
    train_data_list = []
    for i in range(client_num):
        [train_X, train_Y] = get_train_batch(training_files[i])
        train_set = CustomedDataSet(train_x=train_X, train_y=train_Y)
        train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        train_data_list.append(train_data)

    # 加载测试数据
        # 计算损失函数
    [val_train_X, val_train_Y] = get_val_val_batch(val_dir)
    val_train_set = CustomedDataSet(test_x=val_train_X, test_y=val_train_Y, train=False, val=True)
    val_train_data = DataLoader(dataset=val_train_set, batch_size=1000, shuffle=False)

        # 计算准确率
    [val_val_X, val_val_Y] = get_val_val_batch(val_dir)
    val_val_set = CustomedDataSet(test_x=val_val_X, test_y=val_val_Y, train=False, val=True)
    val_val_data = DataLoader(dataset=val_val_set, batch_size=1000, shuffle=False)

    # 设置计时参数
    sum_delay = 0
    tic = time.time()

    alpha_factor_cache = []
    model_cache = []
    worker_delay_cache = []

    # 全局训练
    for epoch in range(global_epoch):

        train_num = min(math.ceil((epoch + 1) / 200), 10)

        for i in range(train_num):

            # 选择模型

            # print(ts_list)
            # print(len(params_prev_list))
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

            #     for param in model.parameters():
            #         print(param)

            # 记录选取的模型
            worker_ts = ts_list[model_idx]
            print("worker_ts")
            print(worker_ts)

            # 客户端训练
            train_data = random.choice(train_data_list)
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            for local_epoch in range(iterations):
                train(model, device, train_data, optimizer, rho, params_prev)

            # 计算模型延迟
            worker_delay = epoch - worker_ts
            sum_delay += worker_delay
            alpha_factor = calculate_alpha_factor(worker_delay, alpha_type)
            # alpha_scaled = alpha * alpha_factor

            # 模型存储至缓存
            params_prev = [param.clone() for param in model.parameters()]
            for param_prev in params_prev:
                param_prev[:] = param_prev * alpha_factor

            model_cache.append(params_prev)
            alpha_factor_cache.append(alpha_factor)
            worker_delay_cache.append(worker_delay)

            if len(model_cache) == train_num:
                print("模型聚合")
                print(epoch)
                print(alpha_factor_cache)
                print(worker_delay_cache)

                # worker_delay聚合
                sum_worker_delay = 0
                for worker_delay in worker_delay_cache:
                    sum_worker_delay += worker_delay
                avg_worker_delay = sum_worker_delay / train_num
                avg_alpha_factor = calculate_alpha_factor(avg_worker_delay)
                alpha_scaled = alpha * avg_alpha_factor

                # alpha_factor聚合
                sum_alpha_factor = 0
                for alpha_factor in alpha_factor_cache:
                    sum_alpha_factor += alpha_factor

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

        # 模型测试
        if epoch % interval == 0 or epoch == epochs - 1:
            # 测试集获取准确率
            test(model, device, val_val_data)
            # 训练集获取损失函数
            # test(model, device, val_train_data)
            tic = time.time()