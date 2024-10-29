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
from torchvision import transforms, datasets
from fedlab.contrib.dataset import PartitionedCIFAR10
from torch.utils.data import Dataset, DataLoader
import time
import ipfshttpclient
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化IPFS客户端
ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')

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
        loss = torch.nn.functional.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()

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
        loss = torch.nn.functional.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 添加正则化
        if rho > 0:
            for param, param_prev in zip(model.named_parameters(), params_prev):
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
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    #print('\nEpoch : {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    logging.info('\nEpoch : {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    write.writerow({'EPOCH': epoch, 'ACC': 100. * correct / len(test_loader.dataset), 'LOSS': test_loss})

# def calculate_alpha_factor(worker_delay, alpha_type='power', alpha_adaptive=0.6, alpha_adaptive2=1):
#     if alpha_type == 'power':
#         if alpha_adaptive > 0:
#             alpha_factor = 1 / math.pow(worker_delay + 1.0, alpha_adaptive)
#         else:
#             alpha_factor = 1
#     elif alpha_type == 'exp':
#         alpha_factor = math.exp(-worker_delay * alpha_adaptive)
#     elif alpha_type == 'sigmoid':
#         # a soft-gated function in the range of (1/a, 1]
#         # maximum value
#         a = alpha_adaptive2
#         # slope
#         c = alpha_adaptive
#         b = math.exp(- c * worker_delay)
#         alpha_factor = (1 - b) / a + b
#     elif alpha_type == 'hinge':
#         if worker_delay <= alpha_adaptive2:
#             alpha_factor = 1
#         else:
#             alpha_factor = 1.0 / ((worker_delay - alpha_adaptive2) * alpha_adaptive + 1)
#             # alpha_factor = math.exp(- (worker_delay-args.alpha_adaptive2) * args.alpha_adaptive)
#     else:
#         alpha_factor = 1
#     return alpha_factor

def calculate_alpha_factor(worker_delay, alpha_type='power', alpha_adaptive=0.6, alpha_adaptive2=1):
    if alpha_type == 'power':
        alpha_factor = 1 / math.pow(worker_delay + 1.0, alpha_adaptive) if alpha_adaptive > 0 else 1
    elif alpha_type == 'exp':
        alpha_factor = math.exp(-worker_delay * alpha_adaptive)
    elif alpha_type == 'sigmoid':
        a = alpha_adaptive2
        c = alpha_adaptive
        b = math.exp(- c * worker_delay)
        alpha_factor = (1 - b) / a + b
    elif alpha_type == 'hinge':
        alpha_factor = 1 if worker_delay <= alpha_adaptive2 else 1.0 / ((worker_delay - alpha_adaptive2) * alpha_adaptive + 1)
    else:
        alpha_factor = 1
    return alpha_factor

def upload_model_to_ipfs(model, epoch, i):
    """
        上传模型到 IPFS 并将生成的哈希值写入文件。

        :param model: 要上传的模型
        :param epoch: 当前的 epoch
        :param i: 当前的迭代次数
        :return: 生成的哈希值和上传时间
        """
    # 将模型保存为临时文件
    temp_model_path = f'temp_model_{epoch}_{i}.pt'
    torch.save(model.state_dict(), temp_model_path)

    # 连接到 IPFS
    client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
    # 上传模型文件
    start_time = time.time()
    ipfs_hash = client.add(temp_model_path)['Hash']
    upload_time = time.time() - start_time

    # 删除临时文件
    os.remove(temp_model_path)

    # 将哈希值写入文件
    file_path = 'hashes.txt'
    with open(file_path, 'a') as file:
        file.write(f"{ipfs_hash} {epoch} {i}\n")

    return ipfs_hash, upload_time

def generate_hashes_file(hashes_data, file_path):
    """
    生成包含哈希值及其对应 epoch 和 i 的文件。

    :param hashes_data: 包含哈希值及其对应 epoch 和 i 的列表
    :param file_path: 输出文件的路径
    """
    with open(file_path, 'w') as file:
        for ipfs_hash, epoch, i in hashes_data:
            file.write(f"{ipfs_hash} {epoch} {i}\n")
def read_hashes_from_file(file_path='hashes.txt'):
    """
        从文件中读取哈希值及其对应的 epoch 和 i。

        :param file_path: 输入文件的路径，默认为 'hashes.txt'
        :return: 包含哈希值及其对应 epoch 和 i 的列表
        """
    hashes_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                ipfs_hash, epoch, i = parts
                hashes_data.append((ipfs_hash, int(epoch), int(i)))
            else:
                logging.warning(f"Invalid line format: {line.strip()}")
    return hashes_data

def download_model_from_ipfs(ipfs_hash, epoch, i, device):
    start_time = time.time()

    # 指定保存模型的文件夹路径
    result_folder = r'E:\论文2\DGCFL\resultipfs'

    # 构建绝对路径
    downloaded_model_path = os.path.join(result_folder, f'downloaded_model_{epoch}_{i}.pt')

    # 确保目录存在
    os.makedirs(result_folder, exist_ok=True)

    # 下载模型
    try:
        ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        ipfs_client.get(ipfs_hash, target=downloaded_model_path)
    except Exception as e:
        logging.error(f"Error downloading model from IPFS: {e}")
        return None, None

    # # 创建文件并设置权限
    # try:
    #     with open(downloaded_model_path, 'w') as f:
    #         pass
    #     os.chmod(downloaded_model_path, 0o666)
    # except PermissionError as e:
    #     logging.error(f"Permission error setting file permissions: {e}")
    #     return None, None
    # except Exception as e:
    #     logging.error(f"Error setting file permissions: {e}")
    #     return None, None

    # 加载模型
    try:
        downloaded_model = Net(3, 10, 10).to(device)
        downloaded_model.load_state_dict(torch.load(downloaded_model_path))
        logging.debug(f"Model loaded successfully from: {downloaded_model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

    end_time = time.time()
    download_time = end_time - start_time
    logging.debug(f"Model downloaded from IPFS with hash: {ipfs_hash}")
    logging.debug(f"Download time delay: {download_time:.4f} seconds")

    return downloaded_model, download_time


if __name__ == '__main__':
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.transforms as transforms

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化总下载时间
    total_download_time = 0
    csvfile = open('resultipfs/FedDgc_cifar10_unbalance_noiid_nowarm_IPFS.csv', mode='a', newline='')
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
    RHO = 0.01
    ALPHA = 0.8
    MAX_DELAY = 16
    BATCH_SIZE = 10
    ITERATIONS = 1
    # 自适应参数类型 none power hige
    ALPHA_TYPE = 'power'
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

    total_upload_time = 0
    total_download_time = 0

    # 全局训练
    for epoch in range(global_epoch):

        train_num = min(math.ceil((epoch + 1) / 100), 10)
        # 上升
        # train_num = math.ceil((epoch + 1) / 100)
        # 统一
        # train_num = 10
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
            logging.debug(f"worker_ts: {worker_ts}")
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

            # 上传模型到IPFS
            ipfs_hash, upload_time = upload_model_to_ipfs(model, epoch, i)
            total_upload_time += upload_time

            # 打印上传时间延迟
            print(f"Model uploaded to IPFS with hash: {ipfs_hash}")
            print(f"Upload time delay: {upload_time:.4f} seconds")

            # 读取 hashes.txt 文件
            # 下载模型并记录下载延迟
            downloaded_model, download_time = download_model_from_ipfs(ipfs_hash, epoch, i, device)
            if downloaded_model is not None:
                total_download_time += download_time

                # 打印下载时间延迟
                print(f"Model downloaded from IPFS with hash: {ipfs_hash}, Download time: {download_time:.2f} seconds")

                # 更新缓存
                params_prev_list.append([param.clone() for param in model.parameters()])
                ts_list.append(epoch)
                alpha_factor_cache.append(alpha_factor)
                model_cache.append(downloaded_model)
                worker_delay_cache.append(worker_delay)
                data_len_cache.append(data_len)

                logging.debug(
                    f"Epoch: {epoch}, Client: {client_id}, Worker Delay: {worker_delay}, Alpha Factor: {alpha_factor}, Data Length: {data_len}")

                # 聚合模型
                if (epoch + 1) % interval == 0:
                    aggregated_model = Net(3, 10, 10).to(device)
                    aggregated_params = [torch.zeros_like(param) for param in aggregated_model.parameters()]

                    total_data_len = sum(data_len_cache)
                    for model, data_len, alpha_factor in zip(model_cache, data_len_cache, alpha_factor_cache):
                        for aggregated_param, model_param in zip(aggregated_params, model.parameters()):
                            aggregated_param += (data_len / total_data_len) * alpha_factor * model_param

                    for aggregated_param, model_param in zip(aggregated_params, aggregated_model.parameters()):
                        model_param.data.copy_(aggregated_param)

                    # 更新全局模型
                    model.load_state_dict(aggregated_model.state_dict())

                    # 清空缓存
                    alpha_factor_cache.clear()
                    model_cache.clear()
                    worker_delay_cache.clear()
                    data_len_cache.clear()

                    # 测试聚合后的模型
                    test(model, device, test_set, epoch)

    # 生成哈希值文件（如果需要）
    hashes_data = read_hashes_from_file()
    generate_hashes_file(hashes_data, 'final_hashes.txt')

    # 关闭CSV文件
    csvfile.close()

    # 打印总上传和下载时间
    logging.info(f"Total Upload Time: {total_upload_time:.4f} seconds")
    logging.info(f"Total Download Time: {total_download_time:.4f} seconds")

    # 打印平均延迟
    avg_delay = sum_delay / (global_epoch * train_num)
    logging.info(f"Average Delay: {avg_delay:.4f} seconds")
    #         download_time = download_model_from_ipfs(ipfs_hash)
    #         total_download_time += download_time
    #
    #         # 打印下载时间延迟
    #         print(f"Model downloaded from IPFS with hash: {ipfs_hash}, Download time: {download_time:.2f} seconds")
    #
    #         # 更新缓存
    #         params_prev_list.append([param.clone() for param in model.parameters()])
    #         ts_list.append(epoch)
    #         alpha_factor_cache.append(alpha_factor)
    #         # model_cache.append(model)
    #         model_cache.append(downloaded_model)
    #         worker_delay_cache.append(worker_delay)
    #         data_len_cache.append(data_len)
    #
    #         logging.debug(
    #             f"Epoch: {epoch}, Client: {client_id}, Worker Delay: {worker_delay}, Alpha Factor: {alpha_factor}, Data Length: {data_len}")
    #
    #         # 聚合模型
    #         # 聚合模型
    #         if (epoch + 1) % interval == 0:
    #             aggregated_model = Net(3, 10, 10).to(device)
    #             aggregated_params = [torch.zeros_like(param) for param in aggregated_model.parameters()]
    #
    #             total_data_len = sum(data_len_cache)
    #             for model, data_len, alpha_factor in zip(model_cache, data_len_cache, alpha_factor_cache):
    #                 for aggregated_param, model_param in zip(aggregated_params, model.parameters()):
    #                     aggregated_param += (data_len / total_data_len) * alpha_factor * model_param
    #
    #             for aggregated_param, model_param in zip(aggregated_params, aggregated_model.parameters()):
    #                 model_param.data.copy_(aggregated_param)
    #
    #             # 更新全局模型
    #             model.load_state_dict(aggregated_model.state_dict())
    #
    #             # 清空缓存
    #             alpha_factor_cache.clear()
    #             model_cache.clear()
    #             worker_delay_cache.clear()
    #             data_len_cache.clear()
    #
    #             # 测试聚合后的模型
    #             test(model, device, test_set, epoch)
    #
    # # 生成哈希值文件（如果需要）
    # hashes_data = read_hashes_from_file()
    # generate_hashes_file(hashes_data, 'final_hashes.txt')
    #
    #     # 关闭CSV文件
    # csvfile.close()
    #
    # # 打印总上传和下载时间
    # logging.info(f"Total Upload Time: {total_upload_time:.4f} seconds")
    # logging.info(f"Total Download Time: {total_download_time:.4f} seconds")
    #
    # # 打印平均延迟
    # avg_delay = sum_delay / (global_epoch * train_num)
    # logging.info(f"Average Delay: {avg_delay:.4f} seconds")
