from torchvision import transforms,datasets
from torch.utils.data import Dataset, DataLoader

from fedlab.contrib.dataset import PartitionedFMNIST
from fedlab.contrib.dataset import PartitionedMNIST
from fedlab.contrib.dataset import femnist
from fedlab.contrib.dataset import PartitionedCIFAR10
from fedlab.utils.dataset import FMNISTPartitioner

# partition="noniid-labeldir"
# 首次执行preprocess应设定为true
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
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        # transforms.Normalize(mean = [0.2860], std = [0.3530]),
        transforms.ToTensor()]
    ),
    target_transform=transforms.ToTensor()
)

# test_dataset1 = hetero.get_dataset(cid=1,type="train")
# print(len(test_dataset1))
# print(test_dataset1[0])
test_dataload =  hetero.get_dataloader(1,4)
# test_set = DataLoader(test_dataset1, batch_size=20, shuffle=False)
for batch_idx, (data, target) in enumerate(test_dataload):
    print(data.shape)
    print(data[0])
    print(target.shape)
    break
# test_dataset2 = hetero.get_dataset(cid=2,type="train")
# print(len(test_dataset2))

# for i in range(10):
#     test_dataset = hetero.get_dataset(cid=i, type="train")
#     print('Client {} Num is {}'.format(i,len(test_dataset)))