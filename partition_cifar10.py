from torchvision import transforms,datasets
from fedlab.contrib.dataset import PartitionedCIFAR10

# 首次执行preprocess应设定为true
hetero = PartitionedCIFAR10(
    root='.',
    path='datasets/cifar10_unbalance_iid',
    dataname="cifar10",
    num_clients=100,
    download=False,
    preprocess=True,
    balance=False,
    partition="iid",
    unbalance_sgm=0.3,
    dir_alpha=0.3,
    seed=336,
    transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])]
    ),
    target_transform=transforms.ToTensor()
)

# test_dataload =  hetero.get_dataloader(1,4)
# for batch_idx, (data, target) in enumerate(test_dataload):
#     print(data)
#     print(data.shape)
#     print(target.shape)
#     break
