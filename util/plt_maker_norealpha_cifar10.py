import matplotlib.pyplot as plt
import csv
from calculate_time import fedDgc_time,fedTed_time,fedAsync_time,fedAvg_time

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result/FedDgc_cifar10_unbalance_noiid_nowarm_100_0.8_16.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_2 = []
    fedDgc_loss_2 = []
    for row in reader:
        fedDgc_acc_2.append(float(row[1]))
        fedDgc_loss_2.append(float(row[2]))

with open("../result/FedDgc_cifar10_unbalance_noiid_nowarm_100_norealpha_0.8_16.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_4 = []
    fedDgc_loss_4 = []
    for row in reader:
        fedDgc_acc_4.append(float(row[1]))
        fedDgc_loss_4.append(float(row[2]))

# 横轴坐标
epoch = 2000
mu, sigma, agg = 16,8,3

fedDgc_x = fedDgc_time(mu,sigma,agg,epoch)

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(fedDgc_x, fedDgc_acc_2,lw=1, ls='-', c='#13678A' ,label='baseline')
plt.plot(fedDgc_x, fedDgc_acc_4,lw=1, ls='--', c='#005E54' ,label='norealpha')

plt.legend(fontsize=14,loc = 'lower right')
plt.xlabel('Time(s)',fontdict={'size':14})
plt.ylabel('Accuracy',fontdict={'size': 14})
# plt.grid(True)
plt.ylim(50,85)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(fedDgc_x, fedDgc_loss_2,lw=1, ls='-', c='#13678A' ,label=r'$\alpha$ = 0.2')
plt.plot(fedDgc_x, fedDgc_loss_4,lw=1, ls='--', c='#005E54' ,label=r'$\alpha$ = 0.4')


plt.legend(fontsize=14)
plt.xlabel('Time(s)',fontdict={'size':14})
plt.ylabel('Loss',fontdict={'size': 14})
# plt.grid(True)


# plt.savefig('../pic/cifar10_alpha', dpi=300)
plt.show()
