import matplotlib.pyplot as plt
import csv
import math

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../Invalid_results/FedAvg_fmnist_unbalance_noiid_ascend_nowarm_336.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    ascend_acc_336 = []
    ascend_loss_336 = []
    for row in reader:
        ascend_acc_336.append(float(row[1]))
        ascend_loss_336.append(float(row[2]))

with open("../Invalid_results/FedAvg_fmnist_unbalance_noiid_ascend_nowarm_337.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    ascend_acc_337 = []
    ascend_loss_337 = []
    for row in reader:
        ascend_acc_337.append(float(row[1]))
        ascend_loss_337.append(float(row[2]))

with open("../Invalid_results/FedAvg_fmnist_unbalance_noiid_ascend_nowarm_338.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    ascend_acc_338 = []
    ascend_loss_338 = []
    for row in reader:
        ascend_acc_338.append(float(row[1]))
        ascend_loss_338.append(float(row[2]))

# 横轴坐标
x_m = [i*10 for i in range(31)]

ascend_loss_mean = []
ascend_loss_max = []
ascend_loss_min = []
for x,y,z in zip(ascend_loss_336,ascend_loss_337,ascend_loss_338):
    loss_mean = (x+y+z)/3
    loss_max = max(x,y,z)
    loss_min = min(x,y,z)
    ascend_loss_mean.append(loss_mean)
    ascend_loss_max.append(loss_max)
    ascend_loss_min.append(loss_min)

ascend_acc_mean = []
ascend_acc_max = []
ascend_acc_min = []
for x,y,z in zip(ascend_acc_336,ascend_acc_337,ascend_acc_338):
    acc_mean = (x+y+z)/3
    acc_max = max(x,y,z)
    acc_min = min(x,y,z)
    ascend_acc_mean.append(acc_mean)
    ascend_acc_max.append(acc_max)
    ascend_acc_min.append(acc_min)

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(x_m, ascend_loss_mean,lw=1, ls='-', c='#005E54' ,label='Uniform')
plt.fill_between(x_m, ascend_loss_min,ascend_loss_max , color='blue', alpha=0.3)
# plt.plot(x, ascend_acc,lw=1, ls=':', c='#C2BB00' ,label='Ascend')
# plt.plot(x, descend_acc,lw=1, ls='-.', c='#E1523D' ,label='Descend')

# plt.legend(fontsize=14,loc = 'upper left')
# plt.xlabel('Number of Rounds',fontdict={'size':14})
# plt.ylabel('Accuracy',fontdict={'size': 14})
# plt.grid(True)
# plt.ylim(50,95)


# # 绘制第二张子图
# plt.subplot(1, 2, 2)
#
# plt.plot(x, uniform_loss,lw=1, ls='-', c='#005E54' ,label='Uniform')
# plt.plot(x, ascend_loss,lw=1, ls='--', c='#C2BB00' ,label='Ascend')
# plt.plot(x, descend_loss,lw=1, ls='-.', c='#E1523D' ,label='Descend')
#
# plt.legend(fontsize=14)
# plt.xlabel('Number of Rounds',fontdict={'size':14})
# plt.ylabel('Loss',fontdict={'size': 14})
# plt.grid(True)
#
#
# # plt.savefig('../pic/Fedavg_cifar10_ascend_uniform_descend', dpi=300)
plt.show()
