import matplotlib.pyplot as plt
import csv
from calculate_time import fedDgc_time,fedTed_time,fedAsync_time,fedAvg_time


plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100_0.2.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_2 = []
    fedDgc_loss_2 = []
    for row in reader:
        fedDgc_acc_2.append(float(row[1]))
        fedDgc_loss_2.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100_0.4.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_4 = []
    fedDgc_loss_4 = []
    for row in reader:
        fedDgc_acc_4.append(float(row[1]))
        fedDgc_loss_4.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_6 = []
    fedDgc_loss_6 = []
    for row in reader:
        fedDgc_acc_6.append(float(row[1]))
        fedDgc_loss_6.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100_0.8.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_8 = []
    fedDgc_loss_8 = []
    for row in reader:
        fedDgc_acc_8.append(float(row[1]))
        fedDgc_loss_8.append(float(row[2]))
        if int(row[0]) == 1000:
            break

# 横轴坐标
epoch = 1000
mu, sigma, agg = 10,5,2

fedDgc_x = fedDgc_time(mu,sigma,agg,epoch)

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(fedDgc_x, fedDgc_acc_2,lw=1, ls='-', c='#13678A' ,label=r'$\alpha$ = 0.2')
plt.plot(fedDgc_x, fedDgc_acc_4,lw=1, ls='--', c='#005E54' ,label=r'$\alpha$ = 0.4')
plt.plot(fedDgc_x, fedDgc_acc_6,lw=1, ls=':', c='#C2BB00' ,label=r'$\alpha$ = 0.6')
plt.plot(fedDgc_x, fedDgc_acc_8,lw=1, ls='-.', c='#E1523D' ,label=r'$\alpha$ = 0.8')

plt.legend(fontsize=14,loc = 'lower right')
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Accuracy',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
plt.ylim(70,95)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(fedDgc_x, fedDgc_loss_2,lw=1, ls='-', c='#13678A' ,label=r'$\alpha$ = 0.2')
plt.plot(fedDgc_x, fedDgc_loss_4,lw=1, ls='--', c='#005E54' ,label=r'$\alpha$ = 0.4')
plt.plot(fedDgc_x, fedDgc_loss_6,lw=1, ls=':', c='#C2BB00' ,label=r'$\alpha$ = 0.6')
plt.plot(fedDgc_x, fedDgc_loss_8,lw=1, ls='-.', c='#E1523D' ,label=r'$\alpha$ = 0.8')

plt.legend(fontsize=14)
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Loss',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


# plt.savefig('../pic/fmnist_alpha', dpi=300)
# plt.savefig('../pic_zh/fmnist_alpha', dpi=300)
plt.show()
