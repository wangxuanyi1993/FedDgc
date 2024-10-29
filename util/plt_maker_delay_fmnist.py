import matplotlib.pyplot as plt
import csv
from calculate_time import fedDgc_time,fedTed_time,fedAsync_time,fedAvg_time

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result/FedAsync_fmnist_unbalance_noiid_nowarm_16_338.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedAsync_acc = []
    fedAsync_loss = []
    for row in reader:
        fedAsync_acc.append(float(row[1]))
        fedAsync_loss.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedTed_fmnist_unbalance_noiid_nowarm_16.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedTed_acc = []
    fedTed_loss = []
    for row in reader:
        fedTed_acc.append(float(row[1]))
        fedTed_loss.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100_16.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc = []
    fedDgc_loss = []
    for row in reader:
        fedDgc_acc.append(float(row[1]))
        fedDgc_loss.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedSAFA_fmnist_unbalance_noiid_nowarm_1_norho_16.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedSAFA_acc = []
    fedSAFA_loss = []
    for row in reader:
        fedSAFA_acc.append(float(row[1]))
        fedSAFA_loss.append(float(row[2]))
        if int(row[0]) == 1000:
            break

# 横轴坐标
epoch = 1000
mu, sigma, agg = 10,5,2

fedDgc_x = fedDgc_time(mu,sigma,agg,epoch)
fedTed_x = fedTed_time(mu,sigma,agg,epoch)
fedAsync_x = fedAsync_time(mu,sigma,agg,epoch)
fedSAFA_x = fedTed_time(mu,sigma,agg,epoch)

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(fedAsync_x, fedAsync_acc,lw=1, ls='-', c='#005E54' ,label='FedAsync')
plt.plot(fedTed_x, fedTed_acc,lw=1, ls=':', c='#C2BB00' ,label='FedTed')
plt.plot(fedDgc_x, fedDgc_acc,lw=1, ls='-.', c='#E1523D' ,label='FedDgc')
# plt.plot(fedSAFA_x, fedSAFA_acc,lw=1, ls='-.', c='#818274' ,label='SAFA')

plt.legend(fontsize=14,loc = 'lower right')
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Accuracy',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
plt.ylim(70,95)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(fedAsync_x, fedAsync_loss,lw=1, ls='-', c='#005E54' ,label='FedAsync')
plt.plot(fedTed_x, fedTed_loss,lw=1, ls=':', c='#C2BB00' ,label='FedTed')
plt.plot(fedDgc_x, fedDgc_loss,lw=1, ls='-.', c='#E1523D' ,label='FedDgc')
# plt.plot(fedSAFA_x, fedSAFA_loss,lw=1, ls='-.', c='#818274' ,label='SAFA')

plt.legend(fontsize=14)
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Loss',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


# plt.savefig('../pic/fmnist_delay_16_addSAFA', dpi=300)
# plt.savefig('../pic_zh/fmnist_delay_16', dpi=300)
plt.show()
