import matplotlib.pyplot as plt
import csv
from calculate_time import fedDgc_time,fedTed_time,fedAsync_time,fedAvg_time

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_50.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_50 = []
    fedDgc_loss_50 = []
    for row in reader:
        fedDgc_acc_50.append(float(row[1]))
        fedDgc_loss_50.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_100.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_100 = []
    fedDgc_loss_100 = []
    for row in reader:
        fedDgc_acc_100.append(float(row[1]))
        fedDgc_loss_100.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_150.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_150 = []
    fedDgc_loss_150 = []
    for row in reader:
        fedDgc_acc_150.append(float(row[1]))
        fedDgc_loss_150.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_200.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_200 = []
    fedDgc_loss_200 = []
    for row in reader:
        fedDgc_acc_200.append(float(row[1]))
        fedDgc_loss_200.append(float(row[2]))
        if int(row[0]) == 1000:
            break

with open("../result/FedDgc_fmnist_unbalance_noiid_nowarm_300.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    fedDgc_acc_300 = []
    fedDgc_loss_300 = []
    for row in reader:
        fedDgc_acc_300.append(float(row[1]))
        fedDgc_loss_300.append(float(row[2]))
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

plt.plot(fedDgc_x, fedDgc_acc_50,lw=1, ls='-', c='#13678A' ,label=r'$\nu$ = 50')
plt.plot(fedDgc_x, fedDgc_acc_100,lw=1, ls='--', c='#005E54' ,label=r'$\nu$ = 100')
plt.plot(fedDgc_x, fedDgc_acc_150,lw=1, ls='--', c='#818274' ,label=r'$\nu$ = 150')
plt.plot(fedDgc_x, fedDgc_acc_200,lw=1, ls=':', c='#C2BB00' ,label=r'$\nu$ = 200')
plt.plot(fedDgc_x, fedDgc_acc_300,lw=1, ls='-.', c='#E1523D' ,label=r'$\nu$ = 300')

plt.legend(fontsize=14,loc = 'lower right')
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Accuracy',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
plt.ylim(75,95)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(fedDgc_x, fedDgc_loss_50,lw=1, ls='-', c='#13678A' ,label=r'$\nu$ = 50')
plt.plot(fedDgc_x, fedDgc_loss_100,lw=1, ls='--', c='#005E54' ,label=r'$\nu$ = 100')
plt.plot(fedDgc_x, fedDgc_loss_150,lw=1, ls='--', c='#818274' ,label=r'$\nu$ = 150')
plt.plot(fedDgc_x, fedDgc_loss_200,lw=1, ls=':', c='#C2BB00' ,label=r'$\nu$ = 200')
plt.plot(fedDgc_x, fedDgc_loss_300,lw=1, ls='-.', c='#E1523D' ,label=r'$\nu$ = 300')

plt.legend(fontsize=14)
# plt.xlabel('Time(s)',fontdict={'size':14})
# plt.ylabel('Loss',fontdict={'size': 14})
plt.xlabel('时间',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


# plt.savefig('../pic/fmnist_speed_add150', dpi=300)
# plt.savefig('../pic_zh/fmnist_speed_add150', dpi=300)
plt.show()