import matplotlib.pyplot as plt
import csv

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result/FedAvg_fmnist_unbalance_noiid_uniform_nowarm.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    uniform_acc = []
    uniform_loss = []
    for row in reader:
        uniform_acc.append(float(row[1]))
        uniform_loss.append(float(row[2]))

with open("../result/FedAvg_fmnist_unbalance_noiid_ascend_nowarm.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    ascend_acc = []
    ascend_loss = []
    for row in reader:
        ascend_acc.append(float(row[1]))
        ascend_loss.append(float(row[2]))

with open("../result/FedAvg_fmnist_unbalance_noiid_descend_nowarm.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    descend_acc = []
    descend_loss = []
    for row in reader:
        descend_acc.append(float(row[1]))
        descend_loss.append(float(row[2]))

# 横轴坐标
x = [i*10 for i in range(31)]

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(x, uniform_acc,lw=1, ls='-', c='#005E54' ,label='Uniform')
plt.plot(x, ascend_acc,lw=1, ls='--', c='#C2BB00' ,label='Ascend')
plt.plot(x, descend_acc,lw=1, ls='-.', c='#E1523D' ,label='Descend')

plt.legend(fontsize=14)
# plt.xlabel('Number of Rounds',fontdict={'size':14})
# plt.ylabel('Accuracy',fontdict={'size': 14})
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
plt.ylim(50,95)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(x, uniform_loss,lw=1, ls='-', c='#005E54' ,label='Uniform')
plt.plot(x, ascend_loss,lw=1, ls='--', c='#C2BB00' ,label='Ascend')
plt.plot(x, descend_loss,lw=1, ls='-.', c='#E1523D' ,label='Descend')

plt.legend(fontsize=14)
# plt.xlabel('Number of Rounds',fontdict={'size':14})
# plt.ylabel('Loss',fontdict={'size': 14})
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


# plt.savefig('../pic/Fedavg_fmnist_ascend_uniform_descend', dpi=300)
# plt.savefig('../pic_zh/Fedavg_fmnist_ascend_uniform_descend', dpi=300)
plt.show()
