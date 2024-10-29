import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import MultipleLocator

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result_compression/resnet18_compression_base.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    c_05_acc = []
    c_05_loss = []
    c_04_acc = []
    c_04_loss = []
    c_03_acc = []
    c_03_loss = []
    c_02_acc = []
    c_02_loss = []
    c_01_acc = []
    c_01_loss = []
    c_10_acc = []
    c_10_loss = []
    for row in reader:
        c_05_acc.append(float(row[1]))
        c_05_loss.append(float(row[2]))
        c_04_acc.append(float(row[3]))
        c_04_loss.append(float(row[4]))
        c_03_acc.append(float(row[5]))
        c_03_loss.append(float(row[6]))
        c_02_acc.append(float(row[7]))
        c_02_loss.append(float(row[8]))
        c_01_acc.append(float(row[9]))
        c_01_loss.append(float(row[10]))
        c_10_acc.append(float(row[11]))
        c_10_loss.append(float(row[12]))


# 横轴坐标
x = [i+1 for i in range(20)]

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(x, c_05_acc,lw=1, ls='-', c='#13678A' ,label=r'$p=0.5$')
plt.plot(x, c_04_acc,lw=1, ls='--', c='#005E54' ,label=r'$p=0.4$')
plt.plot(x, c_03_acc,lw=1, ls='-.', c='#818274' ,label=r'$p=0.3$')
plt.plot(x, c_02_acc,lw=1, ls=':', c='#C2BB00' ,label=r'$p=0.2$')
plt.plot(x, c_01_acc,lw=1, ls='-', marker='o', markersize='1', c='#E1523D' ,label=r'$p=0.1$')
plt.plot(x, c_10_acc,lw=1, ls='--', marker='*', markersize='1', c='#000000' ,label=r'$p=1.0$')


plt.legend(fontsize=14,loc = 'lower right')
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
# plt.ylim(50,95)

x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(x, c_05_loss,lw=1, ls='-', c='#13678A' ,label=r'$p=0.5$')
plt.plot(x, c_04_loss,lw=1, ls='--', c='#005E54' ,label=r'$p=0.4$')
plt.plot(x, c_03_loss,lw=1, ls='-.', c='#818274' ,label=r'$p=0.3$')
plt.plot(x, c_02_loss,lw=1, ls=':', c='#C2BB00' ,label=r'$p=0.2$')
plt.plot(x, c_01_loss,lw=1, ls='-', marker='o', markersize='1', c='#E1523D' ,label=r'$p=0.1$')
plt.plot(x, c_10_loss,lw=1, ls='--', marker='*', markersize='1', c='#000000' ,label=r'$p=1.0$')

plt.legend(fontsize=14)
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


x_major_locator=MultipleLocator(2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

# plt.savefig('../pic/Fedavg_fmnist_ascend_uniform_descend', dpi=300)
# plt.savefig('../pic_zh/Resnet18_compression_base', dpi=300)
plt.show()
