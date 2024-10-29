import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import MultipleLocator

plt.rcParams["font.sans-serif"] = ["SimHei"]

# 获取文件某一列数据
with open("../result_compression/resnet50_compression_k357.csv") as data:
    reader = csv.reader(data)
    header = next(reader)
    n_3_acc = []
    n_3_loss = []
    n_5_acc = []
    n_5_loss = []
    n_7_acc = []
    n_7_loss = []
    for row in reader:
        n_3_acc.append(float(row[1]))
        n_3_loss.append(float(row[2]))
        n_5_acc.append(float(row[3]))
        n_5_loss.append(float(row[4]))
        n_7_acc.append(float(row[5]))
        n_7_loss.append(float(row[6]))


# 横轴坐标
x = [i+1 for i in range(50)]

# 控制画布大小
fig = plt.figure(figsize=(12, 5))

# 绘制第一张子图
plt.subplot(1, 2, 1)

plt.plot(x, n_3_acc,lw=1, ls='-', c='#13678A' ,label=r'$k=3$')
plt.plot(x, n_5_acc,lw=1, ls='--', c='#005E54' ,label=r'$k=5$')
plt.plot(x, n_7_acc,lw=1, ls='-.', c='#818274' ,label=r'$k=7$')


plt.legend(fontsize=14,loc = 'lower right')
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('准确率',fontdict={'size': 14})
# plt.grid(True)
# plt.ylim(50,95)

x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)


# 绘制第二张子图
plt.subplot(1, 2, 2)

plt.plot(x, n_3_loss,lw=1, ls='-', c='#13678A' ,label=r'$k=3$')
plt.plot(x, n_5_loss,lw=1, ls='--', c='#005E54' ,label=r'$k=5$')
plt.plot(x, n_7_loss,lw=1, ls='-.', c='#818274' ,label=r'$k=7$')

plt.legend(fontsize=14)
plt.xlabel('训练轮次',fontdict={'size':14})
plt.ylabel('损失',fontdict={'size': 14})
# plt.grid(True)


x_major_locator=MultipleLocator(10)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

# plt.savefig('../pic/Fedavg_fmnist_ascend_uniform_descend', dpi=300)
# plt.savefig('../pic_zh/Resnet50_compression_k357', dpi=300)
plt.show()
