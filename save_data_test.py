# 导包
import csv

# 创建或打开文件
csvfile = open('成绩.csv', mode='w', newline='')
# 标题列表
fieldnames = ['ACC', 'LOSS']
# 创建 DictWriter 对象
write = csv.DictWriter(csvfile, fieldnames=fieldnames)
# 写入表头
write.writeheader()
# 写入数据
for i in range(5):
    acc = i
    loss = i*10
    write.writerow({'ACC': acc, 'LOSS': loss})
for i in range(5):
    acc = i
    loss = i*10
    write.writerow({'ACC': acc, 'LOSS': loss})
# write.writerow({'语文': 80, '高数': 90, '英语': 20, '爬虫': 98, 'python': 89})
# write.writerow({'语文': 84, '高数': 80, '英语': 60, '爬虫': 79, 'python': 91})
# write.writerow({'语文': 70, '高数': 91, '英语': 76, '爬虫': 89, 'python': 72})
# write.writerow({'语文': 89, '高数': 96, '英语': 85, '爬虫': 91, 'python': 90})
# write.writerow({'语文': 60, '高数': 86, '英语': 81, '爬虫': 86, 'python': 81})
# write.writerow({'语文': 77, '高数': 88, '英语': 77, '爬虫': 76, 'python': 79})
