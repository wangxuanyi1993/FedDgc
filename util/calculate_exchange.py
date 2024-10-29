import heapq
import numpy as np
import math
import random

epoch = 450
trainnum = 250
exchange_num = []

for i in range(epoch):
    train_num = min(math.ceil((i + 1) / trainnum), 10)
    exchange_num.append(train_num)
print(sum(exchange_num))