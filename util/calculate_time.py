import heapq
import numpy as np
import math
import random


# def fedDgc_time(mu,sigma,agg,epoch,trainnum):
def fedDgc_time(mu,sigma,agg,epoch):
    mu, sigma = mu, sigma
    aggregation_time = agg
    np.random.seed(20)
    work_time = list(np.random.normal(mu, sigma, 100))
    work_time = [abs(i) for i in work_time]
    interval_time = []
    fedDgc_x = [0]
    remain_time = [i for i in work_time]
    staleness = [0] * 100

    for i in range(epoch):

        update_num = 0
        # train_num = min(math.ceil((i + 1) / trainnum), 10)
        train_num = min(math.ceil((i + 1) / 100), 5)

        interval = max(heapq.nsmallest(train_num, remain_time))
        interval_time.append(interval + aggregation_time)
        fedDgc_x.append(sum(interval_time))
        for i in range(len(remain_time)):
            if staleness[i] < 4:
                remain_time[i] = remain_time[i] - interval
                if remain_time[i] <= 0 and update_num < train_num:
                    remain_time[i] = work_time[i] + aggregation_time
                    staleness[i] = 0
                    update_num += 1
                remain_time[i] = remain_time[i] - aggregation_time
                if remain_time[i] < 0:
                    remain_time[i] = 0
                    staleness[i] += 1
            else:
                remain_time[i] = work_time[i]
                staleness[i] = 0

    fedDgc_x = fedDgc_x[::10]
    return fedDgc_x

def fedTed_time(mu,sigma,agg,epoch):

    mu, sigma = mu, sigma
    aggregation_time = agg
    np.random.seed(20)
    work_time = list(np.random.normal(mu, sigma, 100))
    work_time = [abs(i) for i in work_time]
    interval_time = []
    fedTed_x = [0]
    remain_time = [i for i in work_time]
    staleness = [0] * 100

    for i in range(epoch):

        update_num = 0
        train_num = 10

        interval = max(heapq.nsmallest(train_num, remain_time))
        interval_time.append(interval + aggregation_time)
        fedTed_x.append(sum(interval_time))
        for i in range(len(remain_time)):
            if staleness[i] < 4:
                remain_time[i] = remain_time[i] - interval
                if remain_time[i] <= 0 and update_num < train_num:
                    remain_time[i] = work_time[i] + aggregation_time
                    staleness[i] = 0
                    update_num += 1
                remain_time[i] = remain_time[i] - aggregation_time
                if remain_time[i] < 0:
                    remain_time[i] = 0
                    staleness[i] += 1
            else:
                remain_time[i] = work_time[i]
                staleness[i] = 0

    fedTed_x = fedTed_x[::10]
    return fedTed_x

def fedAsync_time(mu, sigma, agg, epoch):

    mu, sigma = mu, sigma
    aggregation_time = agg
    np.random.seed(20)
    work_time = list(np.random.normal(mu, sigma, 100))
    work_time = [abs(i) for i in work_time]
    interval_time = []
    fedAsync_x = [0]
    remain_time = [i for i in work_time]
    staleness = [0] * 100

    for i in range(epoch):

        update_num = 0
        train_num = 1

        interval = max(heapq.nsmallest(train_num, remain_time))
        interval_time.append(interval + aggregation_time)
        fedAsync_x.append(sum(interval_time))
        for i in range(len(remain_time)):
            if staleness[i] < 4:
                remain_time[i] = remain_time[i] - interval
                if remain_time[i] <= 0 and update_num < train_num:
                    remain_time[i] = work_time[i] + aggregation_time
                    staleness[i] = 0
                    update_num += 1
                remain_time[i] = remain_time[i] - aggregation_time
                if remain_time[i] < 0:
                    remain_time[i] = 0
                    staleness[i] += 1
            else:
                remain_time[i] = work_time[i]
                staleness[i] = 0

    fedAsync_x = fedAsync_x[::10]
    return fedAsync_x

def fedAvg_time(mu, sigma, agg,fedavg_epoch):

    mu, sigma = mu, sigma
    aggregation_time = agg
    np.random.seed(20)
    work_time = list(np.random.normal(mu, sigma, 100))
    work_time = [abs(i) for i in work_time]
    interval_time = []
    fedAvg_x = [0]

    for i in range(fedavg_epoch):
        worker = random.sample(work_time, 10)
        interval = max(worker)
        interval_time.append(interval + aggregation_time)
        fedAvg_x.append(sum(interval_time))

    fedAvg_x = fedAvg_x[::10]
    return fedAvg_x

def calculate_fedDgc_time(mu,sigma,agg,epoch,trainnum):
    mu, sigma = mu, sigma
    aggregation_time = agg
    np.random.seed(20)
    work_time = list(np.random.normal(mu, sigma, 100))
    work_time = [abs(i) for i in work_time]
    interval_time = []
    fedDgc_x = [0]
    remain_time = [i for i in work_time]
    staleness = [0] * 100

    for i in range(epoch):

        update_num = 0
        train_num = min(math.ceil((i + 1) / trainnum), 10)

        interval = max(heapq.nsmallest(train_num, remain_time))
        interval_time.append(interval + aggregation_time)
        fedDgc_x.append(sum(interval_time))
        for i in range(len(remain_time)):
            if staleness[i] < 4:
                remain_time[i] = remain_time[i] - interval
                if remain_time[i] <= 0 and update_num < train_num:
                    remain_time[i] = work_time[i] + aggregation_time
                    staleness[i] = 0
                    update_num += 1
                remain_time[i] = remain_time[i] - aggregation_time
                if remain_time[i] < 0:
                    remain_time[i] = 0
                    staleness[i] += 1
            else:
                remain_time[i] = work_time[i]
                staleness[i] = 0

    fedDgc_x = fedDgc_x[::10]
    return fedDgc_x


epoch = 600
# mu, sigma, agg = 10,5,2
mu, sigma, agg = 16,8,3
# fedDgc_x = calculate_fedDgc_time(mu,sigma,agg,450,250)
fedDgc_x = fedTed_time(mu,sigma,agg,epoch)
print(fedDgc_x[-1])