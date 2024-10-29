# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

if len(model_cache) == train_num:
    print("模型聚合")
    print(epoch)
    print(alpha_factor_cache)
    print(worker_delay_cache)
    print(data_len_cache)

    # worker_delay聚合
    sum_worker_delay = 0
    for worker_delay in worker_delay_cache:
        sum_worker_delay += worker_delay
    avg_worker_delay = sum_worker_delay / train_num
    avg_alpha_factor = calculate_alpha_factor(avg_worker_delay)
    alpha_scaled = alpha * avg_alpha_factor

    # alpha_factor聚合
    sum_alpha_factor = 0
    for alpha_factor in alpha_factor_cache:
        sum_alpha_factor += alpha_factor

    # data_len聚合
    sum_data_len = 0
    for data_len in data_len_cache:
        sum_data_len += data_len

    # 模型聚合
    agg_model = [torch.zeros_like(param_prev) for param_prev in params_prev]
    # agg_model = []
    # for param_prev in model_cache[0]:
    #     x = torch.zeros_like(param_prev)
    #     agg_model.append(x)
    # print('--------------------------------')
    # print('生成')
    # print(agg_model[0][0][0][0])

    for local_model in model_cache:
        for x, y in zip(agg_model, local_model):
            x += y
    # for local_model,data_len in zip(model_cache,data_len_cache):
    #     for x, y in zip(agg_model, local_model):
    #         x += y * (data_len/sum_data_len)

    # print('--------------------------------')
    # print('聚合')
    # print(agg_model[0][0][0][0])
    for layer in agg_model:
        layer[:] = layer / sum_alpha_factor
    # print('--------------------------------')
    # print('平均')
    # print(agg_model[0][0][0][0])

    for param, param_server in zip(agg_model, params_prev_list[-1]):
        param[:] = param_server * (1 - alpha_scaled) + param * alpha_scaled
    # print('--------------------------------')
    # print('加和')
    # print(agg_model[0][0][0][0])

    # push
    params_prev_list.append(agg_model)
    ts_list.append(epoch + 1)
    # pop
    if len(params_prev_list) > max_delay:
        del params_prev_list[0]
        del ts_list[0]
    # 清除缓存
    alpha_factor_cache.clear()
    model_cache.clear()
    worker_delay_cache.clear()
    data_len_cache.clear()