import pandas as pd
import numpy as np
import datetime


def get_w(delta_f, delta_b, current_tradeday, frequency=30):
    all_tradeday = delta_f.index.tolist()
    w = -(delta_b / delta_f)
    i_date = all_tradeday.index(current_tradeday)
    w_30 = w[max(i_date - frequency, 0):i_date]  # 过去30天的数据
    w_30 = w_30.dropna()
    if len(w_30) < 5:
        return 0
    w_30 = sigmoid(w_30)  # 去极值
    w_30 = hp_filter(w_30)  # HP滤波
    hedge_w = w_30[-1]
    # 对w限定取值范围[-0.5,0.5]
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w


def get_w_real(delta_f, delta_b, current_tradeday, frequency=15):
    all_tradeday = delta_f.index.tolist()
    delta_b_roll = delta_b[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    delta_f_roll = delta_f[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    w = -(delta_b_roll / delta_f_roll)
    if np.isnan(w):
        return 0
    hedge_w = w
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w


def get_portfolio_w(asset0, asset1, asset2, current_tradeday):
    all_tradeday = asset1.index.tolist()
    timestamp = datetime.datetime.strptime(current_tradeday, '%Y/%m/%d')
    i_date = all_tradeday.index(timestamp)
    asset0_past = asset0[max(i_date - 30, 0):i_date]
    asset1_past = asset1[max(i_date - 30, 0):i_date]
    asset2_past = asset2[max(i_date - 30, 0):i_date]
    if len(asset1_past) < 30:
        w1 = 0.5
        w2 = 0.5
        return w1, w2
    asset1_mean = np.mean(asset1_past)
    asset2_mean = np.mean(asset2_past)
    deviation_1 = asset1_past - asset1_mean
    deviation_2 = asset2_past - asset2_mean
    asset1_std = (np.sqrt(deviation_1[deviation_1 < 0] ** 2).sum()) / len(deviation_1[deviation_1 < 0])
    asset2_std = (np.sqrt(deviation_2[deviation_2 < 0] ** 2).sum()) / len(deviation_2[deviation_2 < 0])
    w1 = asset2_std / (asset1_std + asset2_std)
    w2 = asset1_std / (asset1_std + asset2_std)
    return w1, w2


def get_w2(delta_f, delta_b, current_tradeday, frequency=30):
    all_tradeday = delta_f.index.tolist()
    delta_f_mean = delta_f.rolling(30).mean()
    delta_b_mean = delta_b.rolling(30).mean()
    bf = (delta_f * delta_b).rolling(30).mean()
    cov = bf - delta_b_mean * delta_f_mean
    var = delta_f.rolling(30).var()
    w = cov / var
    w_30 = w[all_tradeday.index(current_tradeday) - frequency:all_tradeday.index(current_tradeday)]  # 取过去三十天的数据
    w_30 = w_30.dropna()
    if len(w_30) < 5:
        return 0
    w_30 = hp_filter(w_30)  # HP滤波
    hedge_w = w_30[-1]
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w


def get_w2_real(delta_f, delta_b, current_tradeday, frequency=30):
    all_tradeday = delta_f.index.tolist()
    delta_f_mean = delta_f.rolling(30).mean()
    delta_b_mean = delta_b.rolling(30).mean()
    bf = (delta_f * delta_b).rolling(30).mean()
    cov = bf - delta_b_mean * delta_f_mean
    var = delta_f.rolling(30).var()
    w = cov / var
    w_30 = w[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency]
    if len(w_30) < 15:
        return 0
    w_30 = hp_filter(w_30)
    hedge_w = w_30[-1]
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w


def hp_filter(series, alpha=14400):
    n = len(series)
    d = np.zeros((n - 2, n))
    unit_mat = np.diag(np.ones(n))
    for i in range(n - 2):
        d[i, i: i + 3] = np.array([1, -2, 1])
    alt_series = np.dot(np.linalg.inv(unit_mat + alpha * np.dot(d.T, d)), series)
    return pd.Series(alt_series, index=series.index)


def hp_filter2(series, alpha=14400):
    n = len(series)
    d = np.zeros((n - 2, n))
    unit_mat = np.diag(np.ones(n))
    for i in range(n - 2):
        d[i, i: i + 3] = np.array([1, -2, 1])
    alt_series = np.dot(np.linalg.inv(unit_mat + alpha * np.dot(d.T, d)), series)
    return alt_series[-1]


def sigmoid(df, n=1.4826):
    df2 = df.copy()
    median = df.quantile(0.5)
    new_median = ((df - median).abs()).quantile(0.50)
    max_range = median + 3 * n * new_median
    min_range = median - 3 * n * new_median
    df2[df2 > max_range] = n * median / (1 + np.exp(df2[df2 > max_range])) + 3 * n * new_median
    df2[df2 < min_range] = -(n * median / (1 + np.exp(df2[df2 < min_range])) + 3 * n * new_median)
    return df2


def get_amend(classes, pred, w1, copy_type):
    if copy_type == 1:
        if classes == 'two':
            if pred > 0:
                w = min(w1 + 0.1, 0.5)
            elif pred < 0:
                w = max(w1 - 0.1, -0.5)
        elif classes == 'four':
            if pred == -2:
                w = max(w1 - 0.2, -0.5)
            elif pred == -1:
                w = max(w1 - 0.075, -0.5)
            elif pred == 1:
                w = min(w1 + 0.075, 0.5)
            elif pred == 2:
                w = min(w1 + 0.2, 0.5)
        elif classes == 'six':
            if pred == -3:
                w = max(w1 - 0.2, -0.5)
            elif pred == -2:
                w = max(w1 - 0.1, -0.5)
            elif pred == -1:
                w = max(w1 - 0.05, -0.5)
            elif pred == 1:
                w = min(w1 + 0.05, 0.5)
            elif pred == 2:
                w = min(w1 + 0.1, 0.5)
            elif pred == 3:
                w = min(w1 + 0.2, 0.5)
        else:
            raise Exception('wrong classes')
    elif copy_type == 2:
        if classes == 'two':
            if pred > 0:
                w = min(w1 + 0.03, 0.5)
            elif pred < 0:
                w = max(w1 - 0.03, -0.5)
    return w