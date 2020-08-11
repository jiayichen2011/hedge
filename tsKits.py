
from functools import wraps
import time
import numpy as np
import pandas as pd

"""
时间序列相关工具
"""


class TSKits(object):

    def __init__(self):
        pass

    @staticmethod
    def mad(series):
        """
        mad方式去极值

        Args:
            series:     pd.serious or np.array

        Returns:
            array,
        """
        n = 3
        median = series.quantile(0.5)
        new_median = ((series - median).abs()).quantile(0.50)
        max_range = median + n * new_median
        min_range = median - n * new_median
        return np.clip(series, min_range, max_range)


def check(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):  # 参数类型为可变参数和关键字参数

        a = time.clock()
        x = fun(*args, **kwargs)  # 参数类型为可变参数和关键字参数
        b = time.clock()

        if isinstance(x, pd.DataFrame):

            x[np.isinf(x)] = np.nan
            x.dropna(how='all', axis=1, inplace=True)
            x.dropna(how='all', axis=0, inplace=True)
            x.columns.names = ['windcode']

            x.index.names = ['trade_dt']
            x.name = fun.__name__

            if np.array(x).dtype == 'object':
                print('返回数据存在字符')

            if np.isinf(x).sum().sum() > 0:
                print('返回数据存在无穷大')

            if (~np.isnan(x)).sum(axis=1).min() < x.count(axis=1).mean() * 0.1:
                arg = np.where((~np.isnan(x)).sum(axis=1) < x.count(axis=1).mean() * 0.1)

                print('以下时间有效数据点个数过少: ', x.index[arg[:5]])

        else:

            print('返回数据不是DataFrame')

        print(' %s 已完成 , 耗时 %s 秒' % (fun.__name__, int(b - a)))
        return x

    return wrapper