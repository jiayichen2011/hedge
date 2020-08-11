# import pandas as pd
# ContractCode='IF2001'
# data=pd.read_csv(r'C:\Users\jiayi\Desktop\Fut_TradingQuote.csv')
def k_plot(ContractCode, data, ma1=5, ma2=10, ma3=20):
    """
    需要输入ContractCode(IF2001) 以及data(in dataframe)
    ma1 ma2 ma3有default 但是也可以定义。
    """
    import mplfinance as mpf
    import tushare as ts
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.pylab import date2num
    import numpy as np
    from mplfinance.original_flavor import candlestick_ohlc

    df = data[data['ContractCode'] == ContractCode]
    df = df.drop(['MainContractMark', 'JSID', 'UpdateTime', 'ChangePCTTurnoverValue', 'ChangeOfTurnoverValue'], axis=1)
    df = df.drop(['TurnoverValue', 'ChangePCTTurnoverVolume', 'ChangeOfTurnoverVolume', 'PrevClosePrice'], axis=1)
    df = df.drop(['ContractInnerCode', 'ID', 'ExchangeCode', 'OptionCode', 'SeriesFlag', 'PrevSettlePrice'], axis=1)
    df = df.drop(['ChangeOfCTPS', 'ChangePCTCTPS', 'ChangeOfClosePrice', 'ChangePCTClosePrice', 'SettlePrice',
                  'ChangeOfSettPrice'], axis=1)
    df = df.drop(['ChangePCTSettPrice', 'OpenInterest', 'ChangeOfOpenInterest', 'ChangePCTOpenInterest', 'BasisValue'],
                 axis=1)
    # df=df.drop(['TurnoverVolume'],axis=1)

    df['Date'] = [x[:10] for x in df['TradingDay']]

    order = ['Date', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'TurnoverVolume']
    df = df[order]

    df['Date2'] = df['Date'].copy()
    df['Date'] = pd.to_datetime(df['Date']).map(date2num)

    # SMA:简单移动平均(Simple Moving Average)
    time_period = 20  # SMA的计算周期，默认为20
    stdev_factor = 2  # 上下频带的标准偏差比例因子
    history = []  # 每个计算周期所需的价格数据
    sma_values = []  # 初始化SMA值
    upper_band = []  # 初始化阻力线价格
    lower_band = []  # 初始化支撑线价格

    # 构造列表形式的绘图数据
    for close_price in df['ClosePrice']:
        #
        history.append(close_price)

        # 计算移动平均时先确保时间周期不大于20
        if len(history) > time_period:
            del (history[0])

        # 将计算的SMA值存入列表
        sma = np.mean(history)
        sma_values.append(sma)
        # 计算标准差
        stdev = np.sqrt(np.sum((history - sma) ** 2) / len(history))
        upper_band.append(sma + stdev_factor * stdev)
        lower_band.append(sma - stdev_factor * stdev)

    # 将计算的数据合并到DataFrame
    df = df.assign(CP=pd.Series(df['ClosePrice'], index=df.index))
    df = df.assign(middle=pd.Series(sma_values, index=df.index))
    df = df.assign(bolu=pd.Series(upper_band, index=df.index))
    df = df.assign(bold=pd.Series(lower_band, index=df.index))
    # print(df)
    df['ma1'] = df.ClosePrice.rolling(ma1).mean()
    df['ma2'] = df.ClosePrice.rolling(ma2).mean()
    df['ma3'] = df.ClosePrice.rolling(ma3).mean()

    df['Date3'] = pd.to_datetime(df['Date2'], format='%Y/%m/%d')
    df.set_index(['Date3'], inplace=True)
    # print(df)

    import mplfinance as mpf

    df2 = df.drop(['CP', 'Date', 'middle'], axis=1)
    df2.rename(columns={'OpenPrice': 'open', 'HighPrice': 'high', 'LowPrice': 'low', 'ClosePrice': 'close',
                        'TurnoverVolume': 'volume'}, inplace=True)
    # Plot candlestick.
    # Add volume.
    # Add moving averages: 3,6,9.
    # Save graph to *.png.

    # add_plot = mpf.make_addplot(df2[['bolu', 'bold']],width=0.4)
    # add_plot = mpf.make_addplot(df2[['ma1']],width=0.4,color='y')
    # add_plot = mpf.make_addplot(df2[['ma2']],width=0.4,color='g')
    # add_plot = mpf.make_addplot(df2[['ma3']],width=0.4,color='black')

    add_plot = [mpf.make_addplot(df2[['bolu', 'bold']], width=0.4),
                mpf.make_addplot(df2[['ma1']], width=0.4, color='y'),
                mpf.make_addplot(df2[['ma2']], width=0.4, color='g'),
                mpf.make_addplot(df2[['ma3']], width=0.4, color='black')
                ]

    mpf.plot(df2, type='candle', style='charles', addplot=add_plot, datetime_format='%y.%m.%d', volume=True,
             figscale=1.5, title=f"\n{ContractCode}")
    # ax.legend(ma1, ma2, ma3), ('ab', 'ba', 'av'))

    # plt.show()

    # add_plot = mplfinance.make_addplot(df2[['bolu', 'bold']])
    # mplfinance.plot(data, addplot=add_plot)

# k_plot(ContractCode, data)
