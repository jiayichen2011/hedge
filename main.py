from project01.base import *


# ------------------------------------------------------------------------------------------------------------------
class FutHedge(object):
    def __init__(self, start_date='2020/1/8', end_date='2020/6/19',
                 sample_start_date='2019/6/18', hedge_fut='IF'):
        self.hedge_fut = hedge_fut
        self.anaData = AnaData(start_date=sample_start_date, end_date=end_date, hedge_fut=hedge_fut)

        self.start_date = start_date
        self.sample_start_date = sample_start_date
        self.end_date = end_date
        self.index = self.anaData.fut_close.loc[start_date:end_date, :].index

        # FuturePortGen配置
        self.futGen = FuturePortGen(self.anaData)
        self.futGen.hedge_fut = self.hedge_fut
        self.futGen.start_date = self.start_date
        self.futGen.end_date = self.end_date

        # BKT配置
        self.bkt_obj = BKT(ana_data=self.anaData, start_date=self.start_date,
                           end_date=self.end_date, hedge_fut=self.hedge_fut)
        self._cache = {}

    def current_month_basis(self, value='close', start_date=None, end_date=None, contract_type='IF'):
        """data为close则返回收盘价的数据，data为basis则返回基差数据"""
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        val = self.anaData.get_contract_data(value=value, contract_type=contract_type)
        val = val.loc[start_date:end_date, :]
        val.dropna(how='all', axis=1, inplace=True)

        date_list = val.index.tolist()  # 所有交易日序列
        contract_list = val.columns.tolist()

        # 用于记录每一交易日的近月合约及次月合约的持仓比例
        position = pd.DataFrame(0.0, index=date_list, columns=contract_list)

        for date in date_list:
            print(date)
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.anaData.rollover_code_judge(self.hedge_fut, date)
            # 近月合约持仓比例
            position.loc[date, current_contract] = 1

        port = (position * val).sum(axis=1)
        delta = (val.diff() * position.shift(1)).sum(axis=1)
        return port, delta

    def next_month_basis(self, value='close', start_date=None, end_date=None, contract_type='IF'):
        """data为close则返回收盘价的数据，data为basis则返回基差数据"""
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        val = self.anaData.get_contract_data(value=value, contract_type=contract_type)
        val = val.loc[start_date:end_date, :]
        val.dropna(how='all', axis=1, inplace=True)

        date_list = val.index.tolist()  # 所有交易日序列
        contract_list = val.columns.tolist()

        # 用于记录每一交易日的近月合约及次月合约的持仓比例
        position = pd.DataFrame(0.0, index=date_list, columns=contract_list)

        for date in date_list:
            print(date)
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.anaData.rollover_code_judge(self.hedge_fut, date)
            # 近月合约持仓比例
            position.loc[date, next_contract] = 1

        port = (position * val).sum(axis=1)
        delta = (val.diff() * position.shift(1)).sum(axis=1)
        return port, delta

    def get_sample(self, frequency):
        equity_close = self.anaData.equity_close
        port_f, delta_f, position_f = self.anaData.linear_interpolation()
        port_b, delta_b, position_b = self.anaData.linear_interpolation('basis')
        index = port_f.index
        index = [datetime.datetime.strptime(d, "%Y/%m/%d") for d in index]
        equity_close = equity_close[index]
        equity_close.index = port_f.index

        delta_b = delta_b.rolling(frequency).sum()
        delta_f = delta_f.rolling(frequency).sum()
        delta_s = equity_close.diff().rolling(frequency).sum()
        pred_boverf = (delta_b / delta_f).rolling(20).apply(hp_filter2, raw=False)
        soverf = delta_s / delta_f

        delta_f_mean = delta_f.rolling(20).mean()
        delta_b_mean = delta_b.rolling(20).mean()
        bf = (delta_f * delta_b).rolling(20).mean()
        cov = bf - delta_b_mean * delta_f_mean
        var = delta_f.rolling(20).var()
        w2 = cov / var

        std_f = port_f.rolling(20).std()

        real_boverf = pred_boverf.shift(-frequency - 1)
        real_boverf[-frequency - 1:] = real_boverf[-frequency - 2]  # 填充空值

        deviation = pred_boverf - real_boverf

        s1y_variable = pd.DataFrame(index=real_boverf.index)  #
        s2y_variable = pd.DataFrame(index=real_boverf.index)
        # 对模型进行n分类，获取n分类的y
        s1y2 = np.where(real_boverf >= pred_boverf, 1, -1)
        s1y4 = np.where(deviation < -0.15, -2,
                        np.where(deviation < 0, -1,
                                 np.where(deviation < 0.15, 1, 2)))
        s1y6 = np.where(deviation < -0.2, -3,
                        np.where(deviation < -0.1, -2,
                                 np.where(deviation < 0, -1,
                                          np.where(deviation < 0.1, 1,
                                                   np.where(deviation < 0.2, 2, 3)))))

        s1y_variable['two'] = s1y2
        s1y_variable['four'] = s1y4
        s1y_variable['six'] = s1y6

        real_w2 = w2.shift(-10)
        real_w2[-10:] = real_w2[-11]

        s2y2 = np.where(real_w2 > w2, 1, -1)
        s2y_variable['two'] = s2y2
        # 合并特征向量
        x_variable = pd.concat([delta_b, delta_f, delta_s, pred_boverf, deviation.shift(1), soverf, w2, std_f], axis=1)
        x_variable.columns = ['delta_b', 'delta_f', 'delta_s', 'pred_boverf', 'previous_s1', 'soverf', 'w2', 'std_f']
        # 筛掉向量当中的空值
        x_variable = x_variable.iloc[40:, :]
        s1y_variable = s1y_variable.iloc[40:, :]
        s2y_variable = s2y_variable.iloc[40:, :]

        return x_variable, s1y_variable, s2y_variable

    def get_sample2(self, frequency):
        equity_close = self.anaData.equity_close
        port_f, delta_f, position_f = self.linear_interpolation()
        port_b, delta_b, position_b = self.linear_interpolation('basis')
        index = port_f.index
        index = [datetime.datetime.strptime(d, "%Y/%m/%d") for d in index]
        equity_close = equity_close[index]
        equity_close.index = port_f.index

        delta_b = delta_b.rolling(frequency).sum()
        delta_f = delta_f.rolling(frequency).sum()
        delta_s = equity_close.diff().rolling(frequency).sum()
        pred_boverf = (delta_b / delta_f).rolling(20).apply(hp_filter2, raw=False)
        soverf = delta_s / delta_f

        delta_f_mean = delta_f.rolling(20).mean()
        delta_b_mean = delta_b.rolling(20).mean()
        bf = (delta_f * delta_b).rolling(20).mean()
        cov = bf - delta_b_mean * delta_f_mean
        var = delta_f.rolling(20).var()
        w2 = cov / var

        std_f = port_f.rolling(20).std()

        real_boverf = pred_boverf.shift(-frequency-1)
        real_boverf[-frequency-1:] = real_boverf[-frequency-2]
        deviation2 = pred_boverf - real_boverf
        greed_rate = np.where(delta_f > 0, -0.005, 0.005)
        greed_rate = pd.Series(greed_rate, index=port_f.index)
        real_boverf = real_boverf + greed_rate * frequency

        deviation = pred_boverf - real_boverf

        y2 = np.where(real_boverf >= pred_boverf, 1, -1)
        y2 = pd.DataFrame(y2, index=real_boverf.index, columns=['two'])

        y4 = np.where(deviation < -0.15, -2,
                      np.where(deviation < 0, -1,
                              np.where(deviation < 0.15, 1, 2)))
        y4 = pd.DataFrame(y4, index=real_boverf.index, columns=['four'])

        y6 = np.where(deviation < -0.2, -3,
                     np.where(deviation<-0.1, -2,
                              np.where(deviation<0, -1,
                                      np.where(deviation<0.1, 1,
                                              np.where(deviation < 0.2, 2, 3)))))
        y6 = pd.DataFrame(y6, index=real_boverf.index, columns=['six'])

        x_variable = pd.concat([delta_b, delta_f, delta_s, pred_boverf, deviation2.shift(1), soverf, w2, std_f], axis=1)
        y_variable = pd.concat([y2, y4, y6], axis=1)
        x_variable = x_variable.iloc[40:, :]
        y_variable = y_variable.iloc[40:, :]
        return x_variable, y_variable

    #
    def hedge(self, contract=1, copy_type=1, frequency=-1, correct='nan', classes='nan'):
        """
        对冲策略回测

        Args:
            contract:       合约种类， 0-主力合约； 1-连续展期合约
            copy_type:      -1: 1比1数量对冲;
                            0: -1比1市值对冲;
                            1: delta;
                            2: 波动率;
            frequency:      -1: 仓位比例变化，即重新计算对冲比例
                            n: n大于0, n天固定窗口调整对冲比例
            correct:        修正用的机器学习模型
            classes:        模型几分类
        Returns:
            tuple of pd.serious,     策略净值， 无成本策略净值， 空头累计收益
        """
        if contract == 0:
            fut = self.futGen.get_main_future()
        elif contract == 1:
            fut = self.futGen.get_linear_future()
        elif contract == 2:
            fut = self.futGen.get_select_future()
        else:
            raise Exception("wrong contract!")

        if copy_type == 0:
            st = Strategy00()
        elif copy_type == -1:
            st = StrategyN1()
        elif copy_type == 1 and correct == 'nan':
            st = Strategy01()
        elif copy_type == 2 and correct == 'nan':
            st = Strategy02()
        elif copy_type == 6:
            st = Strategy06(bkt_obj=self.bkt_obj, fut_gen=self.futGen, frequency=frequency)
        elif correct == 'RF' and classes != 'nan':
            if copy_type == 1:
                raw_st = Strategy01()
            elif copy_type == 2:
                raw_st = Strategy02()
            st = RFStrategy(strategy=raw_st, classes=classes, copy_type=copy_type)
            st.set_param(fut_port=fut, ana_data=self.anaData)
        else:
            raise Exception("wrong strategy!")

        self.bkt_obj.set_strategy(st=st)
        self.bkt_obj.set_future(fut=fut)
        ret = self.bkt_obj.run_bkt(frequency=frequency)
        print(st.name, ' frequency: ', frequency)
        return ret

    def select_time(self, copy_type=1, frequency=-1):
        """
        :param copy_type: 额外对冲比例w， 1：delta B/delta F；
                                        2：cov(delta B,delta F)/var(delta F)；
                                        3、4:用上未来数据
        :param frequency: 换仓频率 day
        :param select: linear 对合成合约择时，main 对主力合约择时
        :return: npv策略净值，npv_nocost:无交易成本下策略净值，npv_f:空头端期货的净值
        """
        ret = self.hedge(contract=2, copy_type=copy_type, frequency=frequency)
        print('select_time copy_type:', copy_type, " frequency:", frequency)
        return ret

    def randomforest(self, copy_type=1, frequency=1, correct='randomforest'):
        self.bkt_obj.set_model(model=self.get_randomforest(frequency=frequency))
        self.bkt_obj.set_future(fut=self.futGen.get_linear_future())
        ret = self.bkt_obj.run_bkt(frequency=frequency, copy_type=copy_type, correct=correct)
        print(correct, ' copy_type:', copy_type, " frequency:", frequency)
        return ret


if __name__ == '__main__':
    obj = FutHedge(start_date='2018/1/4', sample_start_date='2017/1/4')
    res_dic = {'F0_s0_num': obj.hedge(contract=0, copy_type=-1),
               # 'FS_RFs1_d1': obj.hedge(copy_type=1, frequency=1, correct='RF', classes='two'),
               # 'FS_RFs1_d2': obj.hedge(copy_type=1, frequency=2, correct='RF', classes='two'),
               # 'FS_RFs1_d3': obj.hedge(copy_type=1, frequency=3, correct='RF', classes='two'),
               'FS_RFs2_d1': obj.hedge(copy_type=2, frequency=1, correct='RF', classes='two'),
               'FS_RFs2_d3': obj.hedge(copy_type=2, frequency=3, correct='RF', classes='two'),
               'FS_RFs2_d5': obj.hedge(copy_type=2, frequency=5, correct='RF', classes='two'),
               'FS_RFs2_d7': obj.hedge(copy_type=2, frequency=7, correct='RF', classes='two'),
               # 'FS_s1_d1': obj.hedge(copy_type=1, frequency=1),
               # 'FS_s1_d2': obj.hedge(copy_type=1, frequency=2),
               # 'FS_s1_d3': obj.hedge(copy_type=1, frequency=3),
               # 'FS_s1_d5': obj.hedge(copy_type=1, frequency=5),
               # ‘FS_s1_d7': obj.hedge(copy_type=1, frequency=7),
               'FS_s2_d1': obj.hedge(copy_type=2, frequency=1),
               #'FS_s1_d2': obj.hedge(copy_type=2, frequency=2),
               'FS_s2_d3': obj.hedge(copy_type=2, frequency=3),
               'FS_s2_d5': obj.hedge(copy_type=2, frequency=5),
               'FS_s2_d7': obj.hedge(copy_type=2, frequency=7),
               # 'FS_RFs2_d1': obj.hedge(copy_type=2, frequency=1, correct='RF', classes= 'two'),
               # 'F0_s0_mkt': obj.fundamental_hedge(copy_type=0),
               # 'FS_s0_mkt':  obj.fundamental3(),
               # 'FS_s6_d1': obj.portfolio(copy_type=6, frequency=1),
               # 'FS_s6_d5': obj.portfolio(copy_type=6, frequency=5),
               # 'FS_s6_d10': obj.portfolio(copy_type=6, frequency=10),
               # 'FS_s7_d1': obj.portfolio(copy_type=7, frequency=1),
               # 'FS_s7_d5': obj.portfolio(copy_type=7, frequency=5),
               # 'FS_s7_d10': obj.portfolio(copy_type=7, frequency=10),
               # 'FS_s1_d1': obj.copy2_hs300(copy_type=1, frequency=1),
               # 'FS_s2_d1': obj.copy2_hs300(copy_type=2, frequency=1),
               # 'FS_s1_d5': obj.copy2_hs300(copy_type=1, frequency=5),
               # 'FS_s2_d5': obj.copy2_hs300(copy_type=2, frequency=5),
               # 'FS_s1_d10': obj.copy2_hs300(copy_type=1, frequency=10),
               # 'FS_s2_d10': obj.copy2_hs300(copy_type=2, frequency=10),
               }

    tmp = 1
    base_key = 'F0_s0_num'

    plt.figure()
    plt.title('/ with cost')
    key_list = list(res_dic.keys())
    for k, v in res_dic.items():
        (res_dic[k][0] / res_dic[base_key][0]).plot()
    plt.legend(key_list)

    plt.figure()
    plt.title('/ no cost')
    key_list = list(res_dic.keys())
    for k, v in res_dic.items():
        (res_dic[k][1] / res_dic[base_key][1]).plot()
    plt.legend(key_list)

    plt.figure()
    plt.title('nav with cost')
    key_list = list(res_dic.keys())
    for k, v in res_dic.items():
        (res_dic[k][0]).plot()
    plt.legend(key_list)

    plt.figure()
    plt.title('nav no cost')
    key_list = list(res_dic.keys())
    for k, v in res_dic.items():
        (res_dic[k][1]).plot()
    plt.legend(key_list)

    plt.show()
    tmp = 2