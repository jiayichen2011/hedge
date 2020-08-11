import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time, datetime
import re
import matplotlib.dates as mdate
import matplotlib
import talib as ta
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from talib.abstract import *


class AnaData(object):

    def __init__(self, start_date='2018/1/3', end_date='2020/5/18', hedge_fut='IF'):
        self.path = os.path.abspath('..')
        self.hedge_fut = hedge_fut

    @property
    def fut_close(self):
        ret = pd.read_csv(self.path + '/datasets/Fut_ClosePrice.csv', index_col=0)
        return ret

    @property
    def equity_close(self):
        df = pd.read_excel(self.path + '/datasets/hs300.xlsx', index_col=0, encoding='unicode_escape')
        ret = df.loc[:, 'CLOSE']
        return ret

    @property
    def fut_volume(self):
        ret = pd.read_csv(self.path + '/datasets/Fut_TurnoverVolume.csv', index_col=0)
        return ret

    @property
    def basis(self):
        ret = pd.read_csv(self.path + '/datasets/Fut_BasisValue.csv', index_col=0)
        return ret

    # 记录了所有合约的剩余到期日
    @property
    def maturity(self):
        ret = pd.read_csv(self.path + '/datasets/IF_maturity.csv', index_col=0)
        return ret

    # 记录了所有合约的最后交易日
    @property
    def last_tradeday(self):
        ret = pd.read_csv(self.path + '/datasets/IF_last_tradeday.csv', index_col=0)
        return ret

    # 记录了远月合约和近月合约的跨期价差
    def spread(self):
        ret = pd.read_csv(self.path + '/datasets/IF_spread0.csv', index_col=0)
        ret = ret.loc[:, 'spread0']
        return ret

    # 记录了远月合约和近月合约的结算价跨期价差
    def settle_spread(self):
        ret = pd.read_csv(self.path + '/datasets/IF_settle_spread0.csv', index_col=0)
        ret = ret.loc[:, 'spread0']
        return ret

    def get_contract_data(self, value, contract_type):
        # 判断合成数据的类型，基差抑或是收盘价
        if value == 'basis':
            df = self.basis
        elif value == 'close':
            df = self.fut_close
        else:
            raise Exception('wrong value!')

        # 判断合成期货的种类
        if contract_type == 'IF':
            IF_code = pd.read_csv(self.path + '/datasets/all_IF.csv', index_col=0)
            data = df[sorted(IF_code.iloc[:, 0])]
        elif contract_type == 'IC':
            IC_code = pd.read_csv(self.path + '/datasets/all_IC.csv', index_col=0)
            data = df[sorted(IC_code.iloc[:, 0])]
        elif contract_type == 'IH':
            IH_code = pd.read_csv(self.path + '/datasets/all_IH.csv', index_col=0)
            data = df[sorted(IH_code.iloc[:, 0])]
        else:
            raise Exception('wrong contract type')
        return data

    def get_maturity(self, contract, date):
        """获取当前合约距离到期日的天数"""
        ret = self.maturity.loc[date, contract]
        return ret

    def rollover_code_judge(self, hedge_fut, date):
        """判断当前日期的近月合约及次月合约的函数"""
        current_date_timestamp = datetime.datetime.strptime(date, "%Y/%m/%d")
        year = current_date_timestamp.year
        month = current_date_timestamp.month
        if month < 8:
            contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
            contract2 = hedge_fut + str(year)[2:] + '0' + str(month + 1)
            contract3 = hedge_fut + str(year)[2:] + '0' + str(month + 2)
        elif month == 8:
            contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
            contract2 = hedge_fut + str(year)[2:] + '0' + str(month + 1)
            contract3 = hedge_fut + str(year)[2:] + str(month + 2)
        elif month == 9:
            contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
            contract2 = hedge_fut + str(year)[2:] + str(month + 1)
            contract3 = hedge_fut + str(year)[2:] + str(month + 2)
        elif month > 9 and month < 11:
            contract1 = hedge_fut + str(year)[2:] + str(month)
            contract2 = hedge_fut + str(year)[2:] + str(month + 1)
            contract3 = hedge_fut + str(year)[2:] + str(month + 2)
        elif month == 11:
            contract1 = hedge_fut + str(year)[2:] + str(month)
            contract2 = hedge_fut + str(year)[2:] + str(month + 1)
            contract3 = hedge_fut + str(year + 1)[2:] + '01'
        elif month == 12:
            contract1 = hedge_fut + str(year)[2:] + str(month)
            contract2 = hedge_fut + str(year + 1)[2:] + '01'
            contract3 = hedge_fut + str(year + 1)[2:] + '02'
        contract1_last_tradeday = self.last_tradeday.loc[contract1, 'last_tradeday']
        contract1_last_tradeday = datetime.datetime.strptime(contract1_last_tradeday, '%Y/%m/%d')
        if current_date_timestamp <= contract1_last_tradeday:
            current_contract = contract1
            next_contract = contract2
        else:
            current_contract = contract2
            next_contract = contract3
            roll_contract = np.nan
        return current_contract, next_contract


class Future(object):

    def __init__(self, data, position=None):
        self._position = None
        self._close = None
        self.delta = None
        self.anaData = data
        self.hedge_fut = None
        self.set_position(position)

    def set_position(self, pos):
        if pos is not None:
            pos = pos.fillna(0)
            self._position = pos
            ft = self._position.columns[0]
            self.hedge_fut = ft[0:2]

    @property
    def position(self):
        return self._position

    @property
    def delta_f(self):
        val = self.anaData.get_contract_data(value='close', contract_type=self.hedge_fut)
        val = val.reindex(self._position.index)
        val.dropna(how='all', axis=1, inplace=True)
        delta = (val.diff() * self._position.shift(1)).sum(axis=1)
        return delta

    @property
    def delta_b(self):
        val = self.anaData.get_contract_data(value='basis', contract_type=self.hedge_fut)
        val = val.reindex(self._position.index)
        val.dropna(how='all', axis=1, inplace=True)
        delta = (val.diff() * self._position.shift(1)).sum(axis=1)
        return delta

    @property
    def trade_date(self):
        ret = self._position[self._position.diff().abs().sum(axis=1) > 0].index.tolist()
        return ret

    def delta_position(self, date):
        pass

    def close(self, date=None):
        if self._close is None:
            self._close = self._cal_close()

        if date is None:
            return self._close
        else:
            return self._close.loc[date]

    def _cal_close(self):
        val = self.anaData.get_contract_data(value='close', contract_type=self.hedge_fut)
        val = val.reindex(self._position.index)
        val.dropna(how='all', axis=1, inplace=True)
        port = (self._position * val).sum(axis=1)
        return port

    def contract_position(self, date):
        if date is None:
            return self._position
        else:
            return self._position.loc[date, :]


class BKT(object):

    def __init__(self, ana_data, start_date, end_date, hedge_fut):
        self.cash = 10000.0 * 10000
        self.commission_rate = 0.001
        self.anaData = ana_data

        self.hedge_fut = hedge_fut
        self.multiplier = 300
        self.fut_port = Future(None)

        self.asset0 = None
        self.asset1 = None
        self.asset2 = None

        self.model = None

        self.start_date = start_date
        self.end_date = end_date
        self.date_list = self.anaData.fut_close.loc[self.start_date:self.end_date, :].index.tolist()
        self._cache = {}

    def set_future(self, fut):
        self.fut_port = fut

    def set_asset(self, asset0, asset1, asset2):
        self.asset0 = asset0
        self.asset1 = asset1
        self.asset2 = asset2

    def set_model(self, model):
        self.model = model

    def get_trade_date(self, frequency):
        """调仓日"""
        if frequency > 0:
            date_list = self.date_list[::frequency]
        else:
            date_list = self.fut_port.trade_date
        return date_list

    def get_hedge_ratio(self, date, copy_type=0, frequency=5, correct=None, beta=None, **kwargs):
        """

        Args:
            date:
            copy_type:  额外对冲比例w，1：delta B/delta F； 2：cov(delta B,delta F)/var(delta F)；3、4:用上未来数据
                                      5：1：1组合策略  6：风险平价组合策略
            frequency:  换仓频率 day
            correct:  修正所用的机器学习模型 randomforest: 随机森林模型
            **kwargs:

        Returns:
            npv策略净值，npv_nocost:无交易成本下策略净值，npv_f:空头端期货的净值
        """
        # 判断对冲比例w的计算方式，
        if copy_type == 0:
            hedge_ratio = self.anaData.equity_close[date] / self.fut_port.close(date)
        elif copy_type == -1:
            hedge_ratio = 1
        elif copy_type == 1:
            hedge_ratio = 1 + get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        elif copy_type == 2:
            hedge_ratio = 1 + get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        elif copy_type == 3:
            hedge_ratio = 1 + get_w_real(self.fut_port.delta_f, self.fut_port.delta_b, date, frequency)
        elif copy_type == 4:
            hedge_ratio = 1 + get_w2_real(self.fut_port.delta_f, self.fut_port.delta_b, date, frequency)
        elif copy_type == 5:
            w1 = 0.5 * get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            w2 = 0.5 * get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            hedge_ratio = 1 + w1 + w2
        elif copy_type == 6:
            w1, w2 = get_portfolio_w(self.asset0, self.asset1, self.asset2, date)
            w1 = w1 * get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            w2 = w2 * get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            hedge_ratio = 1 + w1 + w2
        elif copy_type == 'beta':
            hedge_ratio = get_beta(beta, date)

        if correct == 'randomforest':
            randomforest = self.model[0]
            variable = self.model[1]
            variable = variable.iloc[:, 1:]
            variable_index = variable.index.tolist()
            date_variable = variable.iloc[variable_index.index(date)-1, :]
            pred = randomforest.predict(date_variable.values.reshape(1, len(date_variable)))
            if copy_type == 1:
                w1 = get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            elif copy_type == 2:
                w1 = get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
            if pred > 0:
                w = min(w1 + 0.1, 0.5)
            elif pred < 0:
                w = max(w1 - 0.1, -0.5)
            hedge_ratio = 1 + w
        return hedge_ratio

    def run_bkt(self, frequency=-1, copy_type=0, correct='Nan',beta=None):
        ret = self._cache.get('copy_type' + str(copy_type) + 'frequency' + str(frequency) + correct, None)
        if ret is not None:
            return ret

        trade_date = self.get_trade_date(frequency=frequency)

        contract_list = self.fut_port.position.columns.tolist()
        contract_position = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)
        fut_pnl = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)
        tran_cost = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)

        # 第一天期货持仓头寸多头头寸
        equity_position = int(self.cash / (self.anaData.equity_close[self.start_date] * self.multiplier))
        print(equity_position)
        fut_position = int(equity_position * self.get_hedge_ratio(self.start_date))

        for date in self.date_list:
            if date in trade_date:
                fut_position = int(equity_position * self.get_hedge_ratio(date, copy_type, frequency, correct,beta=beta))
            contract_position.loc[date, :] = self.fut_port.contract_position(date) * fut_position

        # 计算每一交易日的损益
        for contract in contract_position.columns:
            df_position = contract_position[contract]
            contract_close = self.anaData.fut_close.loc[:, contract]
            fut_pnl[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)) * self.multiplier)[
                                1:]
            tran_cost[contract] = (abs(self.multiplier * self.commission_rate * (df_position - df_position.shift(1))
                                       * contract_close))

        s = (self.anaData.equity_close - self.anaData.equity_close.shift(1)) * equity_position * self.multiplier
        fut_pnl.index = [datetime.datetime.strptime(t, '%Y/%m/%d') for t in fut_pnl.index]
        s = s.reindex(fut_pnl.index)
        s[0] = 0
        port_pnl = s - fut_pnl.sum(axis=1)
        cum_pnl = port_pnl.cumsum()
        npv_nocost = 1 + cum_pnl / self.cash

        cum_pnl_cost = (port_pnl - tran_cost.sum(axis=1)).cumsum()
        npv = 1 + cum_pnl_cost / self.cash
        npv_f = 1 + (fut_pnl.sum(axis=1)).cumsum() / self.cash
        self._cache['copy_type' + str(copy_type) + 'frequency' + str(frequency) + correct] = npv, npv_nocost, npv_f
        return npv, npv_nocost, npv_f


class FutHedge(object):
    def __init__(self, start_date='2018/1/4', end_date='2020/6/19',
                 sample_start_date='2017/1/4', hedge_fut='IF'):
        self.hedge_fut = hedge_fut
        self.anaData = AnaData(start_date=sample_start_date, end_date=end_date, hedge_fut=hedge_fut)
        self.start_date = start_date
        self.sample_start_date = sample_start_date
        self.end_date = end_date
        self.index = self.anaData.fut_close.loc[start_date:end_date, :].index
        self.cash = 10000 * 10000

        self.bkt_obj = BKT(ana_data=self.anaData, start_date=self.start_date,
                           end_date=self.end_date, hedge_fut=self.hedge_fut)
        self._cache = {}

    def linear_interpolation(self, value='close', start_date=None, end_date=None, contract_type='IF', T=30):
        """
        返回两个参数，第一个是合成期货的收盘数据，第二个是合成期货的每期损益。
        Args:
            value: 为close则返回收盘价的数据，为basis则返回基差数据
            start_date:
            end_date:
            contract_type:
            T:

        Returns:
            port,
            delta,
            position,
        """

        ret = self._cache.get('linear_interpolation' + value, None)
        if ret is not None:
            return ret

        if start_date is None:
            start_date = self.sample_start_date
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
            print('linear_interpolation ' + date)

            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.anaData.rollover_code_judge(self.hedge_fut, date)

            # 近月合约的剩余到期日
            current_maturity = self.anaData.get_maturity(date=date, contract=current_contract)
            # 次月合约的剩余到期日
            next_maturity = self.anaData.get_maturity(date=date, contract=next_contract)

            # 近月合约持仓比例
            position.loc[date, current_contract] = min(current_maturity / (next_maturity - current_maturity), 1)
            # 次月合约持仓比例
            position.loc[date, next_contract] = 1 - position.loc[date, current_contract]

        port = (position * val).sum(axis=1)
        delta = (val.diff() * position.shift(1)).sum(axis=1)
        self._cache['linear_interpolation' + value] = port, delta, position
        return port, delta, position

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

    # -------------------------------------------------------------------------------------------------
    def get_main_future(self):
        """获取主力合约"""
        hedge_contract = [x for x in self.anaData.fut_close.columns if re.match(self.hedge_fut + '[12]', x)]
        hedge_fut_volume = self.anaData.fut_volume[sorted(hedge_contract)].loc[self.start_date:self.end_date, :]
        hedge_contract_code = hedge_fut_volume.idxmax(axis=1)  # 每个交易日的主力合约(交易量最大)

        df = hedge_contract_code.to_frame(name='contract')
        df['pos'] = 1
        position = df.pivot_table(index=df.index, columns='contract', values='pos').reindex(df.index)

        fut = Future(data=self.anaData, position=position)
        return fut

    def get_linear_future(self):
        """获取平滑展期合约"""
        ret = self._cache.get('get_linear_future', None)
        if ret is not None:
            return ret
        ret = self._cache.get('linear_interpolationclose', None)
        if ret is not None:
            position = ret[2].loc[self.start_date:self.end_date, :]
            ret = Future(data=self.anaData, position=position)
            return ret

        val = self.anaData.get_contract_data(value='close', contract_type=self.hedge_fut)
        val = val.loc[self.start_date:self.end_date, :]
        val.dropna(how='all', axis=1, inplace=True)

        date_list = val.index.tolist()  # 所有交易日序列
        contract_list = val.columns.tolist()

        # 用于记录每一交易日的近月合约及次月合约的持仓比例
        position = pd.DataFrame(0.0, index=date_list, columns=contract_list)

        for date in date_list:
            print('get_linear_future ', date)
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.anaData.rollover_code_judge(self.hedge_fut, date)

            # 近月合约的剩余到期日
            current_maturity = self.anaData.get_maturity(date=date, contract=current_contract)
            # 次月合约的剩余到期日
            next_maturity = self.anaData.get_maturity(date=date, contract=next_contract)

            # 近月合约持仓比例
            position.loc[date, current_contract] = min(current_maturity / (next_maturity - current_maturity), 1)
            # 次月合约持仓比例
            position.loc[date, next_contract] = 1 - position.loc[date, current_contract]

        fut = Future(data=self.anaData, position=position)
        self._cache['get_linear_future'] = fut
        return fut

    def get_select_future(self):
        """获取择时合约"""
        ret = self._cache.get('get_select_future', None)
        if ret is not None:
            return ret

        val = self.anaData.get_contract_data(value='close', contract_type=self.hedge_fut)
        val = val.loc[self.start_date:self.end_date, :]
        val.dropna(how='all', axis=1, inplace=True)

        spread = self.anaData.spread()
        settle_spread = self.anaData.settle_spread()
        spread_index = spread.index.tolist()

        date_list = val.index.tolist()  # 所有交易日序列
        contract_list = val.columns.tolist()

        # 用于记录每一交易日的近月合约及次月合约的持仓比例
        position = pd.DataFrame(0.0, index=date_list, columns=contract_list)

        rollover_sign1 = False
        rollover_sign2 = False
        rollover_sign = False

        for date in date_list:
            print('get_select_future ', date)
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.anaData.rollover_code_judge(self.hedge_fut, date)
            recent = self.anaData.last_tradeday.loc[current_contract, 'last_tradeday']

            # 距离近月合约交割日还剩多少交易日
            deadline = spread_index.index(recent) - spread_index.index(date)

            # 历史二十天的跨期价差数据
            past_spread = spread[spread_index.index(date) - 19:spread_index.index(date) + 1]
            past_spread[-1] = settle_spread[date]

            if deadline <= 10 and rollover_sign == False:
                test_sign = True
            else:
                test_sign = False
            if test_sign:
                hp_spread = hp_filter(past_spread)
                rollover_sign1 = ((hp_spread[-2:] <= past_spread[-2:]).all()) and (past_spread[-2] < past_spread[-1])
                rollover_sign2 = (deadline == 0)
                rollover_sign = rollover_sign1 or rollover_sign2
            if rollover_sign:
                position.loc[date, next_contract] = 1.0
            else:
                # 近月合约的剩余到期日
                current_maturity = self.anaData.get_maturity(date=date, contract=current_contract)
                # 次月合约的剩余到期日
                next_maturity = self.anaData.get_maturity(date=date, contract=next_contract)

                # 近月合约持仓比例
                position.loc[date, current_contract] = min(current_maturity / (next_maturity - current_maturity), 1)
                # 次月合约持仓比例
                position.loc[date, next_contract] = 1 - position.loc[date, current_contract]
            if deadline == 0:
                rollover_sign = False
        fut = Future(data=self.anaData, position=position)
        self._cache['get_select_future'] = fut
        return fut

    def get_beta_future(self):
        position = pd.read_excel(r'C:\Users\jiayi\Desktop\export_dataframe.xlsx')

        fut = Future(data=self.anaData, position=position)
        return fut

    def get_sample(self, frequency):
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
        pred_boverf = (delta_b / delta_f).rolling(20).apply(hp_filter2)
        soverf = delta_s / delta_f

        delta_f_mean = delta_f.rolling(20).mean()
        delta_b_mean = delta_b.rolling(20).mean()
        bf = (delta_f * delta_b).rolling(20).mean()
        cov = bf - delta_b_mean * delta_f_mean
        var = delta_f.rolling(20).var()
        w2 = cov / var

        std_f = port_f.rolling(20).std()

        real_boverf = pred_boverf.shift(-frequency)
        real_boverf[-frequency:] = real_boverf[-frequency-1]

        deviation = (pred_boverf - real_boverf).shift(1)

        y = np.where(real_boverf >= pred_boverf, 1, -1)
        y = pd.Series(y, index=real_boverf.index)
        all_variable = pd.concat([y, delta_b, delta_f, delta_s, pred_boverf, deviation, soverf, w2, std_f], axis=1)
        return all_variable

    def get_GBT(self, frequency=1):
        df = self.get_sample(frequency=frequency)
        df2 = df.dropna(how='any')
        df2 = df2.loc[:self.start_date, :]
        y = df2.iloc[:, 0].values.reshape(-1, 1)
        x = df2.iloc[:, 1:].values.reshape(len(df2), len(df2.T) - 1)
        forest = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0)
        forest.fit(x, y)
        return forest, df

    def get_randomforest(self, frequency=1):
        df = self.get_sample(frequency=frequency)
        df2 = df.dropna(how='any')
        df2 = df2.loc[:self.start_date, :]
        y = df2.iloc[:, 0].values.reshape(-1, 1)
        x = df2.iloc[:, 1:].values.reshape(len(df2), len(df2.T) - 1)
        forest = RandomForestClassifier(n_estimators=300, random_state=0)
        forest.fit(x, y)
        return forest, df

    #
    def hedge(self, contract=0, copy_type=1, frequency=-1):
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
        Returns:
            tuple of pd.serious,     策略净值， 无成本策略净值， 空头累计收益
        """

        if contract == 0:
            fut = self.get_main_future()
        elif contract == 1:
            fut = self.get_linear_future()
        else:
            raise Exception("wrong contract!")

        self.bkt_obj.set_future(fut=fut)
        ret = self.bkt_obj.run_bkt(copy_type=copy_type, frequency=frequency)
        return ret

    # --------------------------------------------------------------------------------------------------------
    def beta_hedge(self,beta, copy_type='beta'):
        self.bkt_obj.set_future(fut=self.get_beta_future())
        ret = self.bkt_obj.run_bkt(copy_type=copy_type,beta=beta)
        print(copy_type)
        return ret

    def fundamental_hedge(self, copy_type=0):
        """主力合约基础策略"""
        # 获取每个交易日的主力合约
        self.bkt_obj.set_future(fut=self.get_main_future())
        print('fundamental_hedge')
        return self.bkt_obj.run_bkt(copy_type=copy_type)

    def fundamental3(self):
        """合成合约基础策略"""
        self.bkt_obj.set_future(fut=self.get_linear_future())
        print('fundamental3')
        return self.bkt_obj.run_bkt()

    def copy2_hs300(self, copy_type=1, frequency=-1):
        """
        :param copy_type: 额外对冲比例w， 1：delta B/delta F；
                                        2：cov(delta B,delta F)/var(delta F)；
                                        3、4:用上未来数据
        :param frequency: 换仓频率 day
        :return: npv策略净值，npv_nocost:无交易成本下策略净值，npv_f:空头端期货的净值
        """
        self.bkt_obj.set_future(fut=self.get_linear_future())
        ret = self.bkt_obj.run_bkt(frequency=frequency, copy_type=copy_type)
        print('copy2_hs300 copy_type:', copy_type, " frequency:", frequency)
        return ret

    def portfolio(self, copy_type=6, frequency=-1):

        self.bkt_obj.set_future(fut=self.get_main_future())
        asset0 = self.bkt_obj.run_bkt(copy_type=-1)[0]
        self.bkt_obj.set_future(fut=self.get_linear_future())
        asset1 = self.bkt_obj.run_bkt(frequency=frequency, copy_type=1)[0]
        asset2 = self.bkt_obj.run_bkt(frequency=frequency, copy_type=2)[0]
        self.bkt_obj.set_asset(asset0=asset0, asset1=asset1, asset2=asset2)
        ret = self.bkt_obj.run_bkt(frequency=frequency, copy_type=copy_type)
        print('portfolio copy_type:', copy_type, " frequency:", frequency)
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
        bkt_obj = BKT(ana_data=self.anaData, start_date=self.start_date,
                      end_date=self.end_date, hedge_fut=self.hedge_fut)
        bkt_obj.set_future(fut=self.get_select_future())
        ret = bkt_obj.run_bkt(frequency=frequency, copy_type=copy_type)
        print('select_time copy_type:', copy_type, " frequency:", frequency)
        return ret

    def randomforest(self, copy_type=1, frequency=1, correct='randomforest'):
        self.bkt_obj.set_model(model=self.get_randomforest(frequency=frequency))
        self.bkt_obj.set_future(fut=self.get_linear_future())
        ret = self.bkt_obj.run_bkt(frequency=frequency, copy_type=copy_type, correct=correct)
        print(correct, ' copy_type:', copy_type, " frequency:", frequency)
        return ret


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

def get_beta(S_beta, current_tradeday):
    return S_beta[current_tradeday]

def get_w_real(delta_f, delta_b, current_tradeday, frequency=15):
    all_tradeday = delta_f.index.tolist()
    delta_b_roll = delta_b[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    delta_f_roll = delta_f[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    if len(delta_b_roll) < 5:
        return 0
    w = -(delta_b_roll / delta_f_roll)
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
    var = delta_f_var = delta_f.rolling(30).var()
    w = cov / var
    w_30 = w[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency]
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


class Drawing(object):
    """
    1为基础策略收益图
    2为期现货走势图
    3为deltaB/deltaF散点图
    4为deltaB/deltaF走势图'''
    """

    def __init__(self):
        self.start_date = '2018/1/3'
        self.end_date = '2020/5/18'

        self.fut_hedge = FutHedge(start_date=self.start_date, end_date=self.end_date)
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        matplotlib.rcParams['axes.unicode_minus'] = False
        self.linear_inter_fut = self.fut_hedge.get_linear_future()
        self.f_linear, self.delta_f, df = self.fut_hedge.linear_interpolation()
        self.b_linear, self.delta_b, df = self.fut_hedge.linear_interpolation(value='basis')

    def drawing1(self):
        """基础策略收益图"""
        npv, npv_nocost, _ = self.fut_hedge.fundamental_hedge()
        x = npv_nocost.index
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range('2010/1/6', '2020/7/16', freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        plt.plot(x, npv_nocost)
        plt.title('无交易成本 基础策略', fontsize=18)
        plt.show()

    def drawing2(self, current=True, next=True):
        """期现货走势对比图"""
        equity_close = pd.read_excel('D://zsfund/data/hs300.xlsx', index_col=0, encoding='unicode_escape')
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range('2010/1/6', '2020/7/16', freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        f0, delta_f0 = self.fut_hedge.current_month_basis('close')
        f1, delta_f1 = self.fut_hedge.next_month_basis('close')
        start_date = (self.f_linear.index)[0]
        end_date = (self.f_linear.index)[-1]
        equity_position = 1 / equity_close.loc[start_date, 'CLOSE']
        delta_s = equity_close.loc[:, 'CLOSE'] - equity_close.loc[:, 'CLOSE'].shift(1)
        delta_s = delta_s[start_date:end_date]
        delta_s[start_date] = 0
        cumsum_s = equity_position * delta_s.cumsum()

        fut_position = 1 / self.f_linear[start_date]
        self.delta_f[start_date] = 0
        cumsum_f = fut_position * self.delta_f[start_date:end_date].cumsum()

        f0_position = 1 / f0[start_date]
        delta_f0[start_date] = 0
        cumsum_f0 = f0_position * delta_f0[start_date:end_date].cumsum()

        f1_position = 1 / f1[start_date]
        delta_f1[start_date] = 0
        cumsum_f1 = f1_position * delta_f1[start_date:end_date].cumsum()

        x = delta_s[start_date:end_date].index
        plt.plot(x, cumsum_f, linewidth=1)
        plt.plot(x, cumsum_s, linewidth=1)
        plt.legend(['合成股指期货', 'hs300'], fontsize=20)
        if current:
            plt.plot(x, cumsum_f0)
            plt.legend(['合成股指期货', 'hs300', '近月合约'], fontsize=20)
        if next:
            plt.plot(x, cumsum_f1)
            plt.legend(['合成股指期货', 'hs300', '次月合约'], fontsize=20)
        if current and next:
            plt.legend(['合成股指期货', 'hs300', '近月合约', '次月合约', ], fontsize=20)
        plt.show()

    def drawing3(self):
        """delta B/delta F 散点图"""
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range(self.start_date, self.end_date, freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        plt.scatter(self.delta_f, self.delta_b, facecolor='g', linewidths=0.1)
        plt.xlabel('delta F', fontsize=18)
        plt.ylabel('delta B', fontsize=18)
        plt.show()

    def drawing4(self, start_date=None, end_date=None, window=[1], filter=True, linestyle='plot'):
        """
        delta F/delta B 走势图

        Args:
            start_date:
            end_date: 选定画图数据的区间，如果报错说明日期不在交易日，日期上下微调几天即可
            window: 要传入一个list。例如，window=[1,5]，则画出窗口期为一天以及五天的走势图
            filter: filter为true，则会对数据去极值处理，否则不会对数据去极值
            linestyle: linstyle为画图类型，plot为折线图，bar为柱状图

        Returns:

        """
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date

        fig, axes = plt.subplots(len(window), 1, figsize=(15, 10))
        x = [datetime.datetime.strptime(d, "%Y/%m/%d") for d in self.delta_b[start_date:end_date].index]
        for i, j in zip(window, range(len(window))):
            y_f = self.delta_f.rolling(i).sum()[start_date:end_date]
            y_b = self.delta_b.rolling(i).sum()[start_date:end_date]
            if filter:
                Y = sigmoid(y_b / y_f)
            else:
                Y = (y_b / y_f)
            if len(window) == 1:
                if linestyle == 'plot':
                    plt.plot(x, Y)
                elif linestyle == 'bar':
                    plt.bar(x, Y)
                plt.legend(['delta B/delta F ' + str(i) + 'days'], fontsize=12, loc='upper right')
            elif len(window) > 1:
                if linestyle == 'plot':
                    Y.plot(ax=axes[j])
                elif linestyle == 'bar':
                    Y.bar(ax=axes[j])
                axes[j].legend(['delta B/delta F ' + str(i) + 'days'], fontsize=12, loc='upper right')

        plt.show()
beta=read()

if __name__ == '__main__':
    obj = FutHedge(start_date='2015/4/16',end_date='2020/6/19')
    res_dic = {'F0_s0_num': obj.fundamental_hedge(beta),
               'F0_beta_d5':obj.beta_hedge()
               #'FS_forest_d1': obj.randomforest(copy_type=1, frequency=1, correct='randomforest'),
               #'FS_forest_d5': obj.randomforest(copy_type='forest', frequency=5, correct='randomforest'),
               #'FS_forest_d10': obj.randomforest(copy_type='forest', frequency=10, correct='randomforest'),
               # 'F0_s0_mkt': obj.fundamental_hedge(copy_type=0),
               # 'FS_s0_mkt':  obj.fundamental3(),
               # 'FS_s6_d1': obj.portfolio(copy_type=6, frequency=1),
               # 'FS_s6_d5': obj.portfolio(copy_type=6, frequency=5),
               # 'FS_s6_d10': obj.portfolio(copy_type=6, frequency=10),
               # 'FS_s7_d1': obj.portfolio(copy_type=7, frequency=1),
               # 'FS_s7_d5': obj.portfolio(copy_type=7, frequency=5),
               # 'FS_s7_d10': obj.portfolio(copy_type=7, frequency=10),
               #'FS_s1_d1': obj.copy2_hs300(copy_type=1, frequency=1),
               # 'FS_s2_d1': obj.copy2_hs300(copy_type=2, frequency=1),
               #'FS_s1_d5': obj.copy2_hs300(copy_type=1, frequency=5),
               # 'FS_s2_d5': obj.copy2_hs300(copy_type=2, frequency=5),
               #'FS_s1_d10': obj.copy2_hs300(copy_type=1, frequency=10),
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