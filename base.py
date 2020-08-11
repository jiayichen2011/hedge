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
from project01.tools import *


class AnaData(object):

    def __init__(self, start_date='2018/1/3', end_date='2020/5/18', hedge_fut='IF'):
        self.path = os.path.abspath('.')
        self.hedge_fut = hedge_fut
        self._cache = {}
        self.start_date = start_date
        self.end_date = end_date

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
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        val = self.get_contract_data(value=value, contract_type=contract_type)
        val = val.loc[start_date:end_date, :]
        val.dropna(how='all', axis=1, inplace=True)

        date_list = val.index.tolist()  # 所有交易日序列
        contract_list = val.columns.tolist()

        # 用于记录每一交易日的近月合约及次月合约的持仓比例
        position = pd.DataFrame(0.0, index=date_list, columns=contract_list)

        for date in date_list:
            print('linear_interpolation ' + date)

            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = self.rollover_code_judge(self.hedge_fut, date)

            # 近月合约的剩余到期日
            current_maturity = self.get_maturity(date=date, contract=current_contract)
            # 次月合约的剩余到期日
            next_maturity = self.get_maturity(date=date, contract=next_contract)

            # 近月合约持仓比例
            position.loc[date, current_contract] = min(current_maturity / (next_maturity - current_maturity), 1)
            # 次月合约持仓比例
            position.loc[date, next_contract] = 1 - position.loc[date, current_contract]

        port = (position * val).sum(axis=1)
        delta = (val.diff() * position.shift(1)).sum(axis=1)
        self._cache['linear_interpolation' + value] = port, delta, position
        return port, delta, position


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

        self.strategy = Strategy()

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

    def set_strategy(self, st):
        self.strategy = st
        self.strategy.anaData = self.anaData
        self.strategy.fut_port = self.fut_port

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

    def get_hedge_ratio(self, date, frequency=5, **kwargs):
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
        hedge_ratio = self.strategy.get_hedge_ratio(date, frequency, **kwargs)
        return hedge_ratio
        #
        # elif copy_type == 3:
        #     hedge_ratio = 1 + get_w_real(self.fut_port.delta_f, self.fut_port.delta_b, date, frequency)
        # elif copy_type == 4:
        #     hedge_ratio = 1 + get_w2_real(self.fut_port.delta_f, self.fut_port.delta_b, date, frequency)
        # elif copy_type == 5:
        #     w1 = 0.5 * get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     w2 = 0.5 * get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     hedge_ratio = 1 + w1 + w2
        # elif copy_type == 6:
        #     w1, w2 = get_portfolio_w(self.asset0, self.asset1, self.asset2, date)
        #     w1 = w1 * get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     w2 = w2 * get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     hedge_ratio = 1 + w1 + w2
        # if correct == 'randomforest':
        #     randomforest = self.model[0]
        #     variable = self.model[1]
        #     variable = variable.iloc[:, 1:]
        #     variable_index = variable.index.tolist()
        #     date_variable = variable.iloc[variable_index.index(date)-1, :]
        #     pred = randomforest.predict(date_variable.values.reshape(1, len(date_variable)))
        #     if copy_type == 1:
        #         w1 = get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     elif copy_type == 2:
        #         w1 = get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        #     if pred > 0:
        #         w = min(w1 + 0.1, 0.5)
        #     elif pred < 0:
        #         w = max(w1 - 0.1, -0.5)
        #     hedge_ratio = 1 + w
        # return hedge_ratio

    def run_bkt(self, frequency=-1, strategy=None):
        if strategy is not None:
            self.set_strategy(strategy)
        cache_key = self.strategy.name+str(frequency)
        ret = self._cache.get(cache_key, None)
        if ret is not None:
            return ret

        trade_date = self.get_trade_date(frequency=frequency)

        contract_list = self.fut_port.position.columns.tolist()
        contract_position = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)
        fut_pnl = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)
        tran_cost = pd.DataFrame(0.0, index=self.date_list, columns=contract_list)

        # 第一天期货持仓头寸多头头寸
        equity_position = int(self.cash / (self.anaData.equity_close[self.start_date] * self.multiplier))
        fut_position = int(equity_position * self.get_hedge_ratio(self.start_date))

        for date in self.date_list:
            if date in trade_date:
                fut_position = int(equity_position * self.strategy.get_hedge_ratio(date, frequency))
            contract_pitioson.loc[date, :] = self.fut_port.contract_position(date) * fut_position

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
        self._cache[cache_key] = npv, npv_nocost, npv_f
        return npv, npv_nocost, npv_f


class FuturePortGen(object):
    _cache = {}

    def __init__(self, anaData=None):
        self.anaData = anaData
        self.hedge_fut = None

        self.start_date = None
        self.end_date = None

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
                rollover_sign1 = ((hp_spread[-2:] <= past_spread[-2:]).all()) and (
                            past_spread[-2] < past_spread[-1])
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


class Strategy(object):

    def __init__(self):
        self.fut_port = None
        self.name = ''

        self.copy_type = None
        self.correct = None
        self.anaData = None

    def set_param(self, fut_port, ana_data):
        self.fut_port = fut_port
        self.anaData = ana_data

    def get_hedge_ratio(self, date, frequency, **kwargs):
        raise NotImplementedError('Strategy must implement.')

    def get_trade_date(self):
        pass


class StrategyN1(Strategy):
    """1:1等数量对冲"""
    def __init__(self):
        super(StrategyN1, self).__init__()
        self.name = 'N1'

    def get_hedge_ratio(self, date, frequency, **kwargs):
        hedge_ratio = 1
        return hedge_ratio


class Strategy00(Strategy):
    """1:1等市值对冲"""
    def __init__(self):
        super(Strategy00, self).__init__()
        self.name = '00'
    def get_hedge_ratio(self, date, frequency, **kwargs):
        hedge_ratio = self.anaData.equity_close[date] / self.fut_port.close(date)
        return hedge_ratio


class Strategy01(Strategy):
    def __init__(self):
        super(Strategy01, self).__init__()
        self.name = 'S1'

    def get_hedge_ratio(self, date, frequency, **kwargs):
        hedge_ratio = 1 + get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        return hedge_ratio


class Strategy02(Strategy):
    def __init__(self):
        super(Strategy02, self).__init__()
        self.name = 'S2'
    def get_hedge_ratio(self, date, frequency, **kwargs):
        hedge_ratio = 1 + get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        return hedge_ratio


class Strategy06(Strategy):
    def __init__(self, bkt_obj, fut_gen, frequency=-1):
        super(Strategy06, self).__init__()
        self.bkt_obj = bkt_obj
        self.fut_gen = fut_gen
        self.frequency = frequency
        self.name = '06'

        self._nav0 = None
        self._nav1 = None
        self._nav2 = None

    @property
    def nav0(self):
        if self._nav0 is None:
            self.bkt_obj.set_future(fut=self.fut_gen.get_main_future())
            self._nav0 = self.bkt_obj.run_bkt(strategy=StrategyN1())[0]
        return self._nav0

    @property
    def nav1(self):
        if self._nav1 is None:
            self.bkt_obj.set_future(fut=self.fut_gen.get_linear_future())
            self._nav1 = self.bkt_obj.run_bkt(frequency=self.frequency, strategy=Strategy01())[0]
        return self._nav1

    @property
    def nav2(self):
        if self._nav2 is None:
            self.bkt_obj.set_future(fut=self.fut_gen.get_linear_future())
            self._nav2 = self.bkt_obj.run_bkt(frequency=self.frequency, strategy=Strategy02())[0]
        return self._nav2

    def get_hedge_ratio(self, date, frequency, **kwargs):
        w1, w2 = get_portfolio_w(self._nav0, self._nav1, self._nav2, date)
        w1 = w1 * get_w(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        w2 = w2 * get_w2(self.fut_port.delta_f, self.fut_port.delta_b, date, 15)
        hedge_ratio = 1 + w1 + w2
        return hedge_ratio


class RFStrategy(Strategy):
    def __init__(self, strategy, classes='four', copy_type=1):
        super(RFStrategy, self).__init__()
        self.raw_st = strategy
        self.classes = classes
        self.copy_type = copy_type
        self.name = 'RF '+classes + ' copy_type: ' + str(copy_type)

        self.anaData = None
        self.fut_port = None
        self.frequency = None

    def set_param(self, fut_port, ana_data):
        self.raw_st.set_param(fut_port, ana_data)
        self.fut_port = fut_port
        self.anaData = ana_data

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
        # 筛掉向量当中的空值
        x_variable = x_variable.iloc[40:, :]
        s1y_variable = s1y_variable.iloc[40:, :]
        s2y_variable = s2y_variable.iloc[40:, :]

        return x_variable, s1y_variable, s2y_variable

    def get_hedge_ratio(self, date, frequency, **kwargs):
        x_variable, y1_variable, y2_variable = self.get_sample(frequency=frequency)
        if self.copy_type == 1:
            y_variable = y1_variable.loc[:, self.classes]
        elif self.copy_type == 2:
            y_variable = y2_variable.loc[:, self.classes]

        variable_index = x_variable.index.tolist()

        n = x_variable.iloc[max(variable_index.index(date) - 252, 0):variable_index.index(date) - 1, :]

        train_x = x_variable.iloc[max(variable_index.index(date) - 252, 0):variable_index.index(date) - 1, :]
        train_x = train_x.values.reshape(len(n), len(x_variable.T))
        test_x = x_variable.iloc[variable_index.index(date) - 1, :].values.reshape(1, len(x_variable.T))
        train_y = y_variable[max(variable_index.index(date) - 252, 0):variable_index.index(date) - 1]
        train_y = train_y.values.ravel()

        RF = RandomForestClassifier(n_estimators=300, random_state=0)
        RF.fit(train_x, train_y)
        pred = RF.predict(test_x)

        w1 = self.raw_st.get_hedge_ratio(date=date, frequency=frequency) - 1
        hedge_ratio = 1 + get_amend(classes=self.classes, pred=pred, w1=w1, copy_type=self.copy_type)
        return hedge_ratio

    def get_randomforest(self, frequency=1):
        df = self.get_sample(frequency=frequency)
        df2 = df.dropna(how='any')
        df2 = df2.loc[:self.start_date, :]
        y = df2.iloc[:, 0].values.reshape(-1, 1)
        x = df2.iloc[:, 1:].values.reshape(len(df2), len(df2.T) - 1)
        forest = RandomForestClassifier(n_estimators=300, random_state=0)
        forest.fit(x, y)
        return forest, df

class Strategybeta(Strategy):
    def __init__(self, bkt_obj, fut_gen, frequency=-1):
        super(Strategy06, self).__init__()
        self.bkt_obj = bkt_obj
        self.fut_gen = fut_gen
        self.name = 'beta'

    def get_hedge_ratio(self, date, frequency=5, **kwargs):
        f_close = self.anaData.linear_interpolation()[0]
        s_close = self.anaData.equity_close
        f_start = f_close.index[0]
        s_close = s_close[f_start:]
        all_tradeday = f_close.index.tolist()
        i_date = all_tradeday.index(date)
        f100 = f_close[max(i_date - 100, 0):i_date]
        s100 = s_close[max(i_date-100, 0):i_date]


        return hedge_ratio




