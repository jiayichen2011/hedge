import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time, datetime
import re
import matplotlib.dates as mdate
import matplotlib
from sklearn import datasets, linear_model


class fut_hedge(object):
    def __init__(self, start_date='2018/1/3', end_date='2020/5/18', hedge_fut='IF'):
        self.hedge_fut = hedge_fut
        self.path=os.path.abspath('..')
        self.fut_close = pd.read_csv(self.path+'/datasets/Fut_ClosePrice.csv', index_col=0)
        self.equity_close = pd.read_excel(self.path+'/datasets/hs300.xlsx', index_col=0, encoding='unicode_escape')
        self.equity_close = self.equity_close.loc[:, 'CLOSE']
        self.fut_volume = pd.read_csv(self.path+'/datasets/Fut_TurnoverVolume.csv', index_col=0)
        self.start_date = start_date
        self.end_date = end_date
        self.basis = pd.read_csv(self.path+'/datasets/Fut_BasisValue.csv', index_col=0)
        self.index = self.fut_close.loc[start_date:end_date, :].index
        self.cash = 10000 * 10000

    def Linear_interpolation(self, data='close', start_date='2010/5/4', end_date='2020/6/19',contract_type='IF', T=30):
        '''返回两个参数，第一个是合成期货的收盘数据，第二个是合成期货的每期损益。
        data为close则返回收盘价的数据，data为basis则返回基差数据'''

        #判断合成数据的类型，基差抑或是收盘价
        if data == 'basis':
            df = pd.read_csv(self.path+'/datasets/Fut_BasisValue.csv', index_col=0)
        elif data == 'close':
            df = pd.read_csv(self.path+'/datasets/Fut_ClosePrice.csv', index_col=0)
        #判断合成期货的种类
        if contract_type == 'IF':
            IF_code = pd.read_csv(self.path+'/datasets/all_IF.csv', index_col=0)
            data = df[sorted(IF_code.iloc[:, 0])]
        elif contract_type == 'IC':
            IC_code = pd.read_csv(self.path+'/datasets/all_IC.csv', index_col=0)
            data= df[sorted(IC_code.iloc[:, 0])]
        elif contract_type == 'IH':
            IH_code = pd.read_csv(self.path+'/datasets/all_IH.csv', index_col=0)
            data = df[sorted(IH_code.iloc[:, 0])]

        data = data.loc[start_date:end_date, :]
        data.dropna(how='all', axis=1, inplace=True)
        tradeday=data.index.tolist()#所有交易日序列
        maturity = pd.read_csv(self.path+'/datasets/IF_maturity.csv', index_col=0)#记录了所有合约的剩余到期日
        last_tradeday=pd.read_csv(self.path+'/datasets/IF_last_tradeday.csv',index_col=0)#记录了所有合约的最后交易日

        df=pd.DataFrame(0.0,index=data.index,columns=data.columns)#用于记录每一交易日的近月合约及次月合约的持仓比例
        f=pd.DataFrame(0.0,index=data.index,columns=data.columns)#用于记录每一交易日的损益
        linear=pd.Series(0.0,index=data.index)#用于记录插值后的数据
        for i in range(len(data)):
            current_tradeday = tradeday[i] # 当前交易日
            current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
            #判断当前交易日的近月合约及次月合约
            current_contract, next_contract= rollover_code_judge(self.hedge_fut, last_tradeday,current_tradeday_timestamp)

            recent_last_tradeday = last_tradeday.loc[current_contract, 'last_tradeday']  # 近月合约的交割日
            recent_last_tradeday_timestamp=datetime.datetime.strptime(recent_last_tradeday,"%Y/%m/%d")
            current_maturity=maturity.loc[current_tradeday,current_contract]#近月合约的剩余到期日
            next_maturity=maturity.loc[current_tradeday,next_contract]#次月合约的剩余到期日
            if current_tradeday_timestamp == recent_last_tradeday_timestamp:#当前日期为展期日
                df[current_contract][current_tradeday] = 0#近月合约持仓比例为0
                df[next_contract][current_tradeday] = 1#次月合约持仓比例为100%
            else:
                df[current_contract][current_tradeday] = (next_maturity-T)/(next_maturity-current_maturity)#近月合约持仓比例
                df[next_contract][current_tradeday] = (T-current_maturity)/(next_maturity-current_maturity)#远月合约持仓比例
            contract1_value=df[current_contract][current_tradeday]*data.loc[current_tradeday,current_contract]#近月合约价格
            contract2_value=df[next_contract][current_tradeday]*data.loc[current_tradeday,next_contract]#远月合约价格
            linear[current_tradeday]=contract1_value+contract2_value

        #计算每一交易日的损益
        for contract in df.columns:
            df_position = df[contract]
            contract_close = data.loc[:, contract]
            f[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)))[1:]
        delta=f.sum(axis=1)
        return linear, delta,df

    def Current_month_basis(self, data='close',start_date='2010/5/4', end_date='2020/6/19', contract_type='IF'):
        '''data为close则返回收盘价的数据，data为basis则返回基差数据'''
        if data == 'basis':
            df = self.basis
        elif data == 'close':
            df = self.fut_close

        if contract_type == 'IF':
            IF_code = pd.read_csv(self.path+'/datasets/all_IF.csv', index_col=0)
            data = df[sorted(IF_code.iloc[:, 0])]
        elif contract_type == 'IC':
            IC_code = pd.read_csv(self.path+'/datasets/all_IC.csv', index_col=0)
            data = df[sorted(IC_code.iloc[:, 0])]
        elif contract_type == 'IH':
            IH_code = pd.read_csv(self.path+'/datasets/all_IH.csv', index_col=0)
            data = df[sorted(IH_code.iloc[:, 0])]
        data = data.loc[start_date:end_date, :]
        data.dropna(how='all', axis=1, inplace=True)
        tradeday = data.index.tolist()  # 所有交易日序列
        maturity = pd.read_csv(self.path + '/datasets/IF_maturity.csv', index_col=0)  # 记录了所有合约的剩余到期日
        last_tradeday = pd.read_csv(self.path + '/datasets/IF_last_tradeday.csv', index_col=0)  # 记录了所有合约的最后交易日

        df = pd.DataFrame(0.0, index=data.index, columns=data.columns)  # 用于记录每一交易日的近月合约的持仓
        f = pd.DataFrame(0.0, index=data.index, columns=data.columns)  # 用于记录每一交易日的损益
        current_month = pd.Series(0.0, index=data.index)  # 用于记录近月合约的连续收盘数据

        for i in range(len(data)):
            current_tradeday = tradeday[i]  # 当前交易日
            current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = rollover_code_judge(self.hedge_fut, last_tradeday,current_tradeday_timestamp)

            recent_last_tradeday = last_tradeday.loc[current_contract, 'last_tradeday']  # 近月合约的交割日
            recent_last_tradeday_timestamp = datetime.datetime.strptime(recent_last_tradeday, "%Y/%m/%d")
            current_maturity = maturity.loc[current_tradeday, current_contract]  # 近月合约的剩余到期日
            next_maturity = maturity.loc[current_tradeday, next_contract]  # 次月合约的剩余到期日
            if current_tradeday_timestamp == recent_last_tradeday_timestamp:  # 当前日期为展期日
                df[next_contract][current_tradeday] = 1.0
            else:
                df[current_contract][current_tradeday] = 1.0
            # 近月合约价格
            contract_value = df[current_contract][current_tradeday] * data.loc[current_tradeday, current_contract]
            current_month[current_tradeday] = contract_value

        # 计算每一交易日的损益
        for contract in df.columns:
            df_position = df[contract]
            contract_close = data.loc[:, contract]
            f[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)))[1:]
        delta = f.sum(axis=1)
        return current_month, delta

    def Next_month_basis(self, data='close',start_date='2010/5/4', end_date='2020/6/19',contract_type='IF'):
        '''data为close则返回收盘价的数据，data为basis则返回基差数据'''
        if data == 'basis':
            df = self.basis
        elif data == 'close':
            df = self.fut_close
        if contract_type == 'IF':
            IF_code = pd.read_csv(self.path+'/datasets/all_IF.csv', index_col=0)
            data = df[sorted(IF_code.iloc[:, 0])]
        elif contract_type == 'IC':
            IC_code = pd.read_csv(self.path+'/datasets/all_IC.csv', index_col=0)
            data = df[sorted(IC_code.iloc[:, 0])]
        elif contract_type == 'IH':
            IH_code = pd.read_csv(self.path+'/datasets/all_IH.csv', index_col=0)
            data = df[sorted(IH_code.iloc[:, 0])]
        data = data.loc[start_date:end_date, :]
        data.dropna(how='all', axis=1, inplace=True)

        tradeday = data.index.tolist()  # 所有交易日序列

        last_tradeday = pd.read_csv(self.path + '/datasets/IF_last_tradeday.csv', index_col=0)  # 记录了所有合约的最后交易日

        df = pd.DataFrame(0.0, index=data.index, columns=data.columns)  # 用于记录每一交易日的近月合约的持仓
        f = pd.DataFrame(0.0, index=data.index, columns=data.columns)  # 用于记录每一交易日的损益
        next_month = pd.Series(0.0, index=data.index)  # 用于记录次月合约的连续收盘数据

        for i in range(len(data)):
            current_tradeday = tradeday[i]  # 当前交易日
            current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
            # 判断当前交易日的近月合约及次月合约
            current_contract, next_contract = rollover_code_judge(self.hedge_fut, last_tradeday,current_tradeday_timestamp)
            df[next_contract][current_tradeday] = 1

            # 次月合约价格
            contract_value = df[next_contract][current_tradeday] * data.loc[current_tradeday, next_contract]
            next_month[current_tradeday] = contract_value

        # 计算每一交易日的损益
        for contract in df.columns:
            df_position = df[contract]
            contract_close = data.loc[:, contract]
            f[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)))[1:]
        delta = f.sum(axis=1)
        return next_month, delta

    def fundamental_hedge(self):
        '''主力合约基础策略'''
        #获取每个交易日的主力合约
        hedge_contract = []
        for x in self.fut_close.columns:
            if re.match(self.hedge_fut + '[12]', x):
                hedge_contract.append(x)
        hedge_fut_volume = self.fut_volume[sorted(hedge_contract)]
        hedge_fut_volume = hedge_fut_volume.loc[self.start_date:self.end_date, :]
        hedge_contract_code = hedge_fut_volume.idxmax(axis=1)#每个交易日的主力合约(交易量最大)

        tradeday = self.index.tolist()#所有交易日序列
        current_tradeday = tradeday[0]#当前交易日
        commission_rate = 0.0028

        current_contract=hedge_contract_code[current_tradeday]#第一天持仓的期货
        previous_contract=np.nan #上一交易日持仓的期货，用于判定是否需要展期
        fut_position = int(self.cash / (self.fut_close.loc[self.start_date,current_contract ] * 300))#第一天期货持仓头寸
        equity_position = int(self.cash / (self.equity_close[self.start_date] * 300))#多头头寸

        df=pd.DataFrame(0.0,index=tradeday,columns=self.fut_close.columns)#用于记录各个合约持仓量
        f=pd.DataFrame(0.0,index=tradeday,columns=self.fut_close.columns)#用于记录delta F
        tran_cost=pd.DataFrame(0.0,index=tradeday,columns=self.fut_close.columns)

        for i in range(len(tradeday)):
            current_tradeday = tradeday[i]  #当前交易日
            current_contract=hedge_contract_code[current_tradeday]#该日持仓的期货
            if i!=0:
                previous_contract=hedge_contract_code[i-1]#上一交易日持仓的期货

            if current_contract!=previous_contract:#上一交易日持仓的期货与当日持仓期货不同，需要展期
                equity_value=equity_position*self.equity_close[current_tradeday]*300
                fut_position=int(equity_value/(self.fut_close.loc[current_tradeday,current_contract]*300))

            df[current_contract][current_tradeday] = fut_position#记录了该日持仓的期货及持仓数量

        #计算delta F以及交易成本
        for contract in df.columns:
            df_position = df[contract]
            contract_close = self.fut_close.loc[:, contract]
            f[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)) * 300)[1:]
            tran_cost[contract] = (abs(300 * commission_rate * (df_position - df_position.shift(1)) * contract_close))

        s = (self.equity_close - self.equity_close.shift(1)) * equity_position * 300 #delta S

        pnl = s - f.sum(axis=1) #不考虑交易成本的每期损益
        cum_pnl = pnl.cumsum()
        npv_nocost = 1 + cum_pnl / self.cash
        pnl = pnl - tran_cost.sum(axis=1)
        cum_pnl = pnl.cumsum()
        npv = 1 + cum_pnl / self.cash
        return npv,npv_nocost

    def fundamental3(self):
        '''合成合约基础策略'''
        tradeday = self.index.tolist()
        last_tradeday = pd.read_csv(self.path+'/datasets/IF_last_tradeday.csv', index_col=0)
        f_linear, delta_f, f_position = self.Linear_interpolation()

        current_tradeday = self.start_date
        current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
        current_contract, next_contract = rollover_code_judge(self.hedge_fut, last_tradeday, current_tradeday_timestamp)
        equity_position = int(self.cash / (self.equity_close[self.start_date] * 300))
        equity_value = equity_position * self.equity_close[self.start_date] * 300
        fut_position = equity_value / (self.fut_close.loc[current_tradeday, current_contract] * 300)

        commission_rate = 0.0028
        df = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)
        f = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)
        tran_cost = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)


        for i in range(len(self.equity_close)):
            current_tradeday = tradeday[i]  # 当前日期
            current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
            current_contract, next_contract = rollover_code_judge(self.hedge_fut, last_tradeday,current_tradeday_timestamp)
            recent_last_tradeday = last_tradeday.loc[current_contract, 'last_tradeday']  # 当月合约的交割日
            recent_last_tradeday_timestamp = datetime.datetime.strptime(recent_last_tradeday, "%Y/%m/%d")

            if current_tradeday_timestamp == recent_last_tradeday_timestamp:
                #当前日期为近月合约的最后交割日
                equity_value = equity_position * self.equity_close[current_tradeday] * 300
                fut_position = equity_value / (self.fut_close.loc[current_tradeday, current_contract] * 300)
            if current_tradeday_timestamp == recent_last_tradeday_timestamp:
                #近月合约全部平仓。并将头寸转移至次月合约
                df[current_contract][current_tradeday] = 0
                df[next_contract][current_tradeday] = int(fut_position)
                # f_postion记录了每一交易日合成合约中近月合约和次月合约的比例
            else:
                df[current_contract][current_tradeday] = round(fut_position * f_position[current_contract][current_tradeday])
                df[next_contract][current_tradeday] = round(fut_position * f_position[next_contract][current_tradeday])

        #计算每一交易日的损益
        for contract in df.columns:
            df_position = df[contract]
            contract_close = self.fut_close.loc[:, contract]
            f[contract] = (df_position.shift(1) * (contract_close - contract_close.shift(1)) * 300)[1:]
            tran_cost[contract] = (abs(300 * commission_rate * (df_position - df_position.shift(1)) * contract_close))

        s = (self.equity_close - self.equity_close.shift(1)) * equity_position * 300
        pnl = s - f.sum(axis=1)
        cum_pnl = pnl.cumsum()
        npv_nocost = 1 + cum_pnl / self.cash
        pnl = pnl - tran_cost.sum(axis=1)
        cum_pnl = pnl.cumsum()
        npv = 1 + cum_pnl / self.cash
        npv_f=1 + (f.sum(axis=1)).cumsum() / self.cash
        return npv, npv_nocost, npv_f

    def copy2_hs300(self,copy_type, frequency):
        '''

        :param copy_type: 额外对冲比例w，1：delta B/delta F； 2：cov(delta B,delta F)/var(delta F)；3、4:用上未来数据
        :param frequency: 换仓频率 day
        :return: npv策略净值，npv_nocost:无交易成本下策略净值，npv_f:空头端期货的净值
        '''
        tradeday = self.index.tolist()#交易日序列
        all_tradeday = self.all_tradeday
        last_tradeday = pd.read_csv(self.path+'/datasets/IF_last_tradeday.csv', index_col=0)#所有合约的最后交易日
        f_linear, delta_f, f_position = self.Linear_interpolation()#合成的delta F
        b_linear, delta_b, b_position = self.Linear_interpolation('basis')#合成的delta B


        current_tradeday = self.start_date#交易开始第一天

        #判断对冲比例w的计算方式，
        if copy_type == 1:
            hedge_w = get_w(delta_f, delta_b, all_tradeday, current_tradeday, 15)
        elif copy_type == 2:
            hedge_w = get_w2(delta_f, delta_b, all_tradeday, current_tradeday, 15)
        elif copy_type == 3:
            hedge_w = get_w_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency)
        elif copy_type == 4:
            hedge_w = get_w2_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency)



        equity_position = self.cash / (self.equity_close[self.start_date] * 300)#多头头寸
        fut_position = equity_position * (1 + hedge_w)#期货头寸
        equity_position = int(equity_position)#取整

        commission_rate = 0.0028

        df = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)#用于记录每一交易日持仓的期货及其头寸数量
        f = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)#用于记录期货端每一期损益
        tran_cost = pd.DataFrame(0.0, index=tradeday, columns=self.fut_close.columns)#交易成本

        rollover_sign = False#展期信号
        position_days = 0#记录已经持仓多少天
        for i in range(len(tradeday)):
            current_tradeday = tradeday[i]  # 当前日期
            current_tradeday_timestamp = datetime.datetime.strptime(current_tradeday, "%Y/%m/%d")
            #当前日期下近月合约以及次月合约
            current_contract, next_contract = rollover_code_judge(self.hedge_fut, last_tradeday,current_tradeday_timestamp)

            recent_last_tradeday = last_tradeday.loc[current_contract, 'last_tradeday']  # 当月合约的交割日
            recent_last_tradeday_timestamp = datetime.datetime.strptime(recent_last_tradeday, "%Y/%m/%d")
            position_days += 1
            #持仓天数到了换仓期，则要对额外对冲比例w进行重新计算
            if position_days == frequency:
                rollover_sign = True
                position_days = 0
            if rollover_sign:
                if copy_type == 1:
                    hedge_w = get_w(delta_f, delta_b, all_tradeday, current_tradeday, 15)
                elif copy_type == 2:
                    hedge_w = get_w2(delta_f, delta_b, all_tradeday, current_tradeday, 15)
                elif copy_type == 3:
                    hedge_w = get_w_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency)
                elif copy_type == 4:
                    hedge_w = get_w2_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency)

                fut_position = equity_position * (1 + hedge_w)#期货头寸
                fut_position = round(fut_position)
                rollover_sign = False
            if current_tradeday_timestamp == recent_last_tradeday_timestamp:
                #近月合约的最后交割日那天要把近月合约全部平仓，并把平仓的头寸转移至次月合约
                df[current_contract][current_tradeday] = 0
                df[next_contract][current_tradeday] = fut_position
                #f_position记录了每一交易日合成合约当中近月合约及次月合约的比例
            else:
                df[current_contract][current_tradeday] = round(fut_position * f_position[current_contract][current_tradeday])
                df[next_contract][current_tradeday] = round(fut_position * f_position[next_contract][current_tradeday])

        #计算每一期的损益
        for contract in df.columns:
            df_position = df[contract]
            contract_close = self.fut_close.loc[:, contract]
            delta_close = contract_close - contract_close.shift(1)
            f[contract] = (df_position.shift(1) * (delta_close) * 300)[1:]
            tran_cost[contract] = (abs(300 * commission_rate * (df_position - df_position.shift(1)) * contract_close))

        s = (self.equity_close - self.equity_close.shift(1)) * equity_position * 300
        pnl = s - f.sum(axis=1)
        cum_pnl = pnl.cumsum()
        npv_nocost = 1 + cum_pnl / self.cash
        pnl = pnl - tran_cost.sum(axis=1)
        cum_pnl = pnl.cumsum()
        npv = 1 + cum_pnl / self.cash
        npv_f = 1 + (f.sum(axis=1).cumsum() / self.cash)
        return npv, npv_nocost, npv_f

def get_w(delta_f, delta_b, all_tradeday, current_tradeday, frequency=30):
    w = -(delta_b / delta_f)
    w_30 = w[all_tradeday.index(current_tradeday) - frequency:all_tradeday.index(current_tradeday)]#过去30天的数据
    w_30 = sigmoid(w_30)#去极值
    w_30 = hp_filter(w_30)#HP滤波
    hedge_w = w_30[-1]
    #对w限定取值范围[-0.5,0.5]
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w
def get_w_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency=15):
    delta_b_roll = delta_b[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    delta_f_roll = delta_f[all_tradeday.index(current_tradeday):all_tradeday.index(current_tradeday) + frequency].sum()
    w = -(delta_b_roll / delta_f_roll)
    hedge_w = w
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w
def get_w2(delta_f, delta_b, all_tradeday, current_tradeday, frequency=30):
    delta_f_mean = delta_f.rolling(30).mean()
    delta_b_mean = delta_b.rolling(30).mean()
    bf = (delta_f * delta_b).rolling(30).mean()
    cov = bf - delta_b_mean * delta_f_mean
    var = delta_f.rolling(30).var()

    w = cov / var
    w_30 = w[all_tradeday.index(current_tradeday) - frequency:all_tradeday.index(current_tradeday)]#取过去三十天的数据
    w_30 = hp_filter(w_30)#HP滤波
    hedge_w = w_30[-1]
    if hedge_w > 0.5:
        hedge_w = 0.5
    elif hedge_w < -0.5:
        hedge_w = -0.5
    return hedge_w
def get_w2_real(delta_f, delta_b, all_tradeday, current_tradeday, frequency=30):
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

    return pd.Series(alt_series,index=series.index)
def sigmoid(df,n=1.4826):
    df2=df.copy()
    median = df.quantile(0.5)
    new_median = ((df - median).abs()).quantile(0.50)
    max_range = median + 3*n*new_median
    min_range = median - 3*n*new_median
    df2[df2>max_range]=n*median/(1+np.exp(df2[df2>max_range]))+3*n*new_median
    df2[df2<min_range]=-(n*median/(1+np.exp(df2[df2<min_range]))+3*n*new_median)
    return df2
def rollover_code_judge(hedge_fut,last_tradeday,current_date_timestamp):
    '''判断当前日期的近月合约及次月合约的函数'''
    year=current_date_timestamp.year
    month=current_date_timestamp.month
    if month < 8:
        contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
        contract2 = hedge_fut + str(year)[2:] + '0' + str(month + 1)
        contract3 = hedge_fut + str(year)[2:] + '0' + str(month + 2)
    elif month==8 :
        contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
        contract2 = hedge_fut + str(year)[2:] + '0'+str(month+1)
        contract3 = hedge_fut + str(year)[2:] + str(month + 2)
    elif month==9:
        contract1 = hedge_fut + str(year)[2:] + '0' + str(month)
        contract2 = hedge_fut + str(year)[2:] + str(month+1)
        contract3 = hedge_fut + str(year)[2:] + str(month + 2)
    elif month > 9 and month < 11:
        contract1 = hedge_fut + str(year)[2:] + str(month)
        contract2 = hedge_fut + str(year)[2:] + str(month+1)
        contract3 = hedge_fut + str(year)[2:] + str(month + 2)
    elif month ==11:
        contract1 = hedge_fut + str(year)[2:] + str(month)
        contract2 = hedge_fut + str(year)[2:] + str(month+1)
        contract3 = hedge_fut + str(year+1)[2:] + '01'
    elif month==12:
        contract1 = hedge_fut + str(year)[2:] + str(month)
        contract2 = hedge_fut + str(year+1)[2:] + '01'
        contract3 = hedge_fut + str(year+1)[2:] + '02'
    contract1_last_tradeday = last_tradeday.loc[contract1, 'last_tradeday']
    contract1_last_tradeday=datetime.datetime.strptime(contract1_last_tradeday,'%Y/%m/%d')
    if current_date_timestamp <= contract1_last_tradeday:
        current_contract = contract1
        next_contract=contract2
    else:
        current_contract = contract2
        next_contract=contract3
        roll_contract=np.nan
    return current_contract,next_contract
class drawing(object):
    ''' 1为基础策略收益图
        2为期现货走势图
        3为deltaB/deltaF散点图
        4为deltaB/deltaF走势图'''
    def __init__(self):
        self.object1 = fut_hedge()
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        matplotlib.rcParams['axes.unicode_minus'] = False
        self.f_linear,self.delta_f,df=self.object1.Linear_interpolation()
        self.b_linear, self.delta_b,df = self.object1.Linear_interpolation(data='basis')
    def drawing1(self):
        '''基础策略收益图'''

        npv, npv_nocost = self.object1.fundamental_hedge()
        x = npv_nocost.index
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range('2010/1/6', '2020/7/16', freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        plt.plot(x, npv_nocost)
        plt.title('无交易成本 基础策略', fontsize=18)
        plt.show()

    def drawing2(self,current=True,next=True):
        '''期现货走势对比图'''
        equity_close = pd.read_excel('D://zsfund/data/hs300.xlsx', index_col=0, encoding='unicode_escape')
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range('2010/1/6', '2020/7/16', freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        f0,delta_f0=self.object1.Current_month_basis('close')
        f1,delta_f1=self.object1.Next_month_basis('close')
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
        delta_f0[start_date]=0
        cumsum_f0=f0_position*delta_f0[start_date:end_date].cumsum()

        f1_position = 1 / f1[start_date]
        delta_f1[start_date]=0
        cumsum_f1=f1_position*delta_f1[start_date:end_date].cumsum()

        x = delta_s[start_date:end_date].index
        plt.plot(x, cumsum_f, linewidth=1)
        plt.plot(x, cumsum_s, linewidth=1)
        plt.legend(['合成股指期货','hs300'],fontsize=20)
        if current:
            plt.plot(x,cumsum_f0)
            plt.legend(['合成股指期货','hs300','近月合约'],fontsize=20)
        if next:
            plt.plot(x,cumsum_f1)
            plt.legend(['合成股指期货','hs300','次月合约'],fontsize=20)
        if current and next:
            plt.legend(['合成股指期货','hs300','近月合约','次月合约',],fontsize=20)


        plt.show()

    def drawing3(self):
        '''delta B/delta F 散点图'''
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
        plt.xticks(pd.date_range('2010/1/6', '2020/7/16', freq='1y'), rotation=30, fontsize=18)
        plt.yticks(fontsize=15)
        plt.scatter(self.delta_f, self.delta_b, facecolor='g', linewidths=0.1)
        plt.xlabel('delta F',fontsize=18)
        plt.ylabel('delta B',fontsize=18)
        plt.show()

    def drawing4(self,start_date='2010/4/23',end_date='2020/6/30',window=[1],filter=True,linestyle='plot'):
        '''delta F/delta B 走势图'''

        #window需要传入一个list。例如，window=[1,5]，则画出窗口期为一天以及五天的走势图
        #filter为true，则会对数据去极值处理，否则不会对数据去极值
        #linstyle为画图类型，plot为折线图，bar为柱状图
        #start_date 以及 end_date可以选定画图数据的区间，如果报错说明日期不在交易日，日期上下微调几天即可
        fig, axes = plt.subplots(len(window), 1, figsize=(15, 10))
        x=[datetime.datetime.strptime(d,"%Y/%m/%d") for d in self.delta_b[start_date:end_date].index]
        for i,j in zip(window,range(len(window))):
            y_f = self.delta_f.rolling(i).sum()[start_date:end_date]
            y_b = self.delta_b.rolling(i).sum()[start_date:end_date]
            if filter:
                Y=sigmoid(y_b / y_f)
            else:
                Y=(y_b/y_f)
            if len(window)==1:
                if linestyle=='plot':
                    plt.plot(x,Y)
                elif linestyle=='bar':
                    plt.bar(x,Y)
                plt.legend(['delta B/delta F '+str(i)+'days'],fontsize=12,loc='upper right')
            elif len(window)>1:
                if linestyle=='plot':
                    Y.plot(ax=axes[j])
                elif linestyle=='bar':
                    Y.bar(ax=axes[j])
                axes[j].legend(['delta B/delta F ' + str(i) + 'days'], fontsize=12, loc='upper right')

        plt.show()


if __name__ == '__main__':
    draw=drawing()
    draw.drawing2()




