#https://www.kaggle.com/vijipai/lesson-7-constrained-portfolio-optimization
#해당 Optimization Tool은 위 사이트의 코드를 참조하여 일부 변형하였습니다.

import pandas as pd
import numpy as np
from scipy import optimize

stock_info = pd.read_csv('./data/weight.csv'
                     , index_col=0
                     )
#종목의 비중값과 거래정지 여부

p = pd.read_csv('./data/price.csv'
                    , index_col=0
                    , parse_dates=True
                    )
#종목 가격 시계열

halted_weight = stock_info[stock_info.Halt==1].Weight.sum()
max_weight = 1 - halted_weight
#거래정지된 종목의 비중값은 조정이 불가능하므로 portfolio max weight에서 제외

weight = stock_info[stock_info.Halt==0].Weight
p = p[weight.index]
#거래정지된 종목을 제외한 비중값과 가격 시계열

p_return = p.pct_change().iloc[1:]
p_mean = p_return.mean()
p_cov = p_return.cov()
#평균값과 분산 행렬 계산

def PortfolioOptimizer(MeanReturns, CovarReturns, MaxWeight, InitialWeight, ActiveBP):
    #최적화를 원하는 포트폴리오 값을 아래 f에 입력한다. 해당 목적함수는 Sharpe Maximizer이다
    def f(x, MeanReturns, CovarReturns):
        PortfolioVariance = np.matmul(np.matmul(x, CovarReturns), x.T)
        PortfolioSTDEV = np.sqrt(PortfolioVariance)
        PortfolioExpReturn = np.matmul(np.array(MeanReturns), x.T)
        func = - PortfolioExpReturn / PortfolioSTDEV #scipy는 minimize 기능밖에 없으므로 음값으로 전환
        return func

    xinit = InitialWeight

    #거래정지된 종목의 비중을 제외한 만큼의 비중값을 사용하도록 제한
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - MaxWeight})

    #종목별 비중값의 상하한선을 +- ActiveBP로 조정
    lowerbound = InitialWeight.apply(lambda x: max(x - ActiveBP/10000, 0)).values
    upperbound = InitialWeight.apply(lambda x: min(x + ActiveBP/10000, 1)).values
    bnds = np.vstack([lowerbound, upperbound]).T

    opt = optimize.minimize(f, x0=xinit, args=(MeanReturns, CovarReturns), \
                            method='SLSQP', bounds=bnds, constraints=cons, \
                            tol=10 ** -5)

    print(opt.message)

    return opt

def SharpeCalculator(MeanReturns, CovarReturns, Weight):
    PortfolioVariance = np.matmul(np.matmul(Weight, CovarReturns), Weight.T)
    PortfolioSTDEV = np.sqrt(PortfolioVariance)
    PortfolioExpReturn = np.matmul(np.array(MeanReturns), Weight.T)
    func = PortfolioExpReturn / PortfolioSTDEV  # scipy는 minimize 기능밖에 없으므로 음값으로 전환
    return func

result = PortfolioOptimizer(MeanReturns=p_mean
                            , CovarReturns=p_cov
                            , MaxWeight=max_weight
                            , InitialWeight=weight
                            , ActiveBP=20)
#최적화 진행

resultweight = pd.Series(result.x, index=weight.index)
print(resultweight)
#최적화된 비중값을 얻었다.

prev_sharpe = SharpeCalculator(p_mean, p_cov, weight)
curr_sharpe = SharpeCalculator(p_mean, p_cov, resultweight)
print('최적화 위험대비 기대수익률은 {}로 기존 {} 대비 {} 개선'
      .format(round(curr_sharpe,4), round(prev_sharpe,4), round(curr_sharpe - prev_sharpe,4)))