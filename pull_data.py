# DB에 저장할 데이터를 pulling하는 파일입니다.
# N의 값을 수정해 검색 대상 종목의 수를 조절할 수 있습니다.
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings
import yfinance as yf

warnings.filterwarnings(action='ignore')

def pull_data(std_date, date_start, N):
    """
    input
    - std_date : 기준일
    - date_start : 주가 정보 가져오기 시작할 날 시작일 std 기준 1년 이상 전으로 해주면 됨
    - N : 시가총액 상위 N개 종목
    output : data
    
    사용 예시
    - pull_data("20220325", "20210102", 1000)
    """

    date_list = [std_date]

    date_list_for_PER = date_list

    df = pd.read_csv('./data/{}_total_stock.csv'.format(std_date), encoding='euc-kr') # 한국거래소 > 정보데이터 시스템 > 전종목 시세

    # 시가총액 상위 N종목 코드 리스트
    top_code = df.sort_values(by='시가총액', ascending=False).head(N).종목코드.to_list()

    # 종목코드를 가지고 종목명을 찾을 수 있는 dictionary
    code_to_name_dic = df[['종목코드', '종목명']].set_index('종목코드').to_dict()['종목명']
    name_to_code_dic = df[['종목코드', '종목명']].set_index('종목명').to_dict()['종목코드']

    # KOSPI = fdr.DataReader('KS11', date_start)
    KOSPI = yf.download('^KS11', date_start[:4]+'-'+date_start[4:6] + '-' + date_start[6:])


    # 종목코드별 주가 데이터 dictionary에 저장
    stock_price_dic = {}
    for code in top_code:
        stock_price_dic[code] = fdr.DataReader(code, date_start)

    def date_add(date, month):
        '''date에 month만큼 더해준다. date는 20210102와 같은 꼴이나 datetime type으로 입력한다.'''
        if type(date) == str: # date가 datetime이 아니라 str 타입으로 들어온 경우
            date = datetime.strptime(date, '%Y%m%d')
        delta = timedelta(days=month*30.5)
        return date + delta


    def earning_rate(stock_price : list, start_date, end_date):
        '''start_date부터 end_date까지 stock_price데이터를 가지고 수익률을 구해주는 함수(종가 기준)'''
        possible_start_date = stock_price[stock_price.index >= start_date].index[0]
        possible_end_date = stock_price[stock_price.index <= end_date].index[-1]
        
        start_price = stock_price.loc[possible_start_date].Close
        end_price = stock_price.loc[possible_end_date].Close
        
        return (end_price - start_price) / start_price

    stock_price = stock_price_dic[top_code[0]]

    n = 12 # n개월 간의 주가 추이 확인

    def ER_list_return(stock_price, n=n, date_list=date_list):
        '''stock_price정보를 가지고 base_date 기준 n개월 전, m개월 후 수익률(Earning Rate)을 구해준다.
        return : 데이터로 사용할 수익률, target을 구할 때 사용할 수익률, base_date'''
        earning_rate_list = []
        base_date_list = []  # 수익률 기준일
        
        for date in date_list:
            try:
            
                date_start = date_add(date, -12) # 수정
                date_base = date
                
                ER_before = earning_rate(stock_price, date_start, date_base)
                
                earning_rate_list.append(ER_before)
                base_date_list.append(date_base)
                # print(date_start, date_base, ER_before)
                    
            except: # 종목코드가 바뀌거나 상장폐지돼서 주가 데이터가 없는 경우가 있는 것 같음. 이를 고려             
                break

        return earning_rate_list, base_date_list


    # DataFrame 만들기 위해 모든 종목들에 대해 시행해주기
    n_month_ER_data = []
    date_data = []
    code_data = []

    for code in top_code:
        stock_price = stock_price_dic[code]
        earning_rate_list, base_date_list = ER_list_return(stock_price)
        n_month_ER_data += earning_rate_list
        date_data += base_date_list
        code_data += [code]*len(base_date_list)

    data = pd.DataFrame({
                        'date' : date_data,
                        'code' : code_data,
                        f'{n}개월 간 수익률': n_month_ER_data,
                        })

    data['name'] = data.code.apply(lambda x: code_to_name_dic[x])

    KOSPI_n_earning, KOSPI_base_date = ER_list_return(KOSPI)

    KOSPI_data = pd.DataFrame({
                        'date' : KOSPI_base_date,
                        f'KOSPI {n}개월 간 수익률': KOSPI_n_earning,
                        })

    data = data.merge(KOSPI_data, on='date')

    # KOSPI 대비 수입률 Feature 만들기
    data[f'KOSPI대비 {n}개월 간 수익률'] = data[f'{n}개월 간 수익률'] - data[f'KOSPI {n}개월 간 수익률']
    data[f'{n}개월 간 KOSPI 이김'] = data[f'KOSPI대비 {n}개월 간 수익률'].apply(lambda x: 1 if x>=0 else 0)



    # PER 파일 불러와서 dic에 저장
    df_PER_dic = {}
    for date in date_list_for_PER:
        # 결측치는 0으로
        df_PER_dic[date] = pd.read_csv('./data/{}_PER_PBR.csv'.format(date), encoding='euc-kr')
        df_PER_dic[date][['PER', 'PBR']] = df_PER_dic[date][['PER', 'PBR']].fillna(10000)
        df_PER_dic[date][['배당수익률']] = df_PER_dic[date][['배당수익률']].fillna(0)

    def near_date_for_PER(date, date_list=date_list_for_PER):
        '''한국거래소에서 받은 PER를 데이터 누수(미래의 PER 사용;) 없이 사용하기 위해 date에 따라 적절한 사용가능 날짜를 date_list에서 뽑아 리턴.
        date는 datetime 타입'''
        for d in date_list:
            d = datetime.strptime(d, '%Y%m%d')
            if d - timedelta(days=30) <= date < d + timedelta(days=336):
                return d.strftime('%Y%m%d')
        return date_list[0]


    def search_for_PER(code, date):
        '''code와 date 조건에 맞는 데이터를 PER가 담긴 df에서 찾아줌'''
        if type(date) != str:
            date = near_date_for_PER(date)
        df = df_PER_dic[date]
        cond = df['종목코드'] == code
        
        if cond.sum() == 0:
            fake_data = [0] * len(df.columns)
            return pd.DataFrame([fake_data], columns=df.columns)
        return df[cond]


    temp_df_list = []
    for date, code in zip(data.date, data.code):
        temp_df_list.append(search_for_PER(code, date))
    data_PER = pd.concat(temp_df_list, ignore_index=True)

    data = pd.concat([ data, data_PER[['PER', 'PBR', '배당수익률']] ], axis=1)
    data['PER_inv'] = 1/data['PER']
    data['PBR_inv'] = 1/data['PBR']

    return data


# 이것만 조정해주고 한국거래소에서 std_date 기준 전종목 시세, PER 데이터 받아서 최상위에 있는 data 디렉토리에 넣어주면 됨
if __name__ == "__main__":
    std_date = '20211101' # 주식 기준일
    date_start = '20200110' # 1년의 데이터는 확보하기 위해서
    N = 1000 # 시가총액 상위 N 종목

    data = pull_data(std_date, date_start, N)

    print(data.head(10))
    # data.to_csv(f'stock_data_{std_date}_{N}.csv',encoding='euc-kr')
    data.to_csv(f'stock_data_{std_date}_{N}.csv',encoding='utf-8') # 이게 더 좋은듯?