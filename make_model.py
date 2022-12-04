import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

#############################################################


def date_add(date, month):
    '''date에 month만큼 더해준다. date는 20210102와 같은 꼴이나 datetime type으로 입력한다.'''
    if type(date) == str:  # date가 datetime이 아니라 str 타입으로 들어온 경우
        date = datetime.strptime(date, '%Y%m%d')
    delta = timedelta(days=month*30.5)

    return date + delta


def earning_rate(stock_price: list, start_date, end_date):
    '''start_date부터 end_date까지 stock_price데이터를 가지고 수익률을 구해주는 함수(종가 기준)'''
    possible_start_date = stock_price[stock_price.index >= start_date].index[0]
    possible_end_date = stock_price[stock_price.index <= end_date].index[-1]

    start_price = stock_price.loc[possible_start_date].Close
    end_price = stock_price.loc[possible_end_date].Close

    return (end_price - start_price) / start_price


def prepare_data(date_list, date_list_for_PER, DATA_PATH, N, n, m):
    date_start = date_list[0]

    df = pd.read_csv(
        f'{DATA_PATH}/{date_start}_total_stock.csv', encoding='euc-kr')

    # 시가총액 상위 N종목 코드 리스트
    top_code = df.sort_values(
        by='시가총액', ascending=False).head(N).종목코드.to_list()

    # KOSPI = fdr.DataReader('KS11', date_start) # fdr이 안 돼서 yf 사용
    KOSPI = yf.download(
        '^KS11', date_start[:4]+'-'+date_start[4:6] + '-' + date_start[6:])

    # 종목코드를 가지고 종목명을 찾을 수 있는 dictionary
    code_to_name_dic = df[['종목코드', '종목명']].set_index('종목코드').to_dict()['종목명']

    # 종목코드별 주가 데이터 dictionary에 저장
    stock_price_dic = {}
    for code in top_code:
        stock_price_dic[code] = fdr.DataReader(code, date_start)

    # DataFrame 만들기 위해 모든 종목들에 대해 시행해주기
    n_month_ER_data = []
    m_month_ER_data = []
    date_data = []
    code_data = []

    def ER_list_return(stock_price, n=n, m=m, date_list=date_list):
        '''stock_price정보를 가지고 base_date 기준 n개월 전, m개월 후 수익률(Earning Rate)을 구해준다.
        return : 데이터로 사용할 수익률, target을 구할 때 사용할 수익률, base_date'''
        earning_rate_list = []
        target_list = []
        base_date_list = []  # 수익률 기준일

        periods = 12//n  # n = 2개월이면 12개월을 6개로 쪼개는 식
        for date in date_list:
            try:
                for period in range(periods):
                    date_begin = date_add(date, period*n)
                    date_end = date_add(date_begin, n)

                    ER_before = earning_rate(stock_price, date_begin, date_end)
                    ER_after = earning_rate(
                        stock_price, date_end, date_add(date_end, m))

                    earning_rate_list.append(ER_before)
                    target_list.append(ER_after)
                    base_date_list.append(date_end)

            except:  # 종목코드가 바뀌거나 상장폐지돼서 주가 데이터가 없는 경우가 있는 것 같음. 이를 고려
                break

        return earning_rate_list, target_list, base_date_list

    for code in top_code:
        stock_price = stock_price_dic[code]
        earning_rate_list, target_list, base_date_list = ER_list_return(
            stock_price)
        n_month_ER_data += earning_rate_list
        m_month_ER_data += target_list
        date_data += base_date_list
        code_data += [code]*len(base_date_list)

    data = pd.DataFrame({
                        'date': date_data,
                        'code': code_data,
                        f'{n}개월 간 수익률': n_month_ER_data,
                        f'{m}개월 후 수익률': m_month_ER_data
                        })

    data['name'] = data.code.apply(lambda x: code_to_name_dic[x])

    KOSPI_n_earning, KOSPI_m_earning, KOSPI_base_date = ER_list_return(KOSPI)

    KOSPI_data = pd.DataFrame({
        'date': KOSPI_base_date,
        f'KOSPI {n}개월 간 수익률': KOSPI_n_earning,
        f'KOSPI {m}개월 후 수익률': KOSPI_m_earning
    })

    data = data.merge(KOSPI_data, on='date')

    # KOSPI 대비 수입률 Feature 만들기
    data[f'KOSPI대비 {n}개월 간 수익률'] = data[f'{n}개월 간 수익률'] - \
        data[f'KOSPI {n}개월 간 수익률']
    data[f'KOSPI대비 {m}개월 후 수익률'] = data[f'{m}개월 후 수익률'] - \
        data[f'KOSPI {m}개월 후 수익률']
    data[f'{n}개월 간 KOSPI 이김'] = data[f'KOSPI대비 {n}개월 간 수익률'].apply(
        lambda x: 1 if x >= 0 else 0)
    data[f'KOSPI보다 많이 오름'] = data[f'KOSPI대비 {m}개월 후 수익률'].apply(
        lambda x: 1 if x >= 0 else 0)

    # PER 파일 불러와서 dic에 저장
    df_PER_dic = {}
    for date in date_list_for_PER:
        # 결측치는 0으로
        df_PER_dic[date] = pd.read_csv(
            f'{PATH}/{date}_PER_PBR.csv', encoding='euc-kr')
        df_PER_dic[date][['PER', 'PBR']] = df_PER_dic[date][[
            'PER', 'PBR']].fillna(10000)
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

    data = pd.concat([data, data_PER[['PER', 'PBR', '배당수익률']]], axis=1)
    data['PER_inv'] = 1/data['PER']
    data['PBR_inv'] = 1/data['PBR']

    # 이유는 모르겠지만 PER_inv에 inf값이 껴있음...(seasonal 모델 생성 시도부터..)
    data = data.drop(data[data['PER_inv'] == np.inf].index)

    print('학습 데이터 형태(row, col) :', data.shape)
    print(data.head(3))
    return data


def train_model(date_list, date_list_for_PER, DATA_PATH, N, n, m):
    data = prepare_data(date_list, date_list_for_PER, DATA_PATH, N, n, m)
    features = [f'{n}개월 간 수익률', f'KOSPI대비 {n}개월 간 수익률',
                f'{n}개월 간 KOSPI 이김', 'PER_inv', 'PBR_inv', '배당수익률']
    # 얘네들을 feature에 넣으면 Data leakage가 발생하게 됨
    targets = ['3개월 후 수익률', 'KOSPI대비 3개월 후 수익률', 'KOSPI보다 많이 오름']
    target = ['KOSPI보다 많이 오름']

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10)

    # basline model 정의
    target_mode = y_train['KOSPI보다 많이 오름'].value_counts(
        normalize=True).sort_values(ascending=False).index[0]
    # precision check를 위해 baseline model은 1로
    baseline_model = [1] * len(y_test)

    # GridSearchCV

    pipe = make_pipeline(
        RandomForestClassifier(random_state=10, oob_score=True, n_jobs=-1)
    )

    dists = {
        'randomforestclassifier__n_estimators': [150, 300, 600],
        'randomforestclassifier__min_samples_leaf': [2, 3, 4],
    }

    clf = GridSearchCV(
        pipe,
        param_grid=dists,
        cv=3,
        scoring='precision',
        verbose=1,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    print('Optimized Hyper parameters : {}'.format(clf.best_params_))
    print('Best precision score : {}'.format(clf.best_score_))

    best_model = clf.best_estimator_

    y_pred = best_model.predict(X_test.values)
    y_proba = best_model.predict_proba(X_test.values)[:, 1]

    threshold = 0.65
    y_pred_thres = np.array(
        list(map(lambda x: 0 if x < threshold else 1, y_proba)))

    print()
    print('Baseline model Precision score : {:.4f}'.format(
        precision_score(y_test, baseline_model)))
    print('Model Cross Vlidation Precision score : {:.4f}'.format(cross_val_score(
        best_model, X.values, y.values.ravel(), cv=5, scoring='precision').mean()))
    print('Model Precision score(Test set) : {:.4f}'.format(
        precision_score(y_test, y_pred)))
    print('Threshold adjusted Model Precision score : {:.4f}'.format(
        precision_score(y_test, y_pred_thres)))
    print('만약 threshold값 이상의 종목을 사고 {}개월 뒤 판매했다면? 기대 수익률 : {:.2f}%'.format(
        m, data.loc[X_test[y_proba > threshold].index][f'{m}개월 후 수익률'].mean()*100))

    # 모델 내보내기(피클화하기)
    print(f'Model made. Name : model_{date_list[-1]}_{N}_{n}_{m}.pkl')
    with open(f'models/model_{date_list[-1]}_{N}_{n}_{m}.pkl', 'wb') as pickle_file:
        pickle.dump(best_model, pickle_file)


def main(date_list, date_list_for_PER, DATA_PATH, N, n, m):
    train_model(date_list, date_list_for_PER, DATA_PATH, N, n, m)


if __name__ == '__main__':
    # 2003년부터 PER, PBR이 기입돼 있어서 2003년부터
    # 상반기용
    # date_list = ['20030102', '20040102', '20050102', '20060102', '20070102', '20080102', '20090102'
    #              , '20110102', '20120102', '20130102', '20140102', '20150102'
    #              , '20160102', '20170102', '20180102', '20190102', '20200102']
    # date_list_for_PER = date_list + ['20210102'] # 왜 date_list보다 하나를 더 넣었을까... 별 의미 없었을 것 같기도...(당시 처음 코드를 짤 때(21년 11월) )
    # PATH = './data'

    # # seasonnal 전체 데이터
    # date_list = ['20031101', '20041101', '20051101', '20061101', '20071101', '20081101', '20091101', '20111101',
    #              '20121101', '20131101', '20141101', '20151101', '20161101', '20171101', '20181101', '20191101', '20201101']
    # date_list_for_PER = date_list + ['20211101']
    # PATH = './data_seasonal'

    # 12월 모델
    date_list = ['20031201', '20041201', '20051201', '20061201', '20071201', '20081201', '20091201', '20111201',
                 '20121201', '20131201', '20141201', '20151201', '20161201', '20171201', '20181201', '20191201', '20201201']
    date_list_for_PER = date_list + ['20211201']
    PATH = './data_December'

    N = 600  # 시가총액 상위 N 종목
    n = 12  # n개월 간의 주가 추이 확인
    m = 6  # m개월 후의 수익률 확인
    main(date_list, date_list_for_PER, PATH, N, n, m)
