import pandas as pd
import os

class Dataset:
    """
        データ成型を行うクラス
    """

    def __init__(self, train_data_path, company_list_path, export_path):
        self.train_data = pd.read_csv(train_data_path)
        self.company_list = pd.read_csv(company_list_path)
        self.train_company_list = list(self.train_data.columns)[1:]
        self.export_path = export_path
        self.set_date(self.train_data)

    def set_date(self, train_data):
        train_data.loc[419,'Date'] = '2019/11/24'
        train_data['Date'] = pd.to_datetime(train_data['Date'])

    def stack(self, train_data):
        train_data = train_data.set_index('Date') \
                            .stack(dropna=False) \
                            .reset_index()
        train_data.rename(columns={'level_1':'id', 0:'y'}, inplace=True)

    def join(self, train_data, company_list):
        company_list['List1'] = company_list[['Symbol', 'List']].groupby('Symbol').transform(lambda x: x.iloc[0])
        company_list['List2'] = company_list[['Symbol', 'List']].groupby('Symbol').transform(lambda x: x.iloc[-1])
        company_list = company_list.drop('List', axis=1).drop_duplicates(subset='Symbol').reset_index(drop=True)
        train_df = pd.merge(train_data, company_list, left_on='id', right_on='Symbol', how='left')
        return train_df.drop('Symbol' ,axis=1)

    def cleansing(self, train_data, company_list):
        self.stack(train_data)
        df = self.join(train_data, company_list)
        return df

    def export(self, df):
        df.to_csv(self.export_path, index=False)

    def read(self):
        if(os.path.exists(self.export_path)):
            return pd.read_csv(self.export_path)
        else:
            return None

    def get_df(self, update=False):
        df = self.read()
        if((df is None) or (update is True)):
            df = self.cleansing(self.train_data, self.company_list)
            self.export(df)
        return df