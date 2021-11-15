import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

class Preprocessor:
    """
    特徴量生成
    """

    def __init__(self, df: pd.DataFrame, export_path):
        self.df = df
        self.export_path = export_path
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        encode_cols = ['Sector', 'Industry', 'List1', 'List2']
        for col in encode_cols:
            self.df[col] = LabelEncoder().fit_transform(self.df[col].fillna('nothing'))

    def apply_log(self, df):
        df['y'] = df['y'].apply(np.log1p)

    def make_objective(self, df):
        df['y_prev'] = df[['id', 'y']].groupby('id')['y'].transform(lambda x: x.shift(1).fillna(method='bfill'))
        df['y_diff'] = df['y'] - df['y_prev']
        df['y_diff_std'] = df[['id', 'y']].groupby('id')['y'].transform(lambda x: x.std())
        df['y_diff_norm'] = df['y_diff'] / df['y_diff_std']

    def make_day_feature(self, df):
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    def _make_lags(self, df, group_col, val_col, lag):
        return df[[group_col, val_col]].groupby(group_col)[val_col].transform(lambda x: x.shift(lag))
    
    def make_lag_feature(self, df):
        lags = [1,2,3,4]
        val_col = 'y_diff_norm'
        for lag in lags:
            df[f'lag_{lag}_{val_col}'] = self._make_lags(df, 'id', val_col, lag)

    def _make_roll_rmean(self, df, group_col, val_col, lag, roll):
        return df[[group_col, val_col]].groupby(group_col)[val_col].transform(lambda x: x.shift(lag).rolling(roll).mean())
    
    def _make_roll_rstd(self, df, group_col, val_col, lag, roll):
        return df[[group_col, val_col]].groupby(group_col)[val_col].transform(lambda x: x.shift(lag).rolling(roll).std())

    def make_roll_feature(self, df):
        lags = [1]
        rolls = [4, 9, 13, 26, 52]
        val_col = 'y_diff_norm'
        for lag in lags:
            for roll in rolls:
                df[f'rmean_{lag}_{val_col}_{roll}'] = self._make_roll_rmean(df, 'id', val_col, lag, roll)
                df[f'rstd_{lag}_{val_col}_{roll}'] = self._make_roll_rstd(df, 'id', val_col, lag, roll)

    def _make_volatility(self, df, val_col, roll):
        return np.log(df[val_col]).diff().rolling(roll).std()
        
    def make_volatility_feature(self, df):
        rolls = [4, 8, 12]
        for roll in rolls:
            df[f'volatility_{roll}'] = self._make_volatility(df, 'y', roll)

    def _make_mean_gap(self, df, roll):
        return df['y'] / (df['y'].rolling(roll).mean())

    def make_mean_gap_feature(self, df):
        rolls = [4, 8, 12]
        for roll in rolls:
            df[f'mean_gap_{roll}'] = self._make_mean_gap(df, roll)

    def make_features(self, df):
        self.apply_log(df)
        self.make_objective(df)
        self.make_day_feature(df)
        self.make_lag_feature(df)
        self.make_roll_feature(df)
        self.make_volatility_feature(df)
        self.make_mean_gap_feature(df)

    def export(self, df):
        df.to_csv(self.export_path, index=False)

    def read(self):
        if(os.path.exists(self.export_path)):
            return pd.read_csv(self.export_path)
        else:
            return None

    def preprocess(self, update=False):
        df_pre = self.read()
        if((df_pre is None) or (update is True)):
            self.make_features(self.df)
            self.export(self.df)
            return self.df
        else:
            return df_pre
