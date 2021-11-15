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

    def _make_lags(self, df, group_col, val_col, lags):
        lag_df = pd.DataFrame()
        for lag in lags:
            lag_df[f'lag_{lag}_{val_col}'] = df[[group_col, val_col]].groupby(
                group_col)[val_col].transform(lambda x: x.shift(lag))
        return lag_df
    
    def make_lag_feature(self, df):
        lag_df = self._make_lags(df, 'id', 'y_diff_norm', [1,2,3,4])
        df = pd.concat([df, lag_df], axis=1)

    def _make_rolls(self, df, group_col, val_col, lags, rolls):
        roll_df = pd.DataFrame()
        for lag in lags:
            for roll in rolls:
                roll_df[f'rmean_{lag}_{val_col}'] = df[[group_col, val_col]].groupby(
                    group_col)[val_col].transform(lambda x: x.shift(lag).rolling(roll).mean())
                roll_df[f'rstd_{lag}_{val_col}'] = df[[group_col, val_col]].groupby(
                    group_col)[val_col].transform(lambda x: x.shift(lag).rolling(roll).std())
        return roll_df

    def make_roll_feature(self, df):
        roll_df = self._make_rolls(df, 'id', 'y_diff_norm', [1], [4, 9, 13, 26, 52])
        df = pd.concat([df, roll_df], axis=1)

    def make_features(self, df):
        self.apply_log(df)
        self.make_objective(df)
        self.make_day_feature(df)
        self.make_lag_feature(df)
        self.make_roll_feature(df)

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
