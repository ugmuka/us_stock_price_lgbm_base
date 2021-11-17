import pandas as pd
import numpy as np
import random
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
import lightgbm as lgb
import yaml
import optuna.integration.lightgbm as lgb_optuna
import mlflow
import datetime

class LGBMModel:
    def __init__(self, config_path, df, submission_template_path, result_path, log_dir):
        with open(config_path, 'r') as yml:
            config = yaml.safe_load(yml)
            self.params = config['params']
            self.config = config['config']
            self.target_col = config['target_col']
            self.feature_cols = config['feature_cols']
            df = self.encode(df)
            self.test_df = df[df['Date']=='2019-11-24'].reset_index(drop=True)
            self.train_df = df[df['Date']!='2019-11-24'].reset_index(drop=True)
            self.sort()
            self.submission_template_path = submission_template_path
            self.result_path = result_path
            mlflow.set_tracking_uri('file://' + log_dir)
            mlflow.set_experiment("lgbm-experiment")

    def encode(self, df):
        encode_cols = ['Sector', 'Industry', 'List1', 'List2']
        for col in encode_cols:
            df[col] = LabelEncoder().fit_transform(df[col].fillna('nothing'))
        return df

    def sort(self):
        self.train_df = self.train_df.sort_values('Date')
    
    def validate(self, n_splits):
        if(self.config['validate_method']=='TimeSeriesSpilit'):
            id_count = len(self.train_df['id'].unique())
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=id_count)
            x_train_sort = self.train_df[self.feature_cols]
            y_train_sort = self.train_df[self.target_col]
            return tscv.split(x_train_sort, y_train_sort)
        else:
            raise ValueError("validate method error")

    def train(self):
        mlflow.start_run()
        x_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]
        y_diff_std =self.train_df['y_diff_std']
        groups = self.train_df['id']
        y_oof = np.zeros(len(self.train_df))
        y_preds = []
        if(hasattr(self, 'best_params')):
                params = self.best_params
        else:
            params = self.params

        mlflow.log_param('params',params)
        mlflow.log_param('feature',self.feature_cols)
        mlflow.log_param('y',self.target_col)

        for fold, (tr_idx, vl_idx) in enumerate(self.validate(n_splits=self.config['validate_splits'])):
            x_tr_fold = x_train.iloc[tr_idx]
            y_tr_fold = y_train.iloc[tr_idx]
            x_vl_fold = x_train.iloc[vl_idx]
            y_vl_fold = y_train.iloc[vl_idx]

            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(
                x_tr_fold, y_tr_fold,
                eval_set=(x_vl_fold, y_vl_fold),
                eval_metric='rmse',
                verbose=False,
                early_stopping_rounds=100,
            )

            y_oof[vl_idx] = self.model.predict(x_vl_fold)
            score = np.sqrt(np.mean(np.square((y_oof[vl_idx] - y_vl_fold) * y_diff_std[vl_idx])))

            print(f'fold {fold} score:{score}')
            mlflow.log_metric(f'fold{fold}_score', score)

        oof_score = np.sqrt(np.mean(np.square((y_oof[vl_idx] - y_vl_fold) * y_diff_std[vl_idx])))
        print('oof score:', oof_score)
        mlflow.log_metric('oof_score', oof_score)
        mlflow.end_run()


    def optuna_tuning(self):
        x_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]
        _, (tr_idx, vl_idx) = self.validate(n_splits=2)
        x_tr_fold = x_train.iloc[tr_idx]
        y_tr_fold = y_train.iloc[tr_idx]
        x_vl_fold = x_train.iloc[vl_idx]
        y_vl_fold = y_train.iloc[vl_idx]

        lgb_train = lgb.Dataset(x_tr_fold, y_tr_fold)
        lgb_eval = lgb.Dataset(x_vl_fold, y_vl_fold)

        booster = lgb_optuna.LightGBMTuner(
            params=self.params, 
            train_set=lgb_train,
            valid_sets=lgb_eval,
            optuna_seed=42,
        )

        booster.run()

        self.best_params = booster.best_params

    def predict(self):
        x_test = self.test_df[self.feature_cols]
        y_preds = self.model.predict(x_test)
        print(y_preds)
        return y_preds

    def submit(self):
        y_preds = self.predict()
        submission_df = pd.read_csv(self.submission_template_path)
        submission_df['y'] = np.expm1(
            np.mean(y_preds, axis=0) * self.test_df['y_diff_std'].values + self.test_df['y_prev'].values
        )
        submission_df.to_csv(self.result_path, index=False)

    def run(self):
        self.optuna_tuning()
        self.train()
        self.submit()