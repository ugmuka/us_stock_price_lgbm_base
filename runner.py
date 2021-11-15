from lib.dataset import Dataset
from lib.preprocessor import Preprocessor
from lib.lightgbm import LGBMModel

class Runner:
    def __init__(self):
        pass

    def exec(self):
    
        train_data_path = './data/train_data.csv'
        company_list_path = './data/company_list.csv'
        export_path = './data/join_df.csv'
        export_path_preprocess = './data/preprocess_df.csv'

        # read dataset
        dataset = Dataset(train_data_path, company_list_path, export_path)
        dataset_df = dataset.get_df(update=False)

        # preprocess
        preprocessor = Preprocessor(dataset_df, export_path_preprocess)
        df = preprocessor.preprocess(update=True)
        # col = ['Year','Month','Day','WeekOfYear','IPOyear','Sector','Industry','List1','List2','lag_1_y_diff_norm','lag_2_y_diff_norm','lag_3_y_diff_norm','lag_4_y_diff_norm','rmean_1_y_diff_norm','rstd_1_y_diff_norm']
        # train_df_sort = df.sort_values(by=['Year','Month','Day'])
        # x_train_sort = train_df_sort[col]
        # y_train_sort = train_df_sort['y_diff_norm']
        # x_train_sort.to_csv('preprocess_x.csv', index=False)
        # y_train_sort.to_csv('preprocess_y.csv', index=False)
        
        # train model
        config_path = './config/lgbm_params.yml'
        submission_template_path = './data/submission_template.csv'
        model = LGBMModel(config_path, df, submission_template_path)
        model.train()
        model.submit()
