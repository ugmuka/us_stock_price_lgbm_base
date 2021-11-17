from lib.dataset import Dataset
from lib.preprocessor import Preprocessor
from lib.lightgbm import LGBMModel
import datetime

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
        df = preprocessor.preprocess(update=False)
        
        # train model
        config_path = './config/lgbm_params.yml'
        submission_template_path = './data/submission_template.csv'
        result_path = './result/submission_' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d_%H:%M:%S') + '.csv'
        log_dir = '/home/jovyan/log'
        model = LGBMModel(config_path, df, submission_template_path, result_path, log_dir)
        model.run()
