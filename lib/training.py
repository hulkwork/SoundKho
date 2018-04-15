import os
import pandas
from lib import sound_feats
import logging
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time

logger = logging.getLogger('SoundKho')
logger.setLevel(logging.DEBUG)
logger.info("Begin Training")


class Training(object):
    def __init__(self, csv_file_train, base_data_path, subset_ratio=0.8):
        self.data_path = base_data_path
        self.train_csv_file = csv_file_train
        self.train = pandas.read_csv(self.train_csv_file)
        if subset_ratio:
            self.train = self.train[:int(subset_ratio * len(self.train))]
        self.model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05,silent=False)

    def get_all_files(self):
        return [os.path.join(self.data_path, filename) for filename in self.train['fname'].values]

    def get_manually_verified(self):
        return [os.path.join(self.data_path, filename) for filename in
                self.train[self.train["manually_verified"]]['fname'].values]

    def upgrade_df(self):
        self.train['features_fft'] = self.train['fname'].apply(
            lambda fname: sound_feats.WaveFeatures(os.path.join(self.data_path, fname), plot=False
                                                   ).get_feat())

    def train_mode(self):
        self.upgrade_df()
        y = self.train['label']
        X_train, X_test, y_train, y_test = train_test_split(self.train['features_fft'], y, test_size=0.33,
                                                            random_state=42)
        X_data_train = [item.tolist()[0].tolist() for item in self.train[["features_fft"]].values]
        df = pandas.DataFrame(X_data_train)

        self.dict_cat = {i: val for val, i in enumerate(self.train['label'].unique())}
        print(len(self.dict_cat))
        time.sleep(10)
        y_transform = [self.dict_cat[item] for item in self.train['label'].values]
        self.model.fit(df, y_transform)
        log = [1 if item == y_test[i] else 0 for item, i in enumerate(self.model.predict(X_test))]
        print(log)
