# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of testing a model
# @Author: llc
# @Time: 2021/2/2 23:04

from Main import readFileToDataFrame, TEST_DATA_FILEPATH, ENCODING, MISSING_VALUES, SHOW_LOG, ColumnDeletion, \
    MissingImputation, Digitization, Binaryzation, Standardization, OneHotEncoding, FeatureCalculation, \
    FeaturePolynomialization, VarianceSelection, ChisquareSelection, MODEL_FILEPATH, ID, TARGET

RESULT_FILEPATH = 'Results/'

from sklearn.base import BaseEstimator, TransformerMixin
class ModelTesting(BaseEstimator, TransformerMixin):
    # 模型预测
    def __init__(self, model_filepath, id_label, result_label, id_data):
        self.model_filepath = model_filepath
        self.id_label = id_label
        self.result_label = result_label
        self.id_data = id_data

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fp = open(self.model_filepath, "rb+")
        import pickle
        model = pickle.load(fp)
        y_pred = model.predict_proba(X)
        import pandas as pd
        set = list(zip(self.id_data, y_pred[:, 0]))
        result = pd.DataFrame(data=set, columns=[self.id_label, self.result_label])
        result.to_csv(RESULT_FILEPATH+self.model_filepath.split('/')[-1]+'.csv', index=False)

def Test():

    # 导入原始数据
    ori_data = readFileToDataFrame(TEST_DATA_FILEPATH, ENCODING, MISSING_VALUES)
    print('原始数据：\n',ori_data.head(10)) if SHOW_LOG == True else ()


    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('del_Columns', ColumnDeletion()),
                           ('impute_miss', MissingImputation()),
                           ('digi_string', Digitization()),
                           ('bin', Binaryzation()),
                           ('scaler', Standardization()),
                           ('onehot', OneHotEncoding()),
                           ('cal_features', FeatureCalculation()),
                           ('poly_features', FeaturePolynomialization()),
                           ('sel_var', VarianceSelection()),
                           ('chi2', ChisquareSelection()),
                           ('test', ModelTesting(MODEL_FILEPATH+'GBDT_0.6874720616370318_20212222515.pickle', ID, TARGET, ori_data[ID]))])
    pipeline.fit_transform(ori_data)



if __name__ == "__main__":
    Test()