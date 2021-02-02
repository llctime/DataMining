# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of testing a model
# @Author: llc
# @Time: 2021/2/2 23:04

from Main import readFileToDataFrame, TEST_DATA_FILEPATH, ENCODING, MISSING_VALUES, SHOW_LOG, ColumnDeletion, \
    MissingImputation, Digitization, Binaryzation, Standardization, OneHotEncoding, FeatureCalculation, \
    FeaturePolynomialization, VarianceSelection, ChisquareSelection

from sklearn.base import BaseEstimator, TransformerMixin
class ModelTesting(BaseEstimator, TransformerMixin):
    # 模型预测
    def __init__(self, label_data, algorithm, params):
        self.label_data = label_data
        self.algorithm = algorithm
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 1.数据切分
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, self.label_data, test_size=0.33, random_state=42)
        # 2.构建模型
        estimator = ALGORITHMS.get(self.algorithm)(**self.params)
        # 3.训练模型
        estimator.fit(X_train, y_train)
        # 4.验证
        y_pred = estimator.predict(X_test)
        y_predprob = estimator.predict_proba(X_test)[:, 1]
        # 5.模型校验
        from sklearn import metrics
        print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
        # 6.保存模型文件
        import time
        year = str(time.localtime().tm_year)
        month = str(time.localtime().tm_mon)
        day = str(time.localtime().tm_mday)
        hour = str(time.localtime().tm_hour)
        minute = str(time.localtime().tm_min)
        second = str(time.localtime().tm_sec)
        model_name = self.algorithm+'_'+str(metrics.roc_auc_score(y_test, y_predprob))+'_'+year+month+day+hour+minute+second+'.pickle'
        import os
        if not os.path.exists(MODEL_FILEPATH):
            os.makedirs(MODEL_FILEPATH)
        import pickle
        with open(MODEL_FILEPATH+model_name, 'wb') as f:
            pickle.dump(estimator, f)

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
                           ('chi2', ChisquareSelection())])
    pipeline.fit_transform(ori_data)



if __name__ == "__main__":
    Test()