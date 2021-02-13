# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model with pipeline
# @Author: llc
# @Time: 2021/2/2 23:04

# ����
SHOW_LOG = True    # ��־�����ʶ

TRAIN_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\train.csv'    # ѵ�������ļ�·��
TEST_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\testA.csv'    # ���������ļ�·��
ENCODING = 'GBK'    # ���ļ������ʽ
MISSING_VALUES = 'NaN'    # ԭʼ����ȱʧֵ

ANALYSIS_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\��ش���ΥԼԤ��ѵ�������ֶη���.csv'
TARGET = 'isDefault'    # ��ǩ
ID = 'id'    # Ψһ��ʶ
import pandas as pd
TRAIN_LABEL = pd.DataFrame()
ALGORITHMS = {
    'RFC': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).RandomForestClassifier,
    'GBDT': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).GradientBoostingClassifier
}    # �㷨�б�
MODEL_FILEPATH = 'Models/'

from sklearn.base import BaseEstimator, TransformerMixin

def readFileToDataFrame(filepath, encoding, missing_values):
    import pandas as pd
    data = pd.read_csv(filepath, na_values=missing_values, encoding=encoding)
    return data

def makeAnalysisFile(filepath, data):
    # ������ʼ�ֶη����ļ�
    import os
    if not os.path.exists(filepath):
        import pandas as pd
        analysis = pd.DataFrame(data=data.dtypes, columns=['type'])
        analysis.to_csv(filepath)

def selectFeatures(data, match_char_list, match_col, output_col):
    # ���߷��������ڽ������е�ÿ���ֶθ��ݲ�ͬ��ƥ������ɸѡ����
    # data: ����
    # match_char_list: ������ָ����ƥ��ؼ���
    # match_col: ָ��������Ҫƥ��Ŀ���ֶε�����
    # output_col: ָ��ƥ��ɹ�������Ҫ������ֶ�
    classified_features = []
    for i in data[match_col].index:
        if data[match_col][i] != 'nan':
            for j in match_char_list:
                tmp = ''
                if j in str(data[match_col][i]):
                    tmp = str(data[output_col][i])
                if len(tmp) > 0:
                    classified_features.append(tmp)
                    break
    return classified_features

class ColumnDeletion(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        after_del_data = X.copy()
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
        after_del_data = X.drop(X[del_features], 1, inplace=False)
        print('ɾ����ʣ������ݣ�\n', after_del_data.head(10)) if SHOW_LOG == True else ()
        return after_del_data

class MissingImputation(BaseEstimator, TransformerMixin):
    # ����ַ��ͺ���ֵ�������ֱ�ȱʧֵ
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        numric_features = selectFeatures(del_ana_data, ['int', 'float'], 'type', 'colname')
        string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
        impute_data = X.copy()
        from sklearn.impute import SimpleImputer
        import numpy as np
        import pandas as pd
        simpleimputer_numric = SimpleImputer(missing_values=np.nan)    # ��ֵ�������þ�ֵ��
        impute_data[numric_features] = pd.DataFrame(data=simpleimputer_numric.fit_transform(X[numric_features]), columns=numric_features)
        print('��ֵ��������ȱʧֵ��\n', impute_data.head(10)) if SHOW_LOG == True else ()
        string_features_most_frequent = impute_data[string_features].mode()
        for col in string_features_most_frequent.columns:
            simpleimputer_string = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=string_features_most_frequent[col])    # �ַ���������������
            impute_data[col] = pd.DataFrame(data=simpleimputer_string.fit_transform(X[col].values.reshape(-1, 1)), columns=[col])
        print('�ַ���������ȱʧֵ��\n', impute_data.head(10)) if SHOW_LOG == True else ()
        return impute_data

class Digitization(BaseEstimator, TransformerMixin):
    # �ַ���������ǩ��תΪ��ֵ��
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        numric_features = selectFeatures(del_ana_data, ['int', 'float'], 'type', 'colname')
        string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
        ordinal_encode_data = X.copy()
        import pandas as pd
        if len(string_features) > 0:
            from sklearn.preprocessing import OrdinalEncoder
            OrdinalEncoder().fit(X[string_features]).categories_
            ordinal_encode_data[string_features] = pd.DataFrame(data=OrdinalEncoder().fit_transform(X[string_features]), columns=string_features)
            print('�ַ���������ǩ����\n', ordinal_encode_data.head(10)) if SHOW_LOG == True else ()
        else:
            print('����Ҫ��ǩ�������ݣ�') if SHOW_LOG == True else ()
        return ordinal_encode_data

class Binaryzation(BaseEstimator, TransformerMixin):
    # �����ͣ���ֵ��
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        bin_con_data = X.copy()
        BINARY_CONTINUOUS_FEATURES = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'colname')    # ���ֵ��������������
        BINARY_THRESHOLD = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'threshold')    # ��ֵ������ֵ
        if len(BINARY_CONTINUOUS_FEATURES) > 0:
            from sklearn.preprocessing import Binarizer
            import pandas as pd
            for i in range(len(BINARY_CONTINUOUS_FEATURES)):
                binarizer = Binarizer(threshold = int(float(BINARY_THRESHOLD[i])))
                bin_data = pd.DataFrame(data=binarizer.fit_transform(X[BINARY_CONTINUOUS_FEATURES[i]].values.reshape(-1, 1)), columns=[BINARY_CONTINUOUS_FEATURES[i]])
                bin_con_data[BINARY_CONTINUOUS_FEATURES[i]] = bin_data
            print('�����ֵ���������Ϊ��\n', bin_con_data.head(10)) if SHOW_LOG == True else ()
        else:
            print('����Ҫ��ֵ�������ݣ�') if SHOW_LOG == True else ()
        return bin_con_data

class Standardization(BaseEstimator, TransformerMixin):
    # ��׼����minmax��
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scale_con_data = X.copy()
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        minmaxScaler = MinMaxScaler().fit(X)
        scale_con_data = pd.DataFrame(data=minmaxScaler.fit_transform(X), columns=X.columns)
        print('������������׼����\n', scale_con_data.head(10)) if SHOW_LOG == True else ()
        return scale_con_data

class OneHotEncoding(BaseEstimator, TransformerMixin):
    # ��������onehot
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        DISORDER_CATEGORICAL_FEATURES = selectFeatures(del_ana_data, ['OneHotEncoder'], 'transform',
                                                       'colname')  # �������������
        encode_disorder_cat_data = X.copy()
        if len(DISORDER_CATEGORICAL_FEATURES) > 0:
            from sklearn.preprocessing import OneHotEncoder
            import pandas as pd
            if len(DISORDER_CATEGORICAL_FEATURES) == 1:
                onehotencoder = OneHotEncoder(categories='auto').fit(X[DISORDER_CATEGORICAL_FEATURES])
                tmp_l = encode_disorder_cat_data.drop(DISORDER_CATEGORICAL_FEATURES, axis=1)
                tmp_r = pd.DataFrame(
                    data=onehotencoder.fit_transform(X[DISORDER_CATEGORICAL_FEATURES]).toarray(),
                    columns=('onehot_' + str(item) for item in range(len(onehotencoder.categories_[0]))))
                encode_disorder_cat_data = pd.concat([tmp_l, tmp_r], axis=1, ignore_index=False)
            else:
                import numpy as np
                onehotencoder = OneHotEncoder(categories='auto').fit(X[DISORDER_CATEGORICAL_FEATURES])
                tmp_l = encode_disorder_cat_data.drop(DISORDER_CATEGORICAL_FEATURES, axis=1)
                tmp_r = pd.DataFrame(
                    data=onehotencoder.fit_transform(X[DISORDER_CATEGORICAL_FEATURES]).toarray(),
                    columns=('onehot_' + str(item) for item in range(len(np.append(*onehotencoder.categories_)))))
                encode_disorder_cat_data = pd.concat([tmp_l, tmp_r], axis=1, ignore_index=False)
            print('����������������ȱ��룺\n', encode_disorder_cat_data.head(10)) if SHOW_LOG == True else ()
        else:
            print('����Ҫonehot�����ݣ�') if SHOW_LOG == True else ()
        return encode_disorder_cat_data

def minusFeatures(X, first_operators, second_operators):
    for i in range(len(first_operators)):
        X[first_operators[i]+'-'+second_operators[i]] = X[first_operators[i]] - X[second_operators[i]]
        X.drop([first_operators[i], second_operators[i]], 1, inplace=True)
    return X

class FeatureCalculation(BaseEstimator, TransformerMixin):
    # ����������������
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        MINUS_FIRST_FEATURES = ['ficoRangeLow']
        MINUS_SECOND_FRATURES = ['ficoRangeHigh']
        cal_data = X.copy()
        if len(MINUS_FIRST_FEATURES) > 0:
            cal_data = minusFeatures(X, MINUS_FIRST_FEATURES, MINUS_SECOND_FRATURES)
            print('�������������������ݣ�\n', cal_data.head(10)) if SHOW_LOG == True else ()
        else:
            print('����Ҫ������������ݣ�') if SHOW_LOG == True else ()
        return cal_data

class FeaturePolynomialization(BaseEstimator, TransformerMixin):
    # ���ɶ���ʽ����������������
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        POLYNOMIAL_FEATURES = []  # �����ɶ���ʽ����������
        import pandas as pd
        polynomial_data = pd.DataFrame()
        if len(POLYNOMIAL_FEATURES) > 0:
            from sklearn.preprocessing import PolynomialFeatures
            polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            polynomial_data = pd.DataFrame(data=polynomialfeatures.fit_transform(X[POLYNOMIAL_FEATURES]),
                                           columns=('poly_' + str(item) for item in
                                                    range(polynomialfeatures.n_output_features_)))
            print('���ɶ���ʽ�������ݣ�\n', polynomial_data) if SHOW_LOG == True else ()
        else:
            print('�������ɶ���ʽ������') if SHOW_LOG == True else ()
        poly_data = pd.concat([X, polynomial_data], axis=1, ignore_index=False)
        print('�������ʽ���������ݣ�\n', poly_data) if SHOW_LOG == True else ()
        return poly_data

class VarianceSelection(BaseEstimator, TransformerMixin):
    # ����ѡ��ֻɸ������Ϊ0����ֻ��һ��ȡֵ������
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        VARIANCE_THRESHOLD = 0  # ����ѡ����ֵ
        from sklearn.feature_selection import VarianceThreshold
        import pandas as pd
        variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(X),
                                    columns=(X.columns[item] for item in
                                             variancethreshold.get_support(indices=True)))
        print('����ѡ�������ݣ�\n', var_sel_data.head(10)) if SHOW_LOG == True else ()
        return var_sel_data

class ChisquareSelection(BaseEstimator, TransformerMixin):
    # ������������ѡ��
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        chi2_sel_data = X.copy()
        CHI2_FEATURES = ['loanAmnt', 'term', 'interestRate', 'installment', 'grade', 'subGrade',
                         'employmentTitle', 'homeOwnership', 'annualIncome',
                         'verificationStatus', 'dti', 'delinquency_2years', 'openAcc', 'pubRec',
                         'pubRecBankruptcies', 'revolUtil', 'initialListStatus',
                         'applicationType', 'n1', 'n2', 'n3', 'n5', 'n7', 'n9', 'n10', 'n14',
                         'onehot_0', 'onehot_1', 'onehot_2', 'onehot_3', 'onehot_4', 'onehot_5',
                         'onehot_6', 'onehot_8', 'onehot_9', 'onehot_10', 'onehot_12',
                         'onehot_14', 'onehot_16', 'onehot_17', 'onehot_19', 'onehot_20',
                         'onehot_21', 'onehot_22', 'onehot_25', 'onehot_26', 'onehot_27',
                         'onehot_29', 'onehot_31', 'onehot_32', 'onehot_33', 'onehot_34',
                         'onehot_35', 'onehot_36', 'onehot_38', 'onehot_39', 'onehot_40',
                         'onehot_41', 'onehot_43', 'onehot_44', 'onehot_46', 'onehot_47',
                         'onehot_48', 'onehot_49', 'onehot_50', 'onehot_51', 'onehot_52',
                         'onehot_53', 'onehot_54', 'onehot_55', 'onehot_56', 'onehot_58',
                         'onehot_59', 'onehot_61']
        chi2_sel_data = X[CHI2_FEATURES]
        # ע������ʱ��ϳ��������һ������֮���ɸѡ��������б��棬֮�����ܴ˹���
        # from sklearn.feature_selection import mutual_info_classif as MIC
        # mic = MIC(X, TRAIN_LABEL)    # ����Ϣ����֤�������ǩ������ԣ����Ժͷ����ԣ���0Ϊ�໥������1Ϊ���
        # CHI_K = mic.shape[0] - (mic <= 0).sum()    # kֵ������������ȥp>0.05���������õ�
        # print('����ѡ���kֵ��\n', CHI_K) if SHOW_LOG == True else ()
        # if CHI_K > 0:
        #     from sklearn.feature_selection import SelectKBest
        #     from sklearn.feature_selection import chi2
        #     selectkbest = SelectKBest(chi2, k=CHI_K)
        #     chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(X, TRAIN_LABEL), columns=(X.columns[item] for item in selectkbest.get_support(indices=True)))
        # else:
        #     print('������п�������ѡ��') if SHOW_LOG == True else ()
        # print('����ѡ����������\n', CHI2_FEATURES) if SHOW_LOG == True else ()
        # print('����ѡ�������ݣ�\n', chi2_sel_data.head(10)) if SHOW_LOG == True else ()
        return chi2_sel_data

def plotEmbedded(algorithm, n_estimators, X, y, partitions, cv):
    # time warning #
    # Ƕ�뷨������(importance��ֵ)ѧϰ����
    # ������
    #   algorithm��ģ���㷨������'RFC'�������ƣ���
    #   n_estimators����������
    #   partitions��feature_importance��0��max֮��ȡ����
    #   cv��������֤����
    import numpy as np
    estimator = ALGORITHMS.get(algorithm)(n_estimators=n_estimators, random_state=0)
    threshold = np.linspace(0, estimator.fit(X, y).feature_importances_.max(), partitions)
    print(estimator.fit(X, y).feature_importances_)
    score = []
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import cross_val_score
    highest_score = 0
    highest_threshold = 0
    for i in threshold:
        X_embedded = SelectFromModel(estimator, threshold=i).fit_transform(X, y)
        once = cross_val_score(estimator, X_embedded, y, cv=cv).mean()
        score.append(once)
        if once > highest_score:
            highest_score = once
            highest_threshold = i
    import matplotlib.pyplot as plt
    plt.plot(threshold, score)
    plt.show()
    return highest_threshold

def plotWrapper(algorithm, n_estimators, X, y, step, cv):
    # time warning #
    # ��װ��ʣ����������(N_FEATURES_TO_SELECT)ѧϰ����
    # ������
    #   algorithm��ģ���㷨������'RFC'�������ƣ���
    #   n_estimators����������
    #   step��ÿ�ε���������������
    #   cv��������֤����
    import numpy as np
    estimator = ALGORITHMS.get(algorithm)(n_estimators=n_estimators, random_state=0)
    score = []
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import cross_val_score
    highest_score = 0
    highest_threshold = 0
    for i in range(1, X.columns.shape[0], step):
        X_wrapper = RFE(estimator, n_features_to_select=i, step=step).fit_transform(X, y)
        once = cross_val_score(estimator, X_wrapper, y, cv=cv).mean()
        score.append(once)
        if once > highest_score:
            highest_score = once
            highest_threshold = i
    import matplotlib.pyplot as plt
    plt.plot(range(1, X.columns.shape[0], step), score)
    plt.show()
    return highest_threshold

def plotPCA(X):
    from sklearn.decomposition import PCA
    pca_line = PCA().fit(X)
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
    plt.xlabel("number of components after dimension reduction")
    plt.ylabel("cumulative explained variance ratio")
    plt.show()

class PCADecomposition(BaseEstimator, TransformerMixin):
    # PCA��ά
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dec_pca_data = X.copy()
        plotPCA(dec_pca_data)    # �ҹյ����������Ϊn_components��������ߺ�ƽ����û�����Թյ㣬�Ͳ�ʹ��pca
        # N_COMPONENTS = 20
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=N_COMPONENTS)
        # dec_pca_data = pd.DataFrame(data=pca.fit_transform(X), columns=('pca_'+ str(item) for item in range(N_COMPONENTS)))
        # print('PCA��ά������ݣ�\n', dec_pca_data) if SHOW_LOG == True else ()
        return dec_pca_data

class ModelTraining(BaseEstimator, TransformerMixin):
    # ѵ��ģ��
    def __init__(self, label_data, algorithm, params):
        self.label_data = label_data
        self.algorithm = algorithm
        self.params = params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # 1.�����з�
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, self.label_data, test_size=0.33, random_state=42)
        # 2.����ģ��
        estimator = ALGORITHMS.get(self.algorithm)(**self.params)
        # 3.ѵ��ģ��
        estimator.fit(X_train, y_train)
        # 4.��֤
        y_pred = estimator.predict(X_test)
        y_predprob = estimator.predict_proba(X_test)[:, 1]
        # 5.ģ��У��
        from sklearn import metrics
        print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
        # 6.����ģ���ļ�
        import time
        year = str(time.localtime().tm_year)
        month = str(time.localtime().tm_mon)
        day = str(time.localtime().tm_mday)
        hour = str(time.localtime().tm_hour)
        minute = str(time.localtime().tm_min)
        second = str(time.localtime().tm_sec)
        model_name = self.algorithm+'_'+str(metrics.roc_auc_score(y_test, y_predprob))+'_'+year+month+day+hour+minute+second+'.pkl'
        import os
        if not os.path.exists(MODEL_FILEPATH):
            os.makedirs(MODEL_FILEPATH)
        import pickle
        with open(MODEL_FILEPATH+model_name, 'wb') as f:
            pickle.dump(estimator, f)

def Main():

    # ����ԭʼ����
    ori_data = readFileToDataFrame(TRAIN_DATA_FILEPATH, ENCODING, MISSING_VALUES)
    TRAIN_LABEL = ori_data[TARGET]
    TRAIN_DATA = ori_data.drop(TARGET, axis=1)
    print('ԭʼ���ݣ�\n',ori_data.head(10)) if SHOW_LOG == True else ()

    # �����жϷ��ࡢɾ���ֶ�
    makeAnalysisFile(ANALYSIS_FILEPATH, TRAIN_DATA)

    gbdt_params = {
        'loss' : 'deviance',
        'learning_rate' : 0.005,
        'n_estimators' : 2,
        'subsample' : 1.0,
        'criterion' : 'friedman_mse',
        'min_samples_split' : 2,
        'min_samples_leaf' : 1,
        'min_weight_fraction_leaf' : 0.,
        'max_depth' : 3,
        'min_impurity_decrease' : 0.,
        'min_impurity_split' : None,
        'init' : None,
        'random_state' : None,
        'max_features' : None,
        'verbose' : 0,
        'max_leaf_nodes' : None,
        'warm_start' : False,
        'validation_fraction' : 0.1,
        'n_iter_no_change' : None,
        'tol' : 1e-4,
        'ccp_alpha' : 0.
    }

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
                           ('train', ModelTraining(TRAIN_LABEL, 'GBDT', gbdt_params))])
    pipeline.fit_transform(TRAIN_DATA)

    # Ƕ�뷨�Ͱ�װ����ɸ������������ѡ���Ϊ��������ʽ���ԣ�������ݽ���û������Ƿ�ѡ�����µ�ѡ�񷽷�

    # # Ƕ�뷨�������㷨ģ����ʵ��֤��������ԣ��õ�feature_importance���ʺ�����������ɸ��Ѱ����Ϊ��ص�������ע������ʱ��ϳ�
    # EMB_ALGORITHM = 'RFC'
    # EMB_N_ESTIMATORS = 10
    # # EMB_PARTITIONS = 10
    # # EMB_CV = 5
    # # emb_threshold = plotEmbedded(EMB_ALGORITHM, EMB_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], EMB_PARTITIONS, EMB_CV)
    # # print('Ƕ�뷨����ֵΪ��\n', emb_threshold) if SHOW_LOG == True else ()
    # emb_threshold = 0.07125063715469972
    # print('Ƕ�뷨����ֵΪ��\n', emb_threshold) if SHOW_LOG == True else ()
    # emb_sel_data = chi2_sel_data.copy()
    # print('Ƕ�뷨ѡ�������ݣ�\n', emb_sel_data) if SHOW_LOG == True else ()
    # # from sklearn.feature_selection import SelectFromModel
    # # emb_estimator = ALGORITHMS.get(EMB_ALGORITHM)(n_estimators=EMB_N_ESTIMATORS, random_state=0)
    # # selectfrommodel = SelectFromModel(emb_estimator, threshold=emb_threshold)
    # # emb_sel_data = pd.DataFrame(data=selectfrommodel.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in selectfrommodel.get_support(indices=True)))
    # # print('Ƕ�뷨ѡ�������ݣ�\n', emb_sel_data) if SHOW_LOG == True else ()
    #
    # # ��װ�����ں�ѡȡ�����Ӽ�������ר�õ��㷨RFE����ɸѡ������Ҫѧϰ������������ֵ��
    # WRAP_ALGORITHM = 'RFC'
    # WRAP_N_ESTIMATORS = 10
    # WRAP_STEP = 5    # RFEÿ�ε���ʱɸ�����ٸ�����
    # WRAP_CV = 3
    # # wrap_n_features_to_select = plotWrapper(WRAP_ALGORITHM, WRAP_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], WRAP_STEP, WRAP_CV)
    # wrap_sel_data = emb_sel_data.copy()
    # print('��װ��ѡ�������ݣ�\n', wrap_sel_data) if SHOW_LOG == True else ()
    # # from sklearn.feature_selection import RFE
    # # wrap_estimator = ALGORITHMS.get(WRAP_ALGORITHM)(n_estimators=WRAP_N_ESTIMATORS, random_state=0)
    # # rfe = RFE(wrap_estimator, wrap_n_features_to_select, step=WRAP_STEP)
    # # print('��װ��������Ҫ��������\n', rfe.fit(chi2_sel_data, ori_data[TARGET]).ranking_) if SHOW_LOG == True else ()
    # # wrap_sel_data = pd.DataFrame(data=rfe.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in rfe.get_support(indices=True)))
    # # print('��װ��ѡ�������ݣ�\n', wrap_sel_data) if SHOW_LOG == True else ()

    # ------------------------------------------------------------------- #



if __name__ == "__main__":
    Main()