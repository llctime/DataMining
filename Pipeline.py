# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# ����
SHOW_LOG = True    # ��־�����ʶ

TRAIN_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\train.csv'    # ѵ�������ļ�·��
TEST_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\testA.csv'    # ���������ļ�·��
ENCODING = 'GBK'    # ���ļ������ʽ
MISSING_VALUES = 'NaN'    # ԭʼ����ȱʧֵ

ANALYSIS_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\��ش���ΥԼԤ��ѵ�������ֶη���.csv'
TARGET = 'isDefault'    # ��ǩ
ALGORITHMS = {
    'RFC': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).RandomForestClassifier
}    # �㷨�б�

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
            print('�ַ���������ǩ����\n', ordinal_encode_data[string_features]) if SHOW_LOG == True else ()
        else:
            print('����Ҫ��ǩ�������ݣ�') if SHOW_LOG == True else ()
        return ordinal_encode_data

def Main():

    # ����ԭʼ����
    ori_data = readFileToDataFrame(TRAIN_DATA_FILEPATH, ENCODING, MISSING_VALUES)
    print('ԭʼ���ݣ�\n',ori_data.head(10)) if SHOW_LOG == True else ()

    # �����жϷ��ࡢɾ���ֶ�
    makeAnalysisFile(ANALYSIS_FILEPATH, ori_data)
    import pandas as pd
    # analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
    # del_features = selectFeatures(analysis_data, ['ɾ��'], 'transform', 'colname')
    # print('Ҫɾ�����ֶΣ�\n',del_features) if SHOW_LOG == True else ()
    # del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
    # print('�����ļ�����ɾ���ֶκ�ʣ������ݣ�\n',del_ana_data) if SHOW_LOG == True else ()

    from sklearn.pipeline import Pipeline
    # pipeline_1 = Pipeline([('del_Columns', ColumnDeletion())])
    # after_del_data = pipeline_1.fit_transform(ori_data)
    # print('ɾ����ʣ������ݣ�\n',after_del_data.head(10)) if SHOW_LOG == True else ()

    # ------------------------------ Ԥ���� ------------------------------ #

    # �Ƚ�����Ԥ����׶ε����ݽ��л��������б���ֵ��/�ַ��ͣ�
    # numric_features = selectFeatures(del_ana_data, ['int','float'], 'type', 'colname')
    # print('��ֵ��������\n',numric_features) if SHOW_LOG == True else ()
    # string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
    # print('�ַ���������\n',string_features) if SHOW_LOG == True else ()

    # ��ȱʧֵ
    # ��ֵ��
    # impute_data = after_del_data.copy()
    # from sklearn.impute import SimpleImputer
    # import numpy as np
    # simpleimputer_numric = SimpleImputer(missing_values=np.nan)
    # impute_data[numric_features] = pd.DataFrame(data=simpleimputer_numric.fit_transform(after_del_data[numric_features]), columns=numric_features)
    # print('��ֵ��������ȱʧֵ��\n', impute_data.head(10)) if SHOW_LOG == True else ()
    # �ַ���
    # string_features_most_frequent = impute_data[string_features].mode()
    # print('�ַ�������������\n', string_features_most_frequent) if SHOW_LOG == True else ()
    # for col in string_features_most_frequent.columns:
    #     simpleimputer_string = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=string_features_most_frequent[col])
    #     impute_data[col] = pd.DataFrame(data=simpleimputer_string.fit_transform(after_del_data[col].values.reshape(-1, 1)), columns=[col])
    # print('�ַ���������ȱʧֵ��\n', impute_data.head(10)) if SHOW_LOG == True else ()

    # from sklearn.pipeline import Pipeline
    # pipeline_2 = Pipeline([('del_Columns', ColumnDeletion()),('impute_miss', MissingImputation())])
    # impute_data = pipeline_2.fit_transform(ori_data)
    # print('������ȱʧֵ��\n', impute_data.head(10)) if SHOW_LOG == True else ()

    # �����ַ��ͣ��ȱ�ǩ��תΪ��ֵ�ͺ���ֵ�ʹ���
    # ������ֵ�ͣ���Ϊȷ����������Ϊ�����ͻ�������ͣ��ֱ���
    # ���������迼�����������ǩδ֪�����ʹ�ó�������ж�����

    # �ַ��ͣ�ֱ�ӱ�ǩ�������ж���������ַ�����������ǩ����Ϳ�ֱ��ʹ�ã���������������ַ��������ٽ���onehot
    # ordinal_encode_data = impute_data.copy()
    # if len(string_features) > 0:
    #     from sklearn.preprocessing import OrdinalEncoder
    #     OrdinalEncoder().fit(impute_data[string_features]).categories_
    #     ordinal_encode_data[string_features] = pd.DataFrame(
    #         data=OrdinalEncoder().fit_transform(impute_data[string_features]),
    #         columns=string_features)
    #     print('�ַ���������ǩ����\n', ordinal_encode_data[string_features]) if SHOW_LOG == True else ()
    # else:
    #     print('����Ҫ��ǩ�������ݣ�') if SHOW_LOG == True else ()
    pipeline_3 = Pipeline([('del_Columns', ColumnDeletion()),('impute_miss', MissingImputation()),('digi_string', Digitization())])
    ordinal_encode_data = pipeline_3.fit_transform(ori_data)
    # print('�ַ���������ǩ����\n', ordinal_encode_data.head(10)) if SHOW_LOG == True else ()

    # �����ͣ���ֵ��
    bin_con_data = ordinal_encode_data.copy()
    BINARY_CONTINUOUS_FEATURES = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'colname')    # ���ֵ��������������
    BINARY_THRESHOLD = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'threshold')    # ��ֵ������ֵ
    if len(BINARY_CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import Binarizer
        for i in range(len(BINARY_CONTINUOUS_FEATURES)):
            binarizer = Binarizer(threshold = int(float(BINARY_THRESHOLD[i])))
            bin_data = pd.DataFrame(data=binarizer.fit_transform(bin_con_data[BINARY_CONTINUOUS_FEATURES[i]].values.reshape(-1, 1)), columns=[BINARY_CONTINUOUS_FEATURES[i]])
            print('������������ֵ����\n',bin_data.head(10)) if SHOW_LOG == True else ()
            bin_con_data[BINARY_CONTINUOUS_FEATURES[i]] = bin_data
        print('�����ֵ���������Ϊ��\n', bin_con_data[BINARY_CONTINUOUS_FEATURES].head(10)) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��ֵ�������ݣ�') if SHOW_LOG == True else ()

    # �����ͣ���׼��(����StandardScaler, MinMaxScaler, MaxAbsScaler��)
    CONTINUOUS_FEATURES = bin_con_data.columns    # ����������
    scale_con_data = bin_con_data.copy()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import MinMaxScaler
        minmaxScaler = MinMaxScaler().fit(scale_con_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=minmaxScaler.fit_transform(scale_con_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('������������׼����\n',scale_con_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��׼�������ݣ�') if SHOW_LOG == True else ()

    # ����ͣ���������������onehot
    DISORDER_CATEGORICAL_FEATURES = selectFeatures(del_ana_data, ['OneHotEncoder'], 'transform', 'colname')    # �������������
    encode_disorder_cat_data = scale_con_data.copy()
    if len(DISORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OneHotEncoder
        if len(DISORDER_CATEGORICAL_FEATURES) == 1:
            onehotencoder = OneHotEncoder(categories='auto').fit(scale_con_data[DISORDER_CATEGORICAL_FEATURES])
            tmp_l = encode_disorder_cat_data.drop(DISORDER_CATEGORICAL_FEATURES,axis=1)
            tmp_r = pd.DataFrame(data=onehotencoder.fit_transform(scale_con_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(onehotencoder.categories_[0]))))
            encode_disorder_cat_data = pd.concat([tmp_l, tmp_r], axis=1, ignore_index=False)
        else:
            onehotencoder = OneHotEncoder(categories='auto').fit(scale_con_data[DISORDER_CATEGORICAL_FEATURES])
            tmp_l = encode_disorder_cat_data.drop(DISORDER_CATEGORICAL_FEATURES,axis=1)
            tmp_r = pd.DataFrame(data=onehotencoder.fit_transform(scale_con_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(np.append(*onehotencoder.categories_)))))
            encode_disorder_cat_data = pd.concat([tmp_l, tmp_r], axis=1, ignore_index=False)
        print('����������������ȱ��룺\n', encode_disorder_cat_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('����Ҫonehot�����ݣ�') if SHOW_LOG == True else ()

    # ------------------------------- END ------------------------------- #

    # ----------------------------- �������� ----------------------------- #

    # ����������������
    MINUS_FIRST_FEATURES = ['ficoRangeLow']
    MINUS_SECOND_FRATURES = ['ficoRangeHigh']
    cal_minus_data = encode_disorder_cat_data.copy()
    if len(MINUS_FIRST_FEATURES) > 0:
        cal_minus_data = minusFeatures(encode_disorder_cat_data, MINUS_FIRST_FEATURES, MINUS_SECOND_FRATURES)
        print('�������������������ݣ�\n', cal_minus_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('����Ҫ������������ݣ�') if SHOW_LOG == True else ()

    # ���ɶ���ʽ����������������
    POLYNOMIAL_FEATURES = []    # �����ɶ���ʽ����������
    polynomial_data = pd.DataFrame()
    if len(POLYNOMIAL_FEATURES) > 0:
        from sklearn.preprocessing import PolynomialFeatures
        polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        polynomial_data = pd.DataFrame(data=polynomialfeatures.fit_transform(cal_minus_data[POLYNOMIAL_FEATURES]), columns=('poly_'+ str(item) for item in range(polynomialfeatures.n_output_features_)))
        print('���ɶ���ʽ�������ݣ�\n', polynomial_data) if SHOW_LOG == True else ()
    else:
        print('�������ɶ���ʽ������') if SHOW_LOG == True else ()
    poly_data = pd.concat([cal_minus_data, polynomial_data], axis=1, ignore_index=False)
    print('�������ʽ���������ݣ�\n', poly_data) if SHOW_LOG == True else ()

    # ���˷�������ѡ�񡢿������飨�ʺϳ���������Ŀ�϶�ʱ��ɸ�����ǩ���Բ���ص�������
    # 1. ����ѡ��ֻɸ������Ϊ0����ֻ��һ��ȡֵ������
    VARIANCE_THRESHOLD = 0    # ����ѡ����ֵ
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('����ѡ�������ݣ�\n', var_sel_data) if SHOW_LOG == True else ()

    # 2. ������������ѡ��ע������ʱ��ϳ��������һ������֮���ɸѡ��������б��棬֮�����ܴ˹���
    chi2_sel_data = var_sel_data.copy()
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
    chi2_sel_data = chi2_sel_data[CHI2_FEATURES]
    print('����ѡ����������\n', CHI2_FEATURES) if SHOW_LOG == True else ()
    print('����ѡ�������ݣ�\n', chi2_sel_data.head(10)) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import mutual_info_classif as MIC
    # mic = MIC(var_sel_data, ori_data[TARGET])    # ����Ϣ����֤�������ǩ������ԣ����Ժͷ����ԣ���0Ϊ�໥������1Ϊ���
    # CHI_K = mic.shape[0] - (mic <= 0).sum()    # kֵ������������ȥp>0.05���������õ�
    # print('����ѡ���kֵ��\n', CHI_K) if SHOW_LOG == True else ()
    # if CHI_K > 0:
    #     from sklearn.feature_selection import SelectKBest
    #     from sklearn.feature_selection import chi2
    #     selectkbest = SelectKBest(chi2, k=CHI_K)
    #     chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(var_sel_data, ori_data[TARGET]), columns=(var_sel_data.columns[item] for item in selectkbest.get_support(indices=True)))
    # else:
    #     print('������п�������ѡ��') if SHOW_LOG == True else ()
    # print('����ѡ����������\n', chi2_sel_data.columns) if SHOW_LOG == True else ()
    # print('����ѡ�������ݣ�\n', chi2_sel_data.head(10)) if SHOW_LOG == True else ()

    # Ƕ�뷨�Ͱ�װ����ɸ������������ѡ���Ϊ��������ʽ���ԣ�������ݽ���û������Ƿ�ѡ�����µ�ѡ�񷽷�

    # Ƕ�뷨�������㷨ģ����ʵ��֤��������ԣ��õ�feature_importance���ʺ�����������ɸ��Ѱ����Ϊ��ص�������ע������ʱ��ϳ�
    EMB_ALGORITHM = 'RFC'
    EMB_N_ESTIMATORS = 10
    # EMB_PARTITIONS = 10
    # EMB_CV = 5
    # emb_threshold = plotEmbedded(EMB_ALGORITHM, EMB_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], EMB_PARTITIONS, EMB_CV)
    # print('Ƕ�뷨����ֵΪ��\n', emb_threshold) if SHOW_LOG == True else ()
    emb_threshold = 0.07125063715469972
    print('Ƕ�뷨����ֵΪ��\n', emb_threshold) if SHOW_LOG == True else ()
    emb_sel_data = chi2_sel_data.copy()
    print('Ƕ�뷨ѡ�������ݣ�\n', emb_sel_data) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import SelectFromModel
    # emb_estimator = ALGORITHMS.get(EMB_ALGORITHM)(n_estimators=EMB_N_ESTIMATORS, random_state=0)
    # selectfrommodel = SelectFromModel(emb_estimator, threshold=emb_threshold)
    # emb_sel_data = pd.DataFrame(data=selectfrommodel.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in selectfrommodel.get_support(indices=True)))
    # print('Ƕ�뷨ѡ�������ݣ�\n', emb_sel_data) if SHOW_LOG == True else ()

    # ��װ�����ں�ѡȡ�����Ӽ�������ר�õ��㷨RFE����ɸѡ������Ҫѧϰ������������ֵ��
    WRAP_ALGORITHM = 'RFC'
    WRAP_N_ESTIMATORS = 10
    WRAP_STEP = 5    # RFEÿ�ε���ʱɸ�����ٸ�����
    WRAP_CV = 3
    # wrap_n_features_to_select = plotWrapper(WRAP_ALGORITHM, WRAP_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], WRAP_STEP, WRAP_CV)
    wrap_sel_data = emb_sel_data.copy()
    print('��װ��ѡ�������ݣ�\n', wrap_sel_data) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import RFE
    # wrap_estimator = ALGORITHMS.get(WRAP_ALGORITHM)(n_estimators=WRAP_N_ESTIMATORS, random_state=0)
    # rfe = RFE(wrap_estimator, wrap_n_features_to_select, step=WRAP_STEP)
    # print('��װ��������Ҫ��������\n', rfe.fit(chi2_sel_data, ori_data[TARGET]).ranking_) if SHOW_LOG == True else ()
    # wrap_sel_data = pd.DataFrame(data=rfe.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in rfe.get_support(indices=True)))
    # print('��װ��ѡ�������ݣ�\n', wrap_sel_data) if SHOW_LOG == True else ()

    # PCA��ά
    dec_pca_data = wrap_sel_data.copy()
    # plotPCA(dec_pca_data)    # �ҹյ����������Ϊn_components��������ߺ�ƽ����û�����Թյ㣬�Ͳ�ʹ��pca
    # N_COMPONENTS = 20
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=N_COMPONENTS)
    # dec_pca_data = pd.DataFrame(data=pca.fit_transform(dec_pca_data), columns=('pca_'+ str(item) for item in range(N_COMPONENTS)))
    print('PCA��ά������ݣ�\n', dec_pca_data) if SHOW_LOG == True else ()




    # ------------------------------------------------------------------- #



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

def minusFeatures(X, first_operators, second_operators):
    for i in range(len(first_operators)):
        X[first_operators[i]+'-'+second_operators[i]] = X[first_operators[i]] - X[second_operators[i]]
        X.drop([first_operators[i], second_operators[i]], 1, inplace=True)
    return X

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



if __name__ == "__main__":
    Main()