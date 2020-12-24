# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# ����
SHOW_LOG = True    # ��־�����ʶ
TARGET = 'label'    # ��ǩ
ALGORITHMS = {
    'RFC': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).RandomForestClassifier
}

def Main():

    # ����ԭʼ����
    DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'    # ԭʼ�����ļ�·��
    ENCODING = 'GBK'    # ���ļ������ʽ
    MISSING_VALUES = 'NaN'    # ԭʼ����ȱʧֵ
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('ԭʼ���ݣ�\n',ori_data) if SHOW_LOG == True else ()
    ori_data_withoutlabel = ori_data.loc[:, ori_data.columns != TARGET]

    # ------------------------------ Ԥ���� ------------------------------ #

    # ��ȱʧֵ
    impute_data = pd.DataFrame()
    if len(ori_data_withoutlabel.columns) > 0:
        from sklearn.impute import SimpleImputer
        import numpy as np
        simpleimputer = SimpleImputer(missing_values=np.nan)
        impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data_withoutlabel), columns=ori_data_withoutlabel.columns)
        print('��ȱʧֵ��\n',impute_data) if SHOW_LOG == True else ()
    else:
        print('�����ݣ�') if SHOW_LOG == True else ()

    # ��Ϊȷ����������Ϊ�����ͻ�������ͣ��ֱ���
    # ���������迼�����������ǩδ֪�����ʹ�ó�������ж�����

    # �����ͣ���ֵ��
    BINARY_CONTINUOUS_FEATURES = ['��B', '��C']    # ���ֵ��������������
    BINARY_THRESHOLD = 3    # ��ֵ������ֵ
    bin_con_data = pd.DataFrame()
    if len(BINARY_CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold = BINARY_THRESHOLD)
        bin_con_data = pd.DataFrame(data=binarizer.fit_transform(impute_data[BINARY_CONTINUOUS_FEATURES]), columns=BINARY_CONTINUOUS_FEATURES)
        print('������������ֵ����\n',bin_con_data) if SHOW_LOG == True else ()
        impute_data[BINARY_CONTINUOUS_FEATURES] = bin_con_data
        print('�����ֵ���������Ϊ��\n', impute_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��ֵ�������ݣ�') if SHOW_LOG == True else ()

    # �����ͣ���׼��(����StandardScaler, MinMaxScaler, MaxAbsScaler��)
    CONTINUOUS_FEATURES = ['��C']    # ����������
    scale_con_data = pd.DataFrame()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import MinMaxScaler
        minmaxScaler = MinMaxScaler().fit(impute_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=minmaxScaler.fit_transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('������������׼����\n',scale_con_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��׼�������ݣ�') if SHOW_LOG == True else ()

    # ����ͣ���������������ֱ�ӱ�ǩ��
    ORDER_CATEGORICAL_FEATURES = ['��A']    # �������������
    encode_order_cat_data = pd.DataFrame()
    if len(ORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OrdinalEncoder
        OrdinalEncoder().fit(impute_data[ORDER_CATEGORICAL_FEATURES]).categories_
        encode_order_cat_data = pd.DataFrame(data=OrdinalEncoder().fit_transform(impute_data[ORDER_CATEGORICAL_FEATURES]), columns=ORDER_CATEGORICAL_FEATURES)
        print('���������������ǩ����\n', encode_order_cat_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��ǩ�������ݣ�') if SHOW_LOG == True else ()

    # ����ͣ���������������onehot
    DISORDER_CATEGORICAL_FEATURES = ['��B']    # �������������
    encode_disorder_cat_data = pd.DataFrame()
    if len(DISORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OneHotEncoder
        if len(DISORDER_CATEGORICAL_FEATURES) == 1:
            onehotencoder = OneHotEncoder(categories='auto').fit(impute_data[DISORDER_CATEGORICAL_FEATURES])
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.fit_transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(onehotencoder.categories_[0]))))
        else:
            onehotencoder = OneHotEncoder(categories='auto').fit(impute_data[DISORDER_CATEGORICAL_FEATURES])
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.fit_transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(np.append(*onehotencoder.categories_)))))
        print('����������������ȱ��룺\n', encode_disorder_cat_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫonehot�����ݣ�') if SHOW_LOG == True else ()

    # �ϲ�
    pre_data = pd.concat([scale_con_data, encode_order_cat_data, encode_disorder_cat_data], axis=1, ignore_index=False)
    print('Ԥ��������ݣ�\n', pre_data) if SHOW_LOG == True else ()

    # ------------------------------- END ------------------------------- #

    # ----------------------------- �������� ----------------------------- #

    # ���ɶ���ʽ����������������
    POLYNOMIAL_FEATURES = ['��A','��C']    # �����ɶ���ʽ����������
    polynomial_data = pd.DataFrame()
    if len(POLYNOMIAL_FEATURES) > 0:
        from sklearn.preprocessing import PolynomialFeatures
        polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        polynomial_data = pd.DataFrame(data=polynomialfeatures.fit_transform(pre_data[POLYNOMIAL_FEATURES]), columns=('poly_'+ str(item) for item in range(polynomialfeatures.n_output_features_)))
        print('���ɶ���ʽ�������ݣ�\n', polynomial_data) if SHOW_LOG == True else ()
    else:
        print('�������ɶ���ʽ������') if SHOW_LOG == True else ()
    poly_data = pd.concat([pre_data, polynomial_data], axis=1, ignore_index=False)
    print('�������ʽ���������ݣ�\n', poly_data) if SHOW_LOG == True else ()

    # ���˷�������ѡ�񡢿������飨�ʺϳ���������Ŀ�϶�ʱ��ɸ�����ǩ���Բ���ص�������
    # 1. ����ѡ��ֻɸ������Ϊ0����ֻ��һ��ȡֵ������
    VARIANCE_THRESHOLD = 0    # ����ѡ����ֵ
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('����ѡ�������ݣ�\n', var_sel_data) if SHOW_LOG == True else ()

    # 2. ������������ѡ��
    chi2_sel_data = var_sel_data
    from sklearn.feature_selection import mutual_info_classif as MIC
    mic = MIC(var_sel_data, ori_data[TARGET])    # ����Ϣ����֤�������ǩ������ԣ����Ժͷ����ԣ���0Ϊ�໥������1Ϊ���
    CHI_K = mic.shape[0] - (mic <= 0).sum()    # kֵ������������ȥp>0.05���������õ�
    print('����ѡ���kֵ��\n', CHI_K) if SHOW_LOG == True else ()
    if CHI_K > 0:
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        selectkbest = SelectKBest(chi2, k=CHI_K)
        chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(var_sel_data, ori_data[TARGET]), columns=(var_sel_data.columns[item] for item in selectkbest.get_support(indices=True)))
    else:
        print('������п�������ѡ��') if SHOW_LOG == True else ()
    print('����ѡ�������ݣ�\n', chi2_sel_data) if SHOW_LOG == True else ()

    # Ƕ�뷨�������㷨ģ����ʵ��֤��������ԣ��õ�feature_importance���ʺ�����������ɸ��Ѱ����Ϊ��ص���������
    EMB_ALGORITHM = 'RFC'
    EMB_N_ESTIMATORS = 10
    EMB_PARTITIONS = 10
    EMB_CV = 2
    emb_threshold = plotEmbedded(EMB_ALGORITHM, EMB_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], EMB_PARTITIONS, EMB_CV)
    print('Ƕ�뷨����ֵΪ��\n', emb_threshold) if SHOW_LOG == True else ()
    from sklearn.feature_selection import SelectFromModel
    emb_estimator = ALGORITHMS.get(EMB_ALGORITHM)(n_estimators=EMB_N_ESTIMATORS, random_state=0)
    selectfrommodel = SelectFromModel(emb_estimator, threshold=emb_threshold)
    emb_sel_data = pd.DataFrame(data=selectfrommodel.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in selectfrommodel.get_support(indices=True)))
    print('Ƕ�뷨ѡ�������ݣ�\n', emb_sel_data) if SHOW_LOG == True else ()

    # ��װ�����ں�ѡȡ�����Ӽ�������ר�õ��㷨RFE����ɸѡ������Ҫѧϰ������������ֵ��
    WRAP_ALGORITHM = 'RFC'
    WRAP_N_ESTIMATORS = 10
    WRAP_STEP = 1    # RFEÿ�ε���ʱɸ�����ٸ�����
    WRAP_CV = 2
    wrap_n_features_to_select = plotWrapper(WRAP_ALGORITHM, WRAP_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], WRAP_STEP, WRAP_CV)
    from sklearn.feature_selection import RFE
    wrap_estimator = ALGORITHMS.get(WRAP_ALGORITHM)(n_estimators=WRAP_N_ESTIMATORS, random_state=0)
    rfe = RFE(wrap_estimator, wrap_n_features_to_select, step=WRAP_STEP)
    print('��װ��������Ҫ��������\n', rfe.fit(chi2_sel_data, ori_data[TARGET]).ranking_) if SHOW_LOG == True else ()
    wrap_sel_data = pd.DataFrame(data=rfe.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in rfe.get_support(indices=True)))
    print('��װ��ѡ�������ݣ�\n', wrap_sel_data) if SHOW_LOG == True else ()
    # ------------------------------------------------------------------- #

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


if __name__ == "__main__":
    Main()