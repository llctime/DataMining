# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# ����
SHOW_LOG = True    # ��־�����ʶ
TARGET = 'label'    # ��ǩ

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

    # ����ѡ��ֻɸ������Ϊ0����ֻ��һ��ȡֵ������
    VARIANCE_THRESHOLD = 0    # ����ѡ����ֵ
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('����ѡ�������ݣ�\n', var_sel_data) if SHOW_LOG == True else ()

    # ������������ѡ��
    plotChi2(3, 1, -1, var_sel_data, ori_data[TARGET], 10, 2)
    CHI_K = 2    # k����ͼ���ߵĸߵ����
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    selectkbest = SelectKBest(chi2, k=CHI_K)
    chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(var_sel_data, ori_data[TARGET]), columns=(var_sel_data.columns[item] for item in selectkbest.get_support(indices=True)))
    print('����ѡ�������ݣ�\n', chi2_sel_data) if SHOW_LOG == True else ()

    # ------------------------------------------------------------------- #

def plotChi2(maxK, minK, step, X, y, n_estimators, cv):
    # ѡ��ͬ��kֵ��������֤��������֮�仯�����ߣ��԰����ҵ�����k����������kֵ������������ƣ���˵������������Ŀ�궼�й�������������ѡ��
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier as RFC
    score = []
    for i in range(maxK, minK, step):
        selectkbest = SelectKBest(chi2, k=i).fit_transform(X, y)
        once = cross_val_score(RFC(n_estimators=n_estimators, random_state=0), selectkbest, y, cv=cv).mean()
        score.append(once)
    plt.plot(range(maxK, minK, step), score)
    plt.show()

if __name__ == "__main__":
    Main()