# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# ����
SHOW_LOG = True    # ��־�����ʶ
DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'    # ԭʼ�����ļ�·��
ENCODING = 'GBK'    # ���ļ������ʽ
MISSING_VALUES = 'NaN'    # ԭʼ����ȱʧֵ
ORDER_CATEGORICAL_FEATURES = ['��A']    # �������������
DISORDER_CATEGORICAL_FEATURES = ['��B']    # �������������
CONTINUOUS_FEATURES = ['��C']    # ����������
POLYNOMIAL_FEATURES = ['��A','��C']    # �����ɶ���ʽ����������


def Main():

    # ����ԭʼ����
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('ԭʼ���ݣ�\n',ori_data) if SHOW_LOG == True else ()

    # ------------------------------ Ԥ���� ------------------------------ #

    # ��ȱʧֵ
    impute_data = pd.DataFrame()
    if len(ori_data.columns) > 0:
        from sklearn.impute import SimpleImputer
        import numpy as np
        simpleimputer = SimpleImputer(missing_values=np.nan)
        impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data), columns=ori_data.columns)
        print('��ȱʧֵ��\n',impute_data) if SHOW_LOG == True else ()
    else:
        print('�����ݣ�') if SHOW_LOG == True else ()

    # ��Ϊȷ����������Ϊ�����ͻ�������ͣ��ֱ���
    # ���������迼�����������ǩδ֪�����ʹ�ó�������ж�����
    print('������������\n', CONTINUOUS_FEATURES) if SHOW_LOG == True else ()
    print('���������������\n', ORDER_CATEGORICAL_FEATURES) if SHOW_LOG == True else ()
    print('���������������\n', DISORDER_CATEGORICAL_FEATURES) if SHOW_LOG == True else ()

    # �����ͣ���׼��(����StandardScaler, MinMaxScaler, MaxAbsScaler��)
    scale_con_data = pd.DataFrame()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import StandardScaler
        standardscaler = StandardScaler().fit(impute_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=standardscaler.fit_transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('������������׼����\n',scale_con_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��׼�������ݣ�') if SHOW_LOG == True else ()

    # ����ͣ���������������ֱ�ӱ�ǩ��
    encode_order_cat_data = pd.DataFrame()
    if len(ORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OrdinalEncoder
        OrdinalEncoder().fit(impute_data[ORDER_CATEGORICAL_FEATURES]).categories_
        encode_order_cat_data = pd.DataFrame(data=OrdinalEncoder().fit_transform(impute_data[ORDER_CATEGORICAL_FEATURES]), columns=ORDER_CATEGORICAL_FEATURES)
        print('���������������ǩ����\n', encode_order_cat_data) if SHOW_LOG == True else ()
    else:
        print('����Ҫ��ǩ�������ݣ�') if SHOW_LOG == True else ()

    # ����ͣ���������������onehot
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

    # ����ѡ��
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=0)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('����ѡ�������ݣ�\n', var_sel_data) if SHOW_LOG == True else ()

    # ------------------------------------------------------------------- #

if __name__ == "__main__":
    Main()