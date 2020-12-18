# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# ����
SHOW_LOG = True
DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'
ENCODING = 'GBK'
MISSING_VALUES = 'NaN'
CATEGORICAL_FEATURES = ['��A','��B']
CONTINUOUS_FEATURES = ['��C']


def Main():

    # ����ԭʼ����
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('ԭʼ���ݣ�\n',ori_data) if SHOW_LOG == True else ()

    # ��ȱʧֵ
    from sklearn.impute import SimpleImputer
    import numpy as np
    simpleimputer = SimpleImputer(missing_values=np.nan)
    impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data), columns=ori_data.columns)
    print('��ȱʧֵ��\n',impute_data) if SHOW_LOG == True else ()

    # ȷ����������Ϊ�����ͻ�������ͣ��ֱ���
    print('������������\n', CONTINUOUS_FEATURES) if SHOW_LOG == True else ()
    print('�����������\n', CATEGORICAL_FEATURES) if SHOW_LOG == True else ()

    # �����ͣ���׼��(����StandardScaler, MinMaxScaler, MaxAbsScaler��)
    from sklearn.preprocessing import MinMaxScaler
    minmaxscaler = MinMaxScaler().fit(impute_data[CONTINUOUS_FEATURES])
    scale_con_data = pd.DataFrame(data=minmaxscaler.transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
    print('������������׼����\n',scale_con_data) if SHOW_LOG == True else ()

if __name__ == "__main__":
    Main()