# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# 常量
SHOW_LOG = True
DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'
ENCODING = 'GBK'
MISSING_VALUES = 'NaN'
CATEGORICAL_FEATURES = ['列A','列B']
CONTINUOUS_FEATURES = ['列C']


def Main():

    # 导入原始数据
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('原始数据：\n',ori_data) if SHOW_LOG == True else ()

    # 补缺失值
    from sklearn.impute import SimpleImputer
    import numpy as np
    simpleimputer = SimpleImputer(missing_values=np.nan)
    impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data), columns=ori_data.columns)
    print('补缺失值：\n',impute_data) if SHOW_LOG == True else ()

    # 确定特征类型为连续型还是类别型，分别处理
    print('连续型特征：\n', CONTINUOUS_FEATURES) if SHOW_LOG == True else ()
    print('类别型特征：\n', CATEGORICAL_FEATURES) if SHOW_LOG == True else ()

    # 连续型：标准化(包括StandardScaler, MinMaxScaler, MaxAbsScaler等)
    from sklearn.preprocessing import MinMaxScaler
    minmaxscaler = MinMaxScaler().fit(impute_data[CONTINUOUS_FEATURES])
    scale_con_data = pd.DataFrame(data=minmaxscaler.transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
    print('连续型特征标准化：\n',scale_con_data) if SHOW_LOG == True else ()

if __name__ == "__main__":
    Main()