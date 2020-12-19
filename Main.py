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
ORDER_CATEGORICAL_FEATURES = ['列A']
DISORDER_CATEGORICAL_FEATURES = ['列B']
CONTINUOUS_FEATURES = ['列C']


def Main():

    # 导入原始数据
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('原始数据：\n',ori_data) if SHOW_LOG == True else ()

    # ------------------------------ 预处理 ------------------------------ #

    # 补缺失值
    global impute_data
    if len(ori_data.columns) > 0:
        from sklearn.impute import SimpleImputer
        import numpy as np
        simpleimputer = SimpleImputer(missing_values=np.nan)
        impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data), columns=ori_data.columns)
        print('补缺失值：\n',impute_data) if SHOW_LOG == True else ()
    else:
        print('无数据！') if SHOW_LOG == True else ()

    # 人为确定特征类型为连续型还是类别型，分别处理
    # 另：后续还需考虑如果特征标签未知，如何使用程序初步判断类型
    print('连续型特征：\n', CONTINUOUS_FEATURES) if SHOW_LOG == True else ()
    print('有序类别型特征：\n', ORDER_CATEGORICAL_FEATURES) if SHOW_LOG == True else ()
    print('无序类别型特征：\n', DISORDER_CATEGORICAL_FEATURES) if SHOW_LOG == True else ()

    # 连续型：标准化(包括StandardScaler, MinMaxScaler, MaxAbsScaler等)
    scale_con_data = pd.DataFrame()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import StandardScaler
        standardscaler = StandardScaler().fit(impute_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=standardscaler.transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('连续型特征标准化：\n',scale_con_data) if SHOW_LOG == True else ()
    else:
        print('无需要标准化的数据！') if SHOW_LOG == True else ()

    # 类别型，对于有序特征，直接标签化
    encode_order_cat_data = pd.DataFrame()
    if len(ORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OrdinalEncoder
        OrdinalEncoder().fit(impute_data[ORDER_CATEGORICAL_FEATURES]).categories_
        encode_order_cat_data = pd.DataFrame(data=OrdinalEncoder().fit_transform(impute_data[ORDER_CATEGORICAL_FEATURES]), columns=ORDER_CATEGORICAL_FEATURES)
        print('有序类别型特征标签化：\n', encode_order_cat_data) if SHOW_LOG == True else ()
    else:
        print('无需要标签化的数据！') if SHOW_LOG == True else ()

    # 类别型，对于无序特征，onehot
    encode_disorder_cat_data = pd.DataFrame()
    if len(DISORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OneHotEncoder
        if len(DISORDER_CATEGORICAL_FEATURES) == 1:
            onehotencoder = OneHotEncoder(categories='auto').fit(impute_data[DISORDER_CATEGORICAL_FEATURES])
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns = onehotencoder.categories_)
        else:
            onehotencoder = OneHotEncoder(categories='auto').fit(impute_data[DISORDER_CATEGORICAL_FEATURES])
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns = np.append(*onehotencoder.categories_))
        print('无序类别型特征独热编码：\n', encode_disorder_cat_data) if SHOW_LOG == True else ()
    else:
        print('无需要onehot的数据！') if SHOW_LOG == True else ()

    # 合并
    pre_data = pd.concat([scale_con_data, encode_order_cat_data, encode_disorder_cat_data], axis=1, ignore_index=False)
    print('预处理后数据：\n', pre_data) if SHOW_LOG == True else ()

    # ------------------------------------------------------------------- #

    # ----------------------------- 特征工程 ----------------------------- #



    # ------------------------------------------------------------------- #

if __name__ == "__main__":
    Main()