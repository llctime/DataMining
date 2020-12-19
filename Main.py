# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# 常量
SHOW_LOG = True    # 日志输出标识
DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'    # 原始数据文件路径
ENCODING = 'GBK'    # 读文件编码格式
MISSING_VALUES = 'NaN'    # 原始数据缺失值
ORDER_CATEGORICAL_FEATURES = ['列A']    # 有序类别型特征
DISORDER_CATEGORICAL_FEATURES = ['列B']    # 无序类别型特征
CONTINUOUS_FEATURES = ['列C']    # 连续型特征
POLYNOMIAL_FEATURES = ['列A','列C']    # 需生成多项式特征的特征


def Main():

    # 导入原始数据
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('原始数据：\n',ori_data) if SHOW_LOG == True else ()

    # ------------------------------ 预处理 ------------------------------ #

    # 补缺失值
    impute_data = pd.DataFrame()
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
        scale_con_data = pd.DataFrame(data=standardscaler.fit_transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
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
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.fit_transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(onehotencoder.categories_[0]))))
        else:
            onehotencoder = OneHotEncoder(categories='auto').fit(impute_data[DISORDER_CATEGORICAL_FEATURES])
            encode_disorder_cat_data = pd.DataFrame(data=onehotencoder.fit_transform(impute_data[DISORDER_CATEGORICAL_FEATURES]).toarray(), columns=('onehot_'+ str(item) for item in range(len(np.append(*onehotencoder.categories_)))))
        print('无序类别型特征独热编码：\n', encode_disorder_cat_data) if SHOW_LOG == True else ()
    else:
        print('无需要onehot的数据！') if SHOW_LOG == True else ()

    # 合并
    pre_data = pd.concat([scale_con_data, encode_order_cat_data, encode_disorder_cat_data], axis=1, ignore_index=False)
    print('预处理后数据：\n', pre_data) if SHOW_LOG == True else ()

    # ------------------------------- END ------------------------------- #

    # ----------------------------- 特征工程 ----------------------------- #

    # 生成多项式特征（交叉特征）
    polynomial_data = pd.DataFrame()
    if len(POLYNOMIAL_FEATURES) > 0:
        from sklearn.preprocessing import PolynomialFeatures
        polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        polynomial_data = pd.DataFrame(data=polynomialfeatures.fit_transform(pre_data[POLYNOMIAL_FEATURES]), columns=('poly_'+ str(item) for item in range(polynomialfeatures.n_output_features_)))
        print('生成多项式特征数据：\n', polynomial_data) if SHOW_LOG == True else ()
    else:
        print('无需生成多项式特征！') if SHOW_LOG == True else ()
    poly_data = pd.concat([pre_data, polynomial_data], axis=1, ignore_index=False)
    print('加入多项式特征后数据：\n', poly_data) if SHOW_LOG == True else ()

    # 方差选择
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=0)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('方差选择后的数据：\n', var_sel_data) if SHOW_LOG == True else ()

    # ------------------------------------------------------------------- #

if __name__ == "__main__":
    Main()