# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# 常量
SHOW_LOG = True    # 日志输出标识
TARGET = 'label'    # 标签
ALGORITHMS = {
    'RFC': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).RandomForestClassifier
}

def Main():

    # 导入原始数据
    DATA_FILEPATH = 'D:\\Dev\\Data\\DataMining\\test.csv'    # 原始数据文件路径
    ENCODING = 'GBK'    # 读文件编码格式
    MISSING_VALUES = 'NaN'    # 原始数据缺失值
    import pandas as pd
    ori_data = pd.read_csv(DATA_FILEPATH, na_values=MISSING_VALUES, encoding=ENCODING)
    print('原始数据：\n',ori_data) if SHOW_LOG == True else ()
    ori_data_withoutlabel = ori_data.loc[:, ori_data.columns != TARGET]

    # ------------------------------ 预处理 ------------------------------ #

    # 补缺失值
    impute_data = pd.DataFrame()
    if len(ori_data_withoutlabel.columns) > 0:
        from sklearn.impute import SimpleImputer
        import numpy as np
        simpleimputer = SimpleImputer(missing_values=np.nan)
        impute_data = pd.DataFrame(data=simpleimputer.fit_transform(ori_data_withoutlabel), columns=ori_data_withoutlabel.columns)
        print('补缺失值：\n',impute_data) if SHOW_LOG == True else ()
    else:
        print('无数据！') if SHOW_LOG == True else ()

    # 人为确定特征类型为连续型还是类别型，分别处理
    # 另：后续还需考虑如果特征标签未知，如何使用程序初步判断类型

    # 连续型：二值化
    BINARY_CONTINUOUS_FEATURES = ['列B', '列C']    # 需二值化的连续型特征
    BINARY_THRESHOLD = 3    # 二值化的阈值
    bin_con_data = pd.DataFrame()
    if len(BINARY_CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold = BINARY_THRESHOLD)
        bin_con_data = pd.DataFrame(data=binarizer.fit_transform(impute_data[BINARY_CONTINUOUS_FEATURES]), columns=BINARY_CONTINUOUS_FEATURES)
        print('连续型特征二值化：\n',bin_con_data) if SHOW_LOG == True else ()
        impute_data[BINARY_CONTINUOUS_FEATURES] = bin_con_data
        print('加入二值化后的数据为：\n', impute_data) if SHOW_LOG == True else ()
    else:
        print('无需要二值化的数据！') if SHOW_LOG == True else ()

    # 连续型：标准化(包括StandardScaler, MinMaxScaler, MaxAbsScaler等)
    CONTINUOUS_FEATURES = ['列C']    # 连续型特征
    scale_con_data = pd.DataFrame()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import MinMaxScaler
        minmaxScaler = MinMaxScaler().fit(impute_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=minmaxScaler.fit_transform(impute_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('连续型特征标准化：\n',scale_con_data) if SHOW_LOG == True else ()
    else:
        print('无需要标准化的数据！') if SHOW_LOG == True else ()

    # 类别型，对于有序特征，直接标签化
    ORDER_CATEGORICAL_FEATURES = ['列A']    # 有序类别型特征
    encode_order_cat_data = pd.DataFrame()
    if len(ORDER_CATEGORICAL_FEATURES) > 0:
        from sklearn.preprocessing import OrdinalEncoder
        OrdinalEncoder().fit(impute_data[ORDER_CATEGORICAL_FEATURES]).categories_
        encode_order_cat_data = pd.DataFrame(data=OrdinalEncoder().fit_transform(impute_data[ORDER_CATEGORICAL_FEATURES]), columns=ORDER_CATEGORICAL_FEATURES)
        print('有序类别型特征标签化：\n', encode_order_cat_data) if SHOW_LOG == True else ()
    else:
        print('无需要标签化的数据！') if SHOW_LOG == True else ()

    # 类别型，对于无序特征，onehot
    DISORDER_CATEGORICAL_FEATURES = ['列B']    # 无序类别型特征
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
    POLYNOMIAL_FEATURES = ['列A','列C']    # 需生成多项式特征的特征
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

    # 过滤法：方差选择、卡方检验（适合初期特征数目较多时，筛掉与标签明显不相关的特征）
    # 1. 方差选择：只筛掉方差为0，即只有一种取值的特征
    VARIANCE_THRESHOLD = 0    # 方差选择阈值
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('方差选择后的数据：\n', var_sel_data) if SHOW_LOG == True else ()

    # 2. 卡方检验特征选择
    chi2_sel_data = var_sel_data
    from sklearn.feature_selection import mutual_info_classif as MIC
    mic = MIC(var_sel_data, ori_data[TARGET])    # 互信息法验证特征与标签的相关性（线性和非线性），0为相互独立，1为相关
    CHI_K = mic.shape[0] - (mic <= 0).sum()    # k值由特征总数减去p>0.05的特征数得到
    print('卡方选择的k值：\n', CHI_K) if SHOW_LOG == True else ()
    if CHI_K > 0:
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        selectkbest = SelectKBest(chi2, k=CHI_K)
        chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(var_sel_data, ori_data[TARGET]), columns=(var_sel_data.columns[item] for item in selectkbest.get_support(indices=True)))
    else:
        print('无需进行卡方检验选择！') if SHOW_LOG == True else ()
    print('卡方选择后的数据：\n', chi2_sel_data) if SHOW_LOG == True else ()

    # 嵌入法：利用算法模型真实验证特征相关性，得到feature_importance（适合特征经过粗筛后，寻找最为相关的特征，）
    EMB_ALGORITHM = 'RFC'
    EMB_N_ESTIMATORS = 10
    EMB_PARTITIONS = 10
    EMB_CV = 2
    emb_threshold = plotEmbedded(EMB_ALGORITHM, EMB_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], EMB_PARTITIONS, EMB_CV)
    print('嵌入法的阈值为：\n', emb_threshold) if SHOW_LOG == True else ()
    from sklearn.feature_selection import SelectFromModel
    emb_estimator = ALGORITHMS.get(EMB_ALGORITHM)(n_estimators=EMB_N_ESTIMATORS, random_state=0)
    selectfrommodel = SelectFromModel(emb_estimator, threshold=emb_threshold)
    emb_sel_data = pd.DataFrame(data=selectfrommodel.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in selectfrommodel.get_support(indices=True)))
    print('嵌入法选择后的数据：\n', emb_sel_data) if SHOW_LOG == True else ()

    # 包装法：黑盒选取特征子集，并有专用的算法RFE进行筛选，不需要学习超参数（即阈值）
    WRAP_ALGORITHM = 'RFC'
    WRAP_N_ESTIMATORS = 10
    WRAP_STEP = 1    # RFE每次迭代时筛掉多少个特征
    WRAP_CV = 2
    wrap_n_features_to_select = plotWrapper(WRAP_ALGORITHM, WRAP_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], WRAP_STEP, WRAP_CV)
    from sklearn.feature_selection import RFE
    wrap_estimator = ALGORITHMS.get(WRAP_ALGORITHM)(n_estimators=WRAP_N_ESTIMATORS, random_state=0)
    rfe = RFE(wrap_estimator, wrap_n_features_to_select, step=WRAP_STEP)
    print('包装法特征重要性排名：\n', rfe.fit(chi2_sel_data, ori_data[TARGET]).ranking_) if SHOW_LOG == True else ()
    wrap_sel_data = pd.DataFrame(data=rfe.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in rfe.get_support(indices=True)))
    print('包装法选择后的数据：\n', wrap_sel_data) if SHOW_LOG == True else ()
    # ------------------------------------------------------------------- #

def plotEmbedded(algorithm, n_estimators, X, y, partitions, cv):
    # time warning #
    # 嵌入法超参数(importance阈值)学习曲线
    # 参数：
    #   algorithm：模型算法，包括'RFC'（待完善）；
    #   n_estimators：评估器；
    #   partitions：feature_importance从0到max之间取点数
    #   cv：交叉验证折数
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
    # 包装法剩余特征参数(N_FEATURES_TO_SELECT)学习曲线
    # 参数：
    #   algorithm：模型算法，包括'RFC'（待完善）；
    #   n_estimators：评估器；
    #   step：每次迭代消除的特征数
    #   cv：交叉验证折数
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