# -*- ecoding: gbk -*-
# @ModuleName: Main
# @Function: the whole process of building a model
# @Author: llc
# @Time: 2020/12/19 1:20

# 常量
SHOW_LOG = True    # 日志输出标识

TRAIN_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\train.csv'    # 训练数据文件路径
TEST_DATA_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\data\\loan\\testA.csv'    # 测试数据文件路径
ENCODING = 'GBK'    # 读文件编码格式
MISSING_VALUES = 'NaN'    # 原始数据缺失值

ANALYSIS_FILEPATH = 'D:\\Dev\\Projects\\DataMining\\天池贷款违约预测训练数据字段分析.csv'
TARGET = 'isDefault'    # 标签
ALGORITHMS = {
    'RFC': __import__('sklearn.ensemble', globals={}, locals={}, fromlist=['ensemble']).RandomForestClassifier
}    # 算法列表

from sklearn.base import BaseEstimator, TransformerMixin

def readFileToDataFrame(filepath, encoding, missing_values):
    import pandas as pd
    data = pd.read_csv(filepath, na_values=missing_values, encoding=encoding)
    return data

def makeAnalysisFile(filepath, data):
    # 创建初始字段分析文件
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
        del_features = selectFeatures(analysis_data, ['删除'], 'transform', 'colname')
        after_del_data = X.drop(X[del_features], 1, inplace=False)
        print('删除后剩余的数据：\n', after_del_data.head(10)) if SHOW_LOG == True else ()
        return after_del_data

class MissingImputation(BaseEstimator, TransformerMixin):
    # 针对字符型和数值型特征分别补缺失值
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['删除'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        numric_features = selectFeatures(del_ana_data, ['int', 'float'], 'type', 'colname')
        string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
        impute_data = X.copy()
        from sklearn.impute import SimpleImputer
        import numpy as np
        import pandas as pd
        simpleimputer_numric = SimpleImputer(missing_values=np.nan)    # 数值型特征用均值补
        impute_data[numric_features] = pd.DataFrame(data=simpleimputer_numric.fit_transform(X[numric_features]), columns=numric_features)
        print('数值型特征补缺失值：\n', impute_data.head(10)) if SHOW_LOG == True else ()
        string_features_most_frequent = impute_data[string_features].mode()
        for col in string_features_most_frequent.columns:
            simpleimputer_string = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=string_features_most_frequent[col])    # 字符型特征用众数补
            impute_data[col] = pd.DataFrame(data=simpleimputer_string.fit_transform(X[col].values.reshape(-1, 1)), columns=[col])
        print('字符型特征补缺失值：\n', impute_data.head(10)) if SHOW_LOG == True else ()
        return impute_data

class Digitization(BaseEstimator, TransformerMixin):
    # 字符型特征标签化转为数值型
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
        del_features = selectFeatures(analysis_data, ['删除'], 'transform', 'colname')
        del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
        numric_features = selectFeatures(del_ana_data, ['int', 'float'], 'type', 'colname')
        string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
        ordinal_encode_data = X.copy()
        import pandas as pd
        if len(string_features) > 0:
            from sklearn.preprocessing import OrdinalEncoder
            OrdinalEncoder().fit(X[string_features]).categories_
            ordinal_encode_data[string_features] = pd.DataFrame(data=OrdinalEncoder().fit_transform(X[string_features]), columns=string_features)
            print('字符型特征标签化：\n', ordinal_encode_data[string_features]) if SHOW_LOG == True else ()
        else:
            print('无需要标签化的数据！') if SHOW_LOG == True else ()
        return ordinal_encode_data

def Main():

    # 导入原始数据
    ori_data = readFileToDataFrame(TRAIN_DATA_FILEPATH, ENCODING, MISSING_VALUES)
    print('原始数据：\n',ori_data.head(10)) if SHOW_LOG == True else ()

    # 初步判断分类、删除字段
    makeAnalysisFile(ANALYSIS_FILEPATH, ori_data)
    import pandas as pd
    # analysis_data = readFileToDataFrame(ANALYSIS_FILEPATH, ENCODING, MISSING_VALUES)
    # del_features = selectFeatures(analysis_data, ['删除'], 'transform', 'colname')
    # print('要删除的字段：\n',del_features) if SHOW_LOG == True else ()
    # del_ana_data = analysis_data[-analysis_data.colname.isin(del_features)]
    # print('分析文件数据删除字段后剩余的数据：\n',del_ana_data) if SHOW_LOG == True else ()

    from sklearn.pipeline import Pipeline
    # pipeline_1 = Pipeline([('del_Columns', ColumnDeletion())])
    # after_del_data = pipeline_1.fit_transform(ori_data)
    # print('删除后剩余的数据：\n',after_del_data.head(10)) if SHOW_LOG == True else ()

    # ------------------------------ 预处理 ------------------------------ #

    # 先将进入预处理阶段的数据进行基本类型判别（数值型/字符型）
    # numric_features = selectFeatures(del_ana_data, ['int','float'], 'type', 'colname')
    # print('数值型特征：\n',numric_features) if SHOW_LOG == True else ()
    # string_features = del_ana_data[-del_ana_data.colname.isin(numric_features)].colname.tolist()
    # print('字符型特征：\n',string_features) if SHOW_LOG == True else ()

    # 补缺失值
    # 数值型
    # impute_data = after_del_data.copy()
    # from sklearn.impute import SimpleImputer
    # import numpy as np
    # simpleimputer_numric = SimpleImputer(missing_values=np.nan)
    # impute_data[numric_features] = pd.DataFrame(data=simpleimputer_numric.fit_transform(after_del_data[numric_features]), columns=numric_features)
    # print('数值型特征补缺失值：\n', impute_data.head(10)) if SHOW_LOG == True else ()
    # 字符型
    # string_features_most_frequent = impute_data[string_features].mode()
    # print('字符型特征众数：\n', string_features_most_frequent) if SHOW_LOG == True else ()
    # for col in string_features_most_frequent.columns:
    #     simpleimputer_string = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=string_features_most_frequent[col])
    #     impute_data[col] = pd.DataFrame(data=simpleimputer_string.fit_transform(after_del_data[col].values.reshape(-1, 1)), columns=[col])
    # print('字符型特征补缺失值：\n', impute_data.head(10)) if SHOW_LOG == True else ()

    # from sklearn.pipeline import Pipeline
    # pipeline_2 = Pipeline([('del_Columns', ColumnDeletion()),('impute_miss', MissingImputation())])
    # impute_data = pipeline_2.fit_transform(ori_data)
    # print('特征补缺失值：\n', impute_data.head(10)) if SHOW_LOG == True else ()

    # 对于字符型，先标签化转为数值型后按数值型处理
    # 对于数值型，人为确定特征类型为连续型还是类别型，分别处理
    # 另：后续还需考虑如果特征标签未知，如何使用程序初步判断类型

    # 字符型，直接标签化，其中对于有序的字符型特征，标签化后就可直接使用，后续对于无序的字符型特征再进行onehot
    # ordinal_encode_data = impute_data.copy()
    # if len(string_features) > 0:
    #     from sklearn.preprocessing import OrdinalEncoder
    #     OrdinalEncoder().fit(impute_data[string_features]).categories_
    #     ordinal_encode_data[string_features] = pd.DataFrame(
    #         data=OrdinalEncoder().fit_transform(impute_data[string_features]),
    #         columns=string_features)
    #     print('字符型特征标签化：\n', ordinal_encode_data[string_features]) if SHOW_LOG == True else ()
    # else:
    #     print('无需要标签化的数据！') if SHOW_LOG == True else ()
    pipeline_3 = Pipeline([('del_Columns', ColumnDeletion()),('impute_miss', MissingImputation()),('digi_string', Digitization())])
    ordinal_encode_data = pipeline_3.fit_transform(ori_data)
    # print('字符型特征标签化：\n', ordinal_encode_data.head(10)) if SHOW_LOG == True else ()

    # 连续型：二值化
    bin_con_data = ordinal_encode_data.copy()
    BINARY_CONTINUOUS_FEATURES = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'colname')    # 需二值化的连续型特征
    BINARY_THRESHOLD = selectFeatures(del_ana_data, ['Binarizer'], 'transform', 'threshold')    # 二值化的阈值
    if len(BINARY_CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import Binarizer
        for i in range(len(BINARY_CONTINUOUS_FEATURES)):
            binarizer = Binarizer(threshold = int(float(BINARY_THRESHOLD[i])))
            bin_data = pd.DataFrame(data=binarizer.fit_transform(bin_con_data[BINARY_CONTINUOUS_FEATURES[i]].values.reshape(-1, 1)), columns=[BINARY_CONTINUOUS_FEATURES[i]])
            print('连续型特征二值化：\n',bin_data.head(10)) if SHOW_LOG == True else ()
            bin_con_data[BINARY_CONTINUOUS_FEATURES[i]] = bin_data
        print('加入二值化后的数据为：\n', bin_con_data[BINARY_CONTINUOUS_FEATURES].head(10)) if SHOW_LOG == True else ()
    else:
        print('无需要二值化的数据！') if SHOW_LOG == True else ()

    # 连续型：标准化(包括StandardScaler, MinMaxScaler, MaxAbsScaler等)
    CONTINUOUS_FEATURES = bin_con_data.columns    # 连续型特征
    scale_con_data = bin_con_data.copy()
    if len(CONTINUOUS_FEATURES) > 0:
        from sklearn.preprocessing import MinMaxScaler
        minmaxScaler = MinMaxScaler().fit(scale_con_data[CONTINUOUS_FEATURES])
        scale_con_data = pd.DataFrame(data=minmaxScaler.fit_transform(scale_con_data[CONTINUOUS_FEATURES]), columns=CONTINUOUS_FEATURES)
        print('连续型特征标准化：\n',scale_con_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('无需要标准化的数据！') if SHOW_LOG == True else ()

    # 类别型，对于无序特征，onehot
    DISORDER_CATEGORICAL_FEATURES = selectFeatures(del_ana_data, ['OneHotEncoder'], 'transform', 'colname')    # 无序类别型特征
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
        print('无序类别型特征独热编码：\n', encode_disorder_cat_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('无需要onehot的数据！') if SHOW_LOG == True else ()

    # ------------------------------- END ------------------------------- #

    # ----------------------------- 特征工程 ----------------------------- #

    # 生成算术运算特征
    MINUS_FIRST_FEATURES = ['ficoRangeLow']
    MINUS_SECOND_FRATURES = ['ficoRangeHigh']
    cal_minus_data = encode_disorder_cat_data.copy()
    if len(MINUS_FIRST_FEATURES) > 0:
        cal_minus_data = minusFeatures(encode_disorder_cat_data, MINUS_FIRST_FEATURES, MINUS_SECOND_FRATURES)
        print('生成算术运算特征数据：\n', cal_minus_data.head(10)) if SHOW_LOG == True else ()
    else:
        print('无需要算术相减的数据！') if SHOW_LOG == True else ()

    # 生成多项式特征（交叉特征）
    POLYNOMIAL_FEATURES = []    # 需生成多项式特征的特征
    polynomial_data = pd.DataFrame()
    if len(POLYNOMIAL_FEATURES) > 0:
        from sklearn.preprocessing import PolynomialFeatures
        polynomialfeatures = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        polynomial_data = pd.DataFrame(data=polynomialfeatures.fit_transform(cal_minus_data[POLYNOMIAL_FEATURES]), columns=('poly_'+ str(item) for item in range(polynomialfeatures.n_output_features_)))
        print('生成多项式特征数据：\n', polynomial_data) if SHOW_LOG == True else ()
    else:
        print('无需生成多项式特征！') if SHOW_LOG == True else ()
    poly_data = pd.concat([cal_minus_data, polynomial_data], axis=1, ignore_index=False)
    print('加入多项式特征后数据：\n', poly_data) if SHOW_LOG == True else ()

    # 过滤法：方差选择、卡方检验（适合初期特征数目较多时，筛掉与标签明显不相关的特征）
    # 1. 方差选择：只筛掉方差为0，即只有一种取值的特征
    VARIANCE_THRESHOLD = 0    # 方差选择阈值
    from sklearn.feature_selection import VarianceThreshold
    variancethreshold = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    var_sel_data = pd.DataFrame(data=variancethreshold.fit_transform(poly_data), columns=(poly_data.columns[item] for item in variancethreshold.get_support(indices=True)))
    print('方差选择后的数据：\n', var_sel_data) if SHOW_LOG == True else ()

    # 2. 卡方检验特征选择，注，运行时间较长，建议第一次跑完之后把筛选后的特征列保存，之后不再跑此过程
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
    print('卡方选择后的特征：\n', CHI2_FEATURES) if SHOW_LOG == True else ()
    print('卡方选择后的数据：\n', chi2_sel_data.head(10)) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import mutual_info_classif as MIC
    # mic = MIC(var_sel_data, ori_data[TARGET])    # 互信息法验证特征与标签的相关性（线性和非线性），0为相互独立，1为相关
    # CHI_K = mic.shape[0] - (mic <= 0).sum()    # k值由特征总数减去p>0.05的特征数得到
    # print('卡方选择的k值：\n', CHI_K) if SHOW_LOG == True else ()
    # if CHI_K > 0:
    #     from sklearn.feature_selection import SelectKBest
    #     from sklearn.feature_selection import chi2
    #     selectkbest = SelectKBest(chi2, k=CHI_K)
    #     chi2_sel_data = pd.DataFrame(data=selectkbest.fit_transform(var_sel_data, ori_data[TARGET]), columns=(var_sel_data.columns[item] for item in selectkbest.get_support(indices=True)))
    # else:
    #     print('无需进行卡方检验选择！') if SHOW_LOG == True else ()
    # print('卡方选择后的特征：\n', chi2_sel_data.columns) if SHOW_LOG == True else ()
    # print('卡方选择后的数据：\n', chi2_sel_data.head(10)) if SHOW_LOG == True else ()

    # 嵌入法和包装法精筛特征，但参数选择较为依赖启发式策略，故需根据结果好坏决定是否选用以下的选择方法

    # 嵌入法：利用算法模型真实验证特征相关性，得到feature_importance（适合特征经过粗筛后，寻找最为相关的特征）注，运行时间较长
    EMB_ALGORITHM = 'RFC'
    EMB_N_ESTIMATORS = 10
    # EMB_PARTITIONS = 10
    # EMB_CV = 5
    # emb_threshold = plotEmbedded(EMB_ALGORITHM, EMB_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], EMB_PARTITIONS, EMB_CV)
    # print('嵌入法的阈值为：\n', emb_threshold) if SHOW_LOG == True else ()
    emb_threshold = 0.07125063715469972
    print('嵌入法的阈值为：\n', emb_threshold) if SHOW_LOG == True else ()
    emb_sel_data = chi2_sel_data.copy()
    print('嵌入法选择后的数据：\n', emb_sel_data) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import SelectFromModel
    # emb_estimator = ALGORITHMS.get(EMB_ALGORITHM)(n_estimators=EMB_N_ESTIMATORS, random_state=0)
    # selectfrommodel = SelectFromModel(emb_estimator, threshold=emb_threshold)
    # emb_sel_data = pd.DataFrame(data=selectfrommodel.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in selectfrommodel.get_support(indices=True)))
    # print('嵌入法选择后的数据：\n', emb_sel_data) if SHOW_LOG == True else ()

    # 包装法：黑盒选取特征子集，并有专用的算法RFE进行筛选，不需要学习超参数（即阈值）
    WRAP_ALGORITHM = 'RFC'
    WRAP_N_ESTIMATORS = 10
    WRAP_STEP = 5    # RFE每次迭代时筛掉多少个特征
    WRAP_CV = 3
    # wrap_n_features_to_select = plotWrapper(WRAP_ALGORITHM, WRAP_N_ESTIMATORS, chi2_sel_data, ori_data[TARGET], WRAP_STEP, WRAP_CV)
    wrap_sel_data = emb_sel_data.copy()
    print('包装法选择后的数据：\n', wrap_sel_data) if SHOW_LOG == True else ()
    # from sklearn.feature_selection import RFE
    # wrap_estimator = ALGORITHMS.get(WRAP_ALGORITHM)(n_estimators=WRAP_N_ESTIMATORS, random_state=0)
    # rfe = RFE(wrap_estimator, wrap_n_features_to_select, step=WRAP_STEP)
    # print('包装法特征重要性排名：\n', rfe.fit(chi2_sel_data, ori_data[TARGET]).ranking_) if SHOW_LOG == True else ()
    # wrap_sel_data = pd.DataFrame(data=rfe.fit_transform(chi2_sel_data, ori_data[TARGET]), columns=(chi2_sel_data.columns[item] for item in rfe.get_support(indices=True)))
    # print('包装法选择后的数据：\n', wrap_sel_data) if SHOW_LOG == True else ()

    # PCA降维
    dec_pca_data = wrap_sel_data.copy()
    # plotPCA(dec_pca_data)    # 找拐点的特征数作为n_components，如果曲线很平滑，没有明显拐点，就不使用pca
    # N_COMPONENTS = 20
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=N_COMPONENTS)
    # dec_pca_data = pd.DataFrame(data=pca.fit_transform(dec_pca_data), columns=('pca_'+ str(item) for item in range(N_COMPONENTS)))
    print('PCA降维后的数据：\n', dec_pca_data) if SHOW_LOG == True else ()




    # ------------------------------------------------------------------- #



def selectFeatures(data, match_char_list, match_col, output_col):
    # 工具方法，用于将数据中的每个字段根据不同的匹配条件筛选出来
    # data: 数据
    # match_char_list: 数据中指定的匹配关键字
    # match_col: 指定数据中要匹配目标字段的列名
    # output_col: 指定匹配成功的行中要输出的字段
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