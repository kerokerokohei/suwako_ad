# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:23:00 2023

@author: kohei
"""

    # def grid_search(self, X, y):
    #     param_grid = {
    #         'max_features': ['sqrt', 'log2',None],
    #         'max_depth': [i for i in range(1, 5)]
    #     }
    #     clf = GridSearchCV(self, param_grid, cv=10)
    #     clf.fit(X, y)
    #     print(f"Best parameters: {clf.best_params_}")
    #     print(f"Best score: {clf.best_score_}")
    #     return clf.best_estimator_



import pandas as pd#pandasのインポート
import numpy as np
import datetime#元データの日付処理のためにインポート
from sklearn.model_selection import train_test_split#データ分割用
from sklearn.ensemble import RandomForestClassifier as RFC#ランダムフォレスト
#from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree,metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import glob
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm

def Data(array):
    X = array[:,0:20] #季節変化をふくめるなら36
    Y = array[:,20]
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1234)
    return X_train, X_test, y_train, y_test

def get_geo(file_10m):
    df_geo = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[2,3])
    df_geo.columns = ["xcoord","ycoord"]
    
    return df_geo

def get_df_for_RF(file_1,file_2,surface_type,year):
   
        df_rgb = pd.read_csv(file_1,encoding="shift-jis",engine='python',usecols=[2,3,4,5])
        df_rgb.columns=["Blue","Green","Red","NIR"]
        
        if year >= 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        df_oth = pd.read_csv(file_2,encoding="shift-jis",engine='python',usecols=[2,3,4,5,6,7])
        df_oth.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
                    "B12(2190)"]
        if year >= 2022:
            df_oth = (df_oth - 1000)/10000 #2022年用
        else:
            df_oth = df_oth/10000
        
        
        if surface_type == "FV":
            c = 1
        elif surface_type == "SV":
            c = 2
        elif surface_type == "OW_C":
            c = 3
        elif surface_type == "OW_S_Sa":
            c = 4
        elif surface_type == "OW_S_Ba":
            c = 5
        else:
            c = 6
        
        df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
        
        df_sf = pd.DataFrame()
        df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
        df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
        df_sf["ave_123"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
        df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(SWIR)"])/(df_rgb["Green"] + df_oth["B11(SWIR)"])
        df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(SWIR)"])/(df_rgb["NIR"] + df_oth["B11(SWIR)"])
        df_sf["NDCI"] = (df_oth["B5(705nm)"] - df_rgb["Red"])/(df_rgb["Red"] + df_oth["B5(705nm)"])
        df_sf["TCT_Wetness"] = 0.2578*df_rgb["Blue"] + 0.2305*df_rgb["Green"] + 0.0883*df_rgb["Red"] + 0.1071*df_rgb["NIR"] + (-0.7611)*df_oth["B11(SWIR)"] + (-0.5308)*df_oth["B12(2190)"]
        df_sf["FAI"] = df_rgb["NIR"] - (df_rgb["Red"] + (df_oth["B11(SWIR)"] - df_rgb["Red"]) * ((842 - 665) / (1610 - 665)))
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        data = pd.DataFrame(index=range(len(df)), columns=['ans'])
        data.fillna(c, inplace=True)
        
        df = pd.concat([df,data],axis = 1)
        df = df.dropna()
        
        return df
    
def get_df_for_CFT(file_10m,file_20m,year):
   
      #このように表記する事で1つのforで同時に複数のリストを回せる
            #os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_hisi')
        #file_1 = os.path.split(i)[1]
        df_rgb = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[4,5,6,7])
        df_rgb.columns = ["Blue","Green","Red","NIR"]      
        
        if year >= 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        #file_2 = os.path.split(j)[1]
        df_oth = pd.read_csv(file_20m,encoding="shift-jis",engine='python',usecols=[4,5,6,7,8,9])
        df_oth.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
                    "B12(2190)"]
        if year >= 2022:
            df_oth = (df_oth - 1000)/10000 
        else:
            df_oth = df_oth/10000
        
        df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
        
        df_sf = pd.DataFrame()
        df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
        df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
        df_sf["ave_123"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
        df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(SWIR)"])/(df_rgb["Green"] + df_oth["B11(SWIR)"])
        df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(SWIR)"])/(df_rgb["NIR"] + df_oth["B11(SWIR)"])
        df_sf["NDCI"] = (df_oth["B5(705nm)"] - df_rgb["Red"])/(df_rgb["Red"] + df_oth["B5(705nm)"])
        df_sf["TCT_Wetness"] = 0.2578*df_rgb["Blue"] + 0.2305*df_rgb["Green"] + 0.0883*df_rgb["Red"] + 0.1071*df_rgb["NIR"] + (-0.7611)*df_oth["B11(SWIR)"] + (-0.5308)*df_oth["B12(2190)"]
        df_sf["FAI"] = df_rgb["NIR"] - (df_rgb["Red"] + (df_oth["B11(SWIR)"] - df_rgb["Red"]) * ((842 - 665) / (1610 - 665)))
        # (("2021_10_24_10m_suwako@4"/10000) - (("2021_10_24_10m_suwako@1"/10000) + (("2021_10_24_suwako_resampled@5"/10000) - ("2021_10_24_10m_suwako@1"/10000))  *  ((842 - 665) / (1610 - 665))))
        
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        
        return df

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\Suwako_all")
df_2021_06_01_all = get_df_for_CFT("2021_06_01_all_10m.csv","2021_06_01_all_20m.csv",2021)
X = df_2021_06_01_all.data

print(X)

# X_feature = np.hsplit(X,"CLD")

# print(X_feature)

class CustomRFC(RFC):
    
    def predict(self, X, feature_name, threshold):
        
        """
        ランダムフォレストによるクラス分類を行う。
        ただし、指定された特徴量が指定されたしきい値よりも大きい場合、
        マスクをかけて親クラスであるRandomForestClassifierの解析対象としない。
        """
        
        X_feature = np.hsplit(X,"CLD")
        
        mask = X[:, feature_name] <= threshold
        masked_X = X[mask]
        
        # マスクをかけたデータを分類
        masked_y = super().predict(masked_X)
        
        # マスクをかけなかったデータを独立したCLDクラスとして分類
        independent_y = [1 if x else 0 for x in mask]
        
        # マスクをかけたデータと独立したデータを結合して返す
        return np.concatenate([masked_y, independent_y])
        
# def RF(X_train, X_test, y_train, y_test):

#     parameters = {  
#         #'n_estimators': [i for i in range(1, 501)],     # 用意する決定木モデルの数
#         'max_features': ('sqrt', 'log2', None),  # ランダムに指定する特徴量の数
#         #"criterion":    ["gini","entropy"]
#         'max_depth':    [i for i in range(1, 5)]# 決定木のノード深さの制限値
#     }
    
#     # モデルインスタンス
#     model = CustomRFC()
    
#     # ハイパーパラメータチューニング(グリッドサーチのコンストラクタにモデルと辞書パラメータを指定)
#     gridsearch = GridSearchCV(estimator = model,        # モデル
#                               param_grid = parameters,  # チューニングするハイパーパラメータ
#                               scoring = "accuracy",      # スコアリング
#                               cv = 10,
#                               n_jobs = -1
                              
#                              )
    
    
    
#     # 演算実行
#     gridsearch.fit(X_train, y_train)
    
#     # グリッドサーチの結果から得られた最適なパラメータ候補を確認
#     print('Best params: {}'.format(gridsearch.best_params_)) 
#     print('Best Score: {}'.format(gridsearch.best_score_))
    
#     model = CustomRFC(n_estimators = 300, # 用意する決定木モデルの数
#                                max_features = gridsearch.best_params_['max_features'], # ランダムに指定する特徴量の数
#                                max_depth    = gridsearch.best_params_['max_depth'],    # 決定木のノード深さの制限値
#                                #criterion=gridsearch.best_params_['criterion'],         # 不純度評価指標の種類(ジニ係数）
#                                min_samples_leaf = 1,                                   # 1ノードの深さの最小値
#                                random_state = 0,                                       # 乱数シード
#                                )

# # モデル学習
#     model.fit(X_train,y_train)
#     predicted = model.predict(X_test)
#     print(metrics.accuracy_score(predicted,y_test))
    
#     return model

# os.chdir("C:\\Users\\kohei\\Desktop\\Analysis_Paper\\FV")
# df_2021_07_21_FV = get_df_for_RF("Train_2021_07_21_FV_1_10m.csv","Train_2021_07_21_FV_1_20m.csv","FV",2021)
# df_2021_08_05_FV = get_df_for_RF("Train_2021_08_05_FV_1_10m.csv","Train_2021_08_05_FV_1_20m.csv","FV",2021)
# df_2021_09_27_FV = get_df_for_RF("Train_2021_09_27_FV_1_10m.csv","Train_2021_09_27_FV_1_20m.csv","FV",2021)
# df_FV = pd.concat([df_2021_07_21_FV,df_2021_08_05_FV,df_2021_09_27_FV])
# df_FV = df_FV.sample(frac=0.674)
# # print(df_FV)

# os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\SV")
# # df_2021_07_21_SV = get_df_for_RF("Train_2021_07_21_SV_1_10m.csv","Train_2021_07_21_SV_1_20m.csv","SV",2021)
# df_2021_08_05_SV = get_df_for_RF("Train_2021_08_05_SV_1_10m.csv","Train_2021_08_05_SV_1_20m.csv","SV",2021)
# df_2021_09_27_SV = get_df_for_RF("Train_2021_09_27_SV_1_10m.csv","Train_2021_09_27_SV_1_20m.csv","SV",2021)
# df_SV = pd.concat([df_2021_08_05_SV,df_2021_09_27_SV])
# # print(df_SV)

# os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_C")
# df_2021_07_21_OW_C = get_df_for_RF("Train_2021_07_21_W_C_1_10m.csv","Train_2021_07_21_W_C_1_20m.csv","OW_C",2021)
# df_2021_08_05_OW_C = get_df_for_RF("Train_2021_08_05_W_C_1_10m.csv","Train_2021_08_05_W_C_1_20m.csv","OW_C",2021)
# df_2021_09_27_OW_C = get_df_for_RF("Train_2021_09_27_W_C_1_10m.csv","Train_2021_09_27_W_C_1_20m.csv","OW_C",2021)
# df_OW_C = pd.concat([df_2021_07_21_OW_C,df_2021_08_05_OW_C,df_2021_09_27_OW_C])
# df_OW_C = df_OW_C.sample(frac=0.0898)
# # print(df_OW_C)

# os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_S_Ba")
# df_2021_07_21_OW_S_Ba = get_df_for_RF("Train_2021_07_21_W_Ba_1_10m.csv","Train_2021_07_21_W_Ba_1_20m.csv","OW_S_Ba",2021)
# df_2021_08_05_OW_S_Ba = get_df_for_RF("Train_2021_08_05_W_Ba_1_10m.csv","Train_2021_08_05_W_Ba_1_20m.csv","OW_S_Ba",2021)
# df_2021_09_27_OW_S_Ba = get_df_for_RF("Train_2021_09_27_W_Ba_1_10m.csv","Train_2021_09_27_W_Ba_1_20m.csv","OW_S_Ba",2021)
# df_OW_S_Ba = pd.concat([df_2021_07_21_OW_S_Ba,df_2021_08_05_OW_S_Ba,df_2021_09_27_OW_S_Ba])
# df_OW_S_Ba = df_OW_S_Ba.sample(frac=0.0521)
# # print(df_OW_S_Ba)

# os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_S_Sa")
# df_2021_07_21_OW_S_Sa = get_df_for_RF("Train_2021_07_21_W_Sa_1_10m.csv","Train_2021_07_21_W_Sa_1_20m.csv","OW_S_Sa",2021)
# df_2021_08_05_OW_S_Sa = get_df_for_RF("Train_2021_08_05_W_Sa_1_10m.csv","Train_2021_08_05_W_Sa_1_20m.csv","OW_S_Sa",2021)
# df_2021_09_27_OW_S_Sa = get_df_for_RF("Train_2021_09_27_W_Sa_1_10m.csv","Train_2021_09_27_W_Sa_1_20m.csv","OW_S_Sa",2021)
# df_OW_S_Sa = pd.concat([df_2021_07_21_OW_S_Sa,df_2021_08_05_OW_S_Sa,df_2021_09_27_OW_S_Sa])
# df_OW_S_Sa = df_OW_S_Sa.sample(frac=0.0575)
# # print(df_OW_S_Sa)

# # os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\St")
# # df_2021_07_21_St = get_df_for_RF("Train_2021_07_21_St_1_10m.csv","Train_2021_07_21_St_1_20m.csv","St",2022)

# df_2021 = pd.concat([df_FV,df_SV,df_OW_C,
#                             df_OW_S_Ba,df_OW_S_Sa],axis = 0)

# df_2021 = df_2021.dropna()


# # ##ここからRF
# X_train, X_test, y_train, y_test = Data(df_2021.values)


