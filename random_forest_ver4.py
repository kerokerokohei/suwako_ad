# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:56:49 2023

@author: kohei
"""

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


FEATURES = ["Blue","Green","Red","NIR","B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
           "B12(2190)",'sf1_ave','WAVI','NDAVI','NDVI','ave_123','MNDWI','NDMI','NDCI',"TCT_wetness","FAI"]
# ,'NDCI'

FEATURES_DIFF = ["Bluediff","Greendiff","Reddiff","NIRdiff","B5(705nm)diff","B6(740nm)diff","B7(783nm)diff","B8A(865nm)diff","B11(SWIR)diff",
           "B12(2190)diff",'sf1_avediff','WAVIdiff','NDAVIdiff','NDVIdiff','ave_123diff','MNDWIdiff','NDMIdiff','NDCIdiff']
# ,'NDCIdiff'

FEATURES_ALL = ["Blue","Green","Red","NIR","B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
           "B12(2190)",'sf1_ave','WAVI','NDAVI','NDVI','ave_123','MNDWI','NDMI','NDCI',"Bluediff","Greendiff","Reddiff","NIRdiff","B5(705nm)diff","B6(740nm)diff","B7(783nm)diff","B8A(865nm)diff","B11(SWIR)diff",
                      "B12(2190)diff",'sf1_avediff','WAVIdiff','NDAVIdiff','NDVIdiff','ave_123diff','MNDWIdiff','NDMIdiff','NDCIdiff']

def rforest_important(X_train, X_test, y_train, y_test):
    rf = RFC(n_estimators=300, max_depth=4, bootstrap=True, random_state=1234)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)

    accu = accuracy_score(y_test, y_pred)
    print('accuracy = {:>.4f}'.format(accu))
    
    # Feature Importance
    fti = rf.feature_importances_ 
    print(fti)

    plt.figure(figsize = (10,6))
    plt.barh(y = range(len(fti)), width = fti)
    plt.yticks(ticks = range(len(FEATURES_ALL)), labels = FEATURES_ALL)
    plt.show()

def Data(array):
    X = array[:,0:20] #季節変化をふくめるなら36
    Y = array[:,20]
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1234)
    return X_train, X_test, y_train, y_test

def RF(X_train, X_test, y_train, y_test):

    parameters = {  
        #'n_estimators': [i for i in range(1, 501)],     # 用意する決定木モデルの数
        'max_features': ('sqrt', 'log2', None),  # ランダムに指定する特徴量の数
        #"criterion":    ["gini","entropy"]
        'max_depth':    [i for i in range(1, 5)]# 決定木のノード深さの制限値
    }
    
    # モデルインスタンス
    model = RFC()
    
    # ハイパーパラメータチューニング(グリッドサーチのコンストラクタにモデルと辞書パラメータを指定)
    gridsearch = GridSearchCV(estimator = model,        # モデル
                              param_grid = parameters,  # チューニングするハイパーパラメータ
                              scoring = "accuracy",      # スコアリング
                              cv = 10,
                              n_jobs = -1
                              
                             )
    
    
    
    # 演算実行
    gridsearch.fit(X_train, y_train)
    
    # グリッドサーチの結果から得られた最適なパラメータ候補を確認
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))
    
    model = RFC(n_estimators = 300, # 用意する決定木モデルの数
                               max_features = gridsearch.best_params_['max_features'], # ランダムに指定する特徴量の数
                               max_depth    = gridsearch.best_params_['max_depth'],    # 決定木のノード深さの制限値
                               #criterion=gridsearch.best_params_['criterion'],         # 不純度評価指標の種類(ジニ係数）
                               min_samples_leaf = 1,                                   # 1ノードの深さの最小値
                               random_state = 0,                                       # 乱数シード
                               )

# モデル学習
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print(metrics.accuracy_score(predicted,y_test))
    
    return model

def get_df_for_RF(file_1,file_2,surface_type,year):
   
        df_rgb = pd.read_csv(file_1,encoding="shift-jis",engine='python',usecols=[2,3,4,5])
        df_rgb.columns=["Blue","Green","Red","NIR"]
        
        if year >= 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        
        #file_2 = os.path.split(j)[1]
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
        df_rgb = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[3,4,5,6])
        df_rgb.columns = ["Blue","Green","Red","NIR"]      
        
        if year >= 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        #file_2 = os.path.split(j)[1]
        df_oth = pd.read_csv(file_20m,encoding="shift-jis",engine='python',usecols=[3,4,5,6,7,8])
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

def get_geo(file_10m):
    df_geo = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[1,2])
    df_geo.columns = ["xcoord","ycoord"]
    
    return df_geo


os.chdir("C:\\Users\\kohei\\Desktop\\Analysis_Paper\\FV")
df_2021_07_29_FV = get_df_for_RF("Train_2021_07_29_FV_1_10m.csv","Train_2021_07_29_FV_1_20m.csv","FV",2021)

print(df_2021_07_29_FV)

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\SV")
df_2021_07_29_SV = get_df_for_RF("Train_2021_07_29_SV_1_10m.csv","Train_2021_07_29_SV_1_20m.csv","SV",2021)

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_C")
df_2021_07_29_OW_C = get_df_for_RF("Train_2021_07_29_W_C_1_10m.csv","Train_2021_07_29_W_C_1_20m.csv","OW_C",2021)

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_S_Ba")
df_2021_07_29_OW_S_Ba = get_df_for_RF("Train_2021_07_29_Ba_1_10m.csv","Train_2021_07_29_Ba_1_20m.csv","OW_S_Ba",2021)

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\OW_S_Sa")
df_2021_07_29_OW_S_Sa = get_df_for_RF("Train_2021_07_29_W_Sa_1_10m.csv","Train_2021_07_29_W_Sa_1_20m.csv","OW_S_Sa",2021)

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\St")
df_2021_07_29_St = get_df_for_RF("Train_2021_07_29_St_1_10m.csv","Train_2021_07_29_St_1_20m.csv","St",2021)

# df_2021_08_ave = pd.concat([df_2021_08_hisi_ave,df_2021_08_kuromo_ave,df_2021_08_water_ave],axis = 0)

# # df_2021 = pd.concat([df_2021_08_ave_d,df_2021_08_06_dif_ave],axis = 1)
# df_2021 = df_2021_08_ave
# # df_2021 = df_2021.interpolate()
# df_2021 = df_2021.dropna()
# # df_2021.to_csv("C:\\Users\kohei\Desktop\phenology_data\\df_2021_08.csv")

# ##ここからRF
# X_train, X_test, y_train, y_test = Data(df_2021.values)

# # #print(y_train)
# # # rforest_important(X_train, X_test, y_train, y_test)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_all")
# # df_2021_06_all = get_df_for_CFT("2021_06_01_10m_all.csv","2021_06_01_20m_all.csv",2021)
# df_geo_2021 = get_geo("2021_08_05_10m_all.csv")

# df_2021_07_11_all = get_df_for_CFT("2021_07_11_10m_all.csv","2021_07_11_20m_all.csv",2021)
# df_2021_07_21_all = get_df_for_CFT("2021_07_21_10m_all.csv","2021_07_21_20m_all.csv",2021)
# df_2021_08_05_all = get_df_for_CFT("2021_08_05_10m_all.csv","2021_08_05_20m_all.csv",2021)
# df_2021_08_28_all = get_df_for_CFT("2021_08_28_10m_all.csv","2021_08_28_20m_all.csv",2021)
# df_2021_08_30_all = get_df_for_CFT("2021_08_30_10m_all.csv","2021_08_30_20m_all.csv",2021)
# df_2021_09_19_all = get_df_for_CFT("2021_09_19_10m_all.csv","2021_09_19_20m_all.csv",2021)
# df_2021_09_27_all = get_df_for_CFT("2021_09_27_10m_all.csv","2021_09_27_20m_all.csv",2021)


# # df_2021_08_ave = (df_2021_08_05_all + df_2021_08_30_all)/2
# df_2021_08_all_conc = df_2021_08_05_all #pd.concat([df_2021_08_05_all , df_2021_08_30_all],axis = 0)


# # # df_2021_diff = df_2021_08_ave - df_2021_06_all
# # # # df_2021_diff = df_2021_08_05_all - df_2021_06_all
# # # df_2021_diff = df_2021_diff.set_axis([FEATURES_DIFF],axis = 1)

# # # # df_2021_all = pd.concat([df_2021_08_ave,df_2021_diff],axis = 1)
# # # df_2021_all = pd.concat([df_2021_08_ave,df_2021_diff],axis = 1)

# df_2021_all = df_2021_08_all_conc
# df_2021_all = df_2021_all.interpolate()
# df_2021_all = df_2021_all.dropna()

# best_model = RF(X_train, X_test, y_train, y_test)

# result = best_model.predict(df_2021_all)

# df_result = pd.DataFrame(data = result)
# df_result = df_result[0].astype("int64")
# df_result = pd.concat([df_result,df_geo_2021],axis = 1)

# df_result.to_csv("C:\\Users\\kohei\\Desktop\\phenology_data\\result_2021_(8_conc)_from_2021_(8_ave)_QC.csv")
# # # print(df_result.dtypes)

