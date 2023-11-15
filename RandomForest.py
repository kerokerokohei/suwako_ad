# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:32:49 2022

@author: guest1
"""

import pandas as pd#pandasのインポート
import numpy as np
import datetime#元データの日付処理のためにインポート
from sklearn.model_selection import train_test_split#データ分割用
from sklearn.ensemble import RandomForestClassifier as RFC#ランダムフォレスト
#from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import glob
import os
from sklearn.metrics import accuracy_score

os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_hisi\\2022_08_hisi')
file_hisi = glob.glob('*hisi.csv')#globは任意の文字列を含むファイル名を全て取得する．
file_hisi_oth = glob.glob("*other.csv")

os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo\\2022_08_kuromo')
file_kuromo = glob.glob('*kuromo.csv')#globは任意の文字列を含むファイル名を全て取得する．
file_kuromo_oth = glob.glob("*other.csv")

os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_water\\2022_08_water')
file_water = glob.glob('*water.csv')#globは任意の文字列を含むファイル名を全て取得する．
file_water_oth = glob.glob("*other.csv")

# def get_average(*df):
#     return (df)/(df)

df_sub = pd.DataFrame()
#print(df_sub)
df_sf_hisi=pd.DataFrame()
df_sf_kuromo=pd.DataFrame()
df_sf_water=pd.DataFrame()
df_pheno = pd.DataFrame()
df_phenology = pd.DataFrame()
df_rgb_hisi_pheno = pd.DataFrame()
df_rgb_oth_kuromo_pheno = pd.DataFrame()
df_rgb_oth_water_pheno = pd.DataFrame()
df_pca = pd.DataFrame()
df_pca_pheno = pd.DataFrame()
for (i,j,k,l,m,n) in zip(file_hisi,file_hisi_oth,file_kuromo,file_kuromo_oth,file_water,file_water_oth):    #このように表記する事で1つのforで同時に複数のリストを回せる
    os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_hisi')
    file_hisi = os.path.split(i)[1]
    df_rgb_hisi = pd.read_csv(file_hisi,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
    df_rgb_hisi.columns=["Blue","Green","Red","NIR"]
    df_rgb_hisi = (df_rgb_hisi - 1000)/10000 #2022年用
    df_rgb_hisi_pheno[file_hisi] = df_rgb_hisi.mean()
    
    # file_hisi_oth = os.path.split(j)[1]
    # df_oth_hisi = pd.read_csv(file_hisi_oth,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
    # df_oth_hisi.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
    #             "B12(2190)"]
    # df_oth_hisi = (df_oth_hisi - 1000)/10000 #2022年用
    
    # df_sub_hisi = pd.DataFrame(pd.date_range(start = "2022/1/1",freq = "D",periods = 1650))#hisi=1650 kuromo=157 water=1414
    # df_resample = pd.concat([df_sub_hisi,df_oth_hisi],axis = 1)
    # df_resample.set_index(0,inplace=True)
    # df_resample = df_resample.resample('6H').ffill()
    # df_resample = df_resample.reset_index(drop=True)
     
    # df_rgb_hisi = df_rgb_hisi.drop(range(6597,6615))#hisi
    # #df_rgb = df_rgb.drop(range(625,628))#kuromo
    # #df_rgb = df_rgb.drop(range(5653,5678))#water
    
    # df_rgb_oth_hisi = pd.concat([df_rgb_hisi,df_resample],axis = 1)
    # #df_rgb_oth_hisi_pheno[file_hisi] = df_rgb_oth_hisi.mean()
    
    # df_sf_hisi["sf1_ave"] = (df_rgb_hisi["NIR"] - ((df_rgb_hisi["Blue"] + df_rgb_hisi["Green"] + df_rgb_hisi["Red"])/3))/(df_rgb_hisi["NIR"] + ((df_rgb_hisi["Blue"] + df_rgb_hisi["Green"] + df_rgb_hisi["Red"])/3))
    # df_sf_hisi["WAVI"] = (1.5*(df_rgb_hisi["NIR"] - df_rgb_hisi["Blue"]))/(0.5 + df_rgb_hisi["NIR"] + df_rgb_hisi["Blue"])
    # df_sf_hisi["NDAVI"] = (df_rgb_hisi["NIR"] - df_rgb_hisi["Blue"])/(df_rgb_hisi["NIR"] + df_rgb_hisi["Blue"])
    # df_sf_hisi["NDVI"] = (df_rgb_hisi["NIR"] - df_rgb_hisi["Red"])/(df_rgb_hisi["NIR"] + df_rgb_hisi["Red"])
    # df_sf_hisi["ave_123"] = ((df_rgb_hisi["Blue"] + df_rgb_hisi["Green"] + df_rgb_hisi["Red"])/3)
    # df_sf_hisi["MNDWI"] = (df_rgb_hisi["Green"] - df_resample["B11(SWIR)"])/(df_rgb_hisi["Green"] + df_resample["B11(SWIR)"])
    # df_sf_hisi["NDWI"] = (df_rgb_hisi["NIR"] - df_resample["B11(SWIR)"])/(df_rgb_hisi["NIR"] + df_resample["B11(SWIR)"])
    # df_sf_hisi["SR"] = df_rgb_hisi["Red"]/df_rgb_hisi["NIR"]
    # df_sf_hisi["SRWC"] = df_rgb_hisi["Red"]/df_rgb_hisi["Blue"]
    
    # df_hisi = pd.concat([df_rgb_oth_hisi,df_sf_hisi],axis = 1)
    # data = pd.DataFrame(index=range(len(df_hisi)), columns=['ans'])
    # data.fillna(1, inplace=True)
    # df_hisi = pd.concat([df_hisi,data],axis = 1)
    
    
    # os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo')
    # file_kuromo = os.path.split(k)[1]
    # df_rgb_kuromo = pd.read_csv(file_kuromo,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
    # df_rgb_kuromo.columns=["Blue","Green","Red","NIR"]
    # df_rgb_kuromo = (df_rgb_kuromo - 1000)/10000 #2022年用
    
    
    # file_kuromo_oth = os.path.split(l)[1]
    # df_oth_kuromo = pd.read_csv(file_kuromo_oth,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
    # df_oth_kuromo.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
    #             "B12(2190)"]
    # df_oth_kuromo = (df_oth_kuromo - 1000)/10000 #2022年用
    
    # df_sub_kuromo = pd.DataFrame(pd.date_range(start = "2022/1/1",freq = "D",periods = 157))#hisi=1650 kuromo=157 water=1414
    # df_resample = pd.concat([df_sub_kuromo,df_oth_kuromo],axis = 1)
    # df_resample.set_index(0,inplace=True)
    # df_resample = df_resample.resample('6H').ffill()
    # df_resample = df_resample.reset_index(drop=True)
     
    # #df_rgb_hisi = df_rgb_hisi.drop(range(6597,6615))#hisi
    # df_rgb_kuromo = df_rgb_kuromo.drop(range(625,628))#kuromo
    # #df_rgb = df_rgb.drop(range(5653,5678))#water
    
    # df_rgb_oth_kuromo = pd.concat([df_rgb_kuromo,df_resample],axis = 1)
    # #df_rgb_oth_kuromo_pheno[file_kuromo] = df_rgb_oth_kuromo.mean()
    
    # df_sf_kuromo["sf1_ave"] = (df_rgb_kuromo["NIR"] - ((df_rgb_kuromo["Blue"] + df_rgb_kuromo["Green"] + df_rgb_kuromo["Red"])/3))/(df_rgb_kuromo["NIR"] + ((df_rgb_kuromo["Blue"] + df_rgb_kuromo["Green"] + df_rgb_kuromo["Red"])/3))
    # df_sf_kuromo["WAVI"] = (1.5*(df_rgb_kuromo["NIR"] - df_rgb_kuromo["Blue"]))/(0.5 + df_rgb_kuromo["NIR"] + df_rgb_kuromo["Blue"])
    # df_sf_kuromo["NDAVI"] = (df_rgb_kuromo["NIR"] - df_rgb_kuromo["Blue"])/(df_rgb_kuromo["NIR"] + df_rgb_kuromo["Blue"])
    # df_sf_kuromo["NDVI"] = (df_rgb_kuromo["NIR"] - df_rgb_kuromo["Red"])/(df_rgb_kuromo["NIR"] + df_rgb_kuromo["Red"])
    # df_sf_kuromo["ave_123"] = ((df_rgb_kuromo["Blue"] + df_rgb_kuromo["Green"] + df_rgb_kuromo["Red"])/3)
    # df_sf_kuromo["MNDWI"] = (df_rgb_kuromo["Green"] - df_resample["B11(SWIR)"])/(df_rgb_kuromo["Green"] + df_resample["B11(SWIR)"])
    # df_sf_kuromo["NDWI"] = (df_rgb_kuromo["NIR"] - df_resample["B11(SWIR)"])/(df_rgb_kuromo["NIR"] + df_resample["B11(SWIR)"])
    # df_sf_kuromo["SR"] = df_rgb_kuromo["Red"]/df_rgb_kuromo["NIR"]
    # df_sf_kuromo["SRWC"] = df_rgb_kuromo["Red"]/df_rgb_kuromo["Blue"]
    
    # df_kuromo = pd.concat([df_rgb_oth_kuromo,df_sf_kuromo],axis = 1)
    # data = pd.DataFrame(index=range(len(df_kuromo)), columns=['ans'])
    # data.fillna(2, inplace=True)
    # df_kuromo = pd.concat([df_kuromo,data],axis = 1)
    
    # os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_water')
    # file_water = os.path.split(m)[1]
    # df_rgb_water = pd.read_csv(file_water,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
    # df_rgb_water.columns=["Blue","Green","Red","NIR"]
    # df_rgb_water = (df_rgb_water - 1000)/10000 #2022年用
    
    
    # file_water_oth = os.path.split(n)[1]
    # df_oth_water = pd.read_csv(file_water_oth,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
    # df_oth_water.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
    #             "B12(2190)"]
    # df_oth_water = (df_oth_water - 1000)/10000 #2022年用
    
    # df_sub_water = pd.DataFrame(pd.date_range(start = "2022/1/1",freq = "D",periods = 1414))#hisi=1650 kuromo=157 water=1414
    # df_resample = pd.concat([df_sub_water,df_oth_water],axis = 1)
    # df_resample.set_index(0,inplace=True)
    # df_resample = df_resample.resample('6H').ffill()
    # df_resample = df_resample.reset_index(drop=True)
     
    # #df_rgb_hisi = df_rgb_hisi.drop(range(6597,6615))#hisi
    # #df_rgb_kuromo = df_rgb_kuromo.drop(range(625,628))#kuromo
    # df_rgb_water = df_rgb_water.drop(range(5653,5678))#water
    
    # df_rgb_oth_water = pd.concat([df_rgb_water,df_resample],axis = 1)
    
    # df_sf_water["sf1_ave"] = (df_rgb_water["NIR"] - ((df_rgb_water["Blue"] + df_rgb_water["Green"] + df_rgb_water["Red"])/3))/(df_rgb_water["NIR"] + ((df_rgb_water["Blue"] + df_rgb_water["Green"] + df_rgb_water["Red"])/3))
    # df_sf_water["WAVI"] = (1.5*(df_rgb_water["NIR"] - df_rgb_water["Blue"]))/(0.5 + df_rgb_water["NIR"] + df_rgb_water["Blue"])
    # df_sf_water["NDAVI"] = (df_rgb_water["NIR"] - df_rgb_water["Blue"])/(df_rgb_water["NIR"] + df_rgb_water["Blue"])
    # df_sf_water["NDVI"] = (df_rgb_water["NIR"] - df_rgb_water["Red"])/(df_rgb_water["NIR"] + df_rgb_water["Red"])
    # df_sf_water["ave_123"] = ((df_rgb_water["Blue"] + df_rgb_water["Green"] + df_rgb_water["Red"])/3)
    # df_sf_water["MNDWI"] = (df_rgb_water["Green"] - df_resample["B11(SWIR)"])/(df_rgb_water["Green"] + df_resample["B11(SWIR)"])
    # df_sf_water["NDWI"] = (df_rgb_water["NIR"] - df_resample["B11(SWIR)"])/(df_rgb_water["NIR"] + df_resample["B11(SWIR)"])
    # df_sf_water["SR"] = df_rgb_water["Red"]/df_rgb_water["NIR"]
    # df_sf_water["SRWC"] = df_rgb_water["Red"]/df_rgb_water["Blue"]
    
    # df_water = pd.concat([df_rgb_oth_water,df_sf_water],axis = 1)
    # data = pd.DataFrame(index=range(len(df_water)), columns=['ans'])
    # data.fillna(0, inplace=True)
    # df_water = pd.concat([df_water,data],axis = 1)
    
    
    # df_data = pd.concat([df_water,df_hisi,df_kuromo],axis = 0)
    # df_data = df_data.dropna()
    
#print(df_data)    

#df_rgb_hisi.to_csv("C:\\Users\\kohei\\Desktop\\test.csv")

FEATURES = ["Blue","Green","Red","NIR","B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
           "B12(2190)",'sf1_ave','WAVI','NDAVI','NDVI','ave_123','MNDWI','NDWI','SR','SRWC']
# df_train_data = pd.read_csv("RF_data.csv")
# array = df_data.values

def rforesti(array):
    X_train = array[:,0:18]
    y_train = array[:,19]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3,random_state=1234)

    rf = RFC(n_estimators=10, max_depth=3, bootstrap=True, random_state=1234)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)

    accu = accuracy_score(y_test, y_pred)
    print('accuracy = {:>.4f}'.format(accu))

    # Feature Importance
    fti = rf.feature_importances_   

    plt.figure(figsize = (10,6))
    plt.barh(y = range(len(fti)), width = fti)
    plt.yticks(ticks = range(len(FEATURES)), labels = FEATURES)
    plt.show()


#rforesti(array)


# print('Feature Importances:')
# for i, feat in enumerate(FEATURES):
#     print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
    
#np.savetxt("result.csv", test, delimiter=",")
#print("score=", rf.score(X_test, y_test))

"""ハイパーパラメータのチューニング
search_params = {
     'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
      'max_features'      : [3, 5],
      'random_state'      : [1234],
      'n_jobs'            : [1],
      'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
      'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
}


gs = GridSearchCV(RFC(),           # 対象の機械学習モデル
                  search_params,   # 探索パラメタ辞書
                  cv=3,            # クロスバリデーションの分割数
                  verbose=True,    # ログ表示
                  n_jobs=-1)       # 並列処理
gs.fit(X_train, y_train)
 
print(gs.best_estimator_)
"""

"""
CLASS_NAMES = ['Water', 'Hisi' , 'Kuromo']#ランダムフォレストのうちの任意の1つの決定木を描画
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(rf.estimators_[0], feature_names=FEATURES, class_names=CLASS_NAMES, filled=True, fontsize=10)
plt.show()
"""










