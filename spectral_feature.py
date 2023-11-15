# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:25:54 2022

@author: kohei
"""

import pandas as pd #pandasをインポート
import numpy as np #numpyをインポート
import datetime as dt
from datetime import time
import glob
import os
import sklearn
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
#from natsort import natsorted

os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2')
file1 = glob.glob('*water.csv')#globは任意の文字列を含むファイル名を全て取得する．
file = glob.glob("*other_ver2.csv")
#print(file1)
# print(file)
#file=natsorted(file)
df_sub = pd.DataFrame()
#print(df_sub)
df_sf=pd.DataFrame()
df_pheno = pd.DataFrame()
df_phenology = pd.DataFrame()
df_rgb_oth_pheno = pd.DataFrame()
for (i,j) in zip(file1,file):    #このように表記する事で1つのforで同時に複数のリストを回せる
    file1=os.path.split(i)[1]
    df_rgb=pd.read_csv(file1,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
    df_rgb.columns=["Blue","Green","Red","NIR"]
    df_rgb = (df_rgb - 1000)/10000 #2022年用
    # df_rgb = (df_rgb)/10000 #2021年用
    
    
    file = os.path.split(j)[1]
    df_oth = pd.read_csv(file,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
    df_oth.columns=["B5(705 nm)","B6(740 nm)","B7(783 nm)","B8A(865 nm)","B11(1610 nm)",
                "B12(2190 nm)"]
    df_oth = (df_oth - 1000)/10000 #2022年用
    # df_oth = (df_oth)/10000 #2021年用
    
    # df_sub = pd.DataFrame(pd.date_range(start = "2022/1/1",freq = "D",periods = 157))#hisi=1650 kuromo=157 water=1414
    # df_resample = pd.concat([df_sub,df_oth],axis = 1)
    # df_resample.set_index(0,inplace=True)
    # df_resample = df_resample.resample('6H').ffill()
    # df_resample = df_resample.reset_index(drop=True)
     
    # #df_rgb = df_rgb.drop(range(6597,6615))#hisi
    # df_rgb = df_rgb.drop(range(625,628))#kuromo
    # #df_rgb = df_rgb.drop(range(5653,5678))#water
    
    # df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
    # df_rgb_oth_pheno[file1] = df_rgb_oth.mean()
     
    df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
    df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
    df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
    df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
    df_sf["ave_RGB"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
    df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(1610 nm)"])/(df_rgb["Green"] + df_oth["B11(1610 nm)"])
    df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(1610 nm)"])/(df_rgb["NIR"] + df_oth["B11(1610 nm)"])
    df_sf["NDCI"] = (df_oth["B5(705 nm)"] - df_rgb["Red"])/(df_rgb["Red"] + df_oth["B5(705 nm)"])
    
    df_pheno[file1] = df_sf.mean()
    
    # df_pca = pd.concat([df_rgb,df_resample],axis = 1)
    # df_pca = df_pca.iloc[:, 0:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    # pca = PCA()
    # pca.fit(df_pca)
    # # データを主成分空間に写像
    # feature = pca.transform(df_pca)
    # df_feature = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(df_pca.columns))])
    
   
# plt.figure(figsize=(6, 6))
# plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df_pca.iloc[:, 0]))
# plt.grid()
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()    
# # 寄与率
# pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(df_pca.columns))])
# #print(pca.explained_variance_ratio_)
# # 累積寄与率を図示する
# import matplotlib.ticker as ticker
# plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
# plt.xlabel("Number of principal components")
# plt.ylabel("Cumulative contribution rate")
# plt.grid()
# plt.show()


df_pheno.to_csv('C:\\Users\kohei\Desktop\phenology_data/2022_pheno_ver2/2022_water_sf_pheno_ver2.csv')
# df_rgb_oth_pheno.to_csv('C:\\Users\kohei\Desktop\phenology_data/2021_pheno_ver2/2021_water_pheno_ver2.csv')

#print(df_feature)
#print(df_sf.mean())
#print(file1)
#print(df_pheno)
#print(df_rgb_oth_pheno) 
#print(df_resample)
#print(df_oth)