# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 23:52:48 2023

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
from sklearn.metrics import confusion_matrix
import seaborn as sns

FEATURES = ["Blue","Green","Red","NIR","B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
           "B12(2190)",'sf1_ave','WAVI','NDAVI','NDVI','ave_RGB','MNDWI','NDMI','NDCI']

FEATURES_DIFF = ["Blue_diff","Green_diff","Red_diff","NIR_diff","B5(705nm)_diff","B6(740nm)_diff","B7(783nm)_diff","B8A(865nm)_diff","B11(SWIR)_diff",
           "B12(2190)_diff",'sf1_ave_diff','WAVI_diff','NDAVI_diff','NDVI_diff','ave_RGB_diff','MNDWI_diff','NDMI_diff' ,'NDCIdiff']

FEATURES_ALL = FEATURES + FEATURES_DIFF

def get_df_for_CFT(file_10m,file_20m,year):
   
        df_rgb = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[3,4,5,6])
        df_rgb.columns = ["Blue","Green","Red","NIR"]      
        
        if year == 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        #file_2 = os.path.split(j)[1]
        df_oth = pd.read_csv(file_20m,encoding="shift-jis",engine='python',usecols=[3,4,5,6,7,8])
        df_oth.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
                    "B12(2190)"]
        if year == 2022:
            df_oth = (df_oth - 1000)/10000 
        else:
            df_oth = df_oth/10000
        
        df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
        
        df_sf = pd.DataFrame()
        df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
        df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
        df_sf["ave_RGB"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
        df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(SWIR)"])/(df_rgb["Green"] + df_oth["B11(SWIR)"])
        df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(SWIR)"])/(df_rgb["NIR"] + df_oth["B11(SWIR)"])
        df_sf["NDCI"] = (df_oth["B5(705nm)"] - df_rgb["Red"])/(df_rgb["Red"] + df_oth["B5(705nm)"])
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        
        return df

def get_geo(file_10m):
    df_geo = pd.read_csv(file_10m,encoding="shift-jis",engine='python',usecols=[1,2])
    df_geo.columns = ["xcoord","ycoord"]
    
    return df_geo


def Data(array):
    X = array[:,0:36]
    Y = array[:,36]
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1234)
    return X_train, X_test, y_train, y_test

def get_df_for_RF(file_1,file_2,surface_type,year):
   
      #このように表記する事で1つのforで同時に複数のリストを回せる
            #os.chdir('C:\\Users\kohei\Desktop\phenology_data\\2022_hisi')
        #file_1 = os.path.split(i)[1]
        df_rgb = pd.read_csv(file_1,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
        df_rgb.columns=["Blue","Green","Red","NIR"]
        
        if year == 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        
        #file_2 = os.path.split(j)[1]
        df_oth = pd.read_csv(file_2,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
        df_oth.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
                    "B12(2190)"]
        if year == 2022:
            df_oth = (df_oth - 1000)/10000 #2022年用
        else:
            df_oth = df_oth/10000
        
        
        if surface_type == "hisi":
            #q = range(6597,6615)
            c = 1
        elif surface_type == "kuromo":
            #q = range(625,628)
            c = 2
        else:
            #q = range(5653,5678)
            c = 0
        
        df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
        #df_rgb_oth_pheno[file] = df_rgb_oth.mean()
        
        df_sf = pd.DataFrame()
        df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
        df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
        df_sf["ave_RGB"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
        df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(SWIR)"])/(df_rgb["Green"] + df_oth["B11(SWIR)"])
        df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(SWIR)"])/(df_rgb["NIR"] + df_oth["B11(SWIR)"])
        df_sf["NDCI"] = (df_oth["B5(705nm)"] - df_rgb["Red"])/(df_rgb["Red"] + df_oth["B5(705nm)"])
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        data = pd.DataFrame(index=range(len(df)), columns=['ans'])
        data.fillna(c, inplace=True)
        
        df = pd.concat([df,data],axis = 1)
        df = df.dropna()
        
        return df

def get_ccp_alphas(X_train, y_train):
    path = DecisionTreeClassifier(max_depth=5).cost_complexity_pruning_path(X_train, y_train)
    display(pd.DataFrame(path))

    _, ax = plt.subplots(figsize=(10, 4))

    # 最終行は1ノードだけの決定木なので出力は無駄
    ax.plot(path.ccp_alphas[:-1], path.impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

    return path

#訓練
def train_with_alphas(X_train, y_train, ccp_alphas):
    clfs = []
    for i, ccp_alpha in enumerate(ccp_alphas): #enumerateでリストの要素と行数を獲得している．
        clf = DecisionTreeClassifier(max_depth=5, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        #print(f'Finished: {i+1}/{len(ccp_alphas)}')
    # ノード数は枝刈りを最後までやった結果なので必ず１
    #print(f"最終決定木のノード数: {clfs[-1].tree_.node_count} with ccp_alpha: {ccp_alphas[-1]}")
    return clfs

#枝刈り単位の情報参照
def output_prune_result(path, clfs, train_scores, test_scores):
    node_counts = [clf.tree_.node_count for clf in clfs] 
    depth = [clf.tree_.max_depth for clf in clfs]
    fig=plt.figure(figsize=(10,12))
    #print(len(node_counts),path.ccp_alphas.shape)
    
    df = pd.DataFrame(data = [node_counts,depth,train_scores,test_scores,path.ccp_alphas[:-1], path.impurities[:-1]],index = ["node_num","depth","train_scores","test_scores","ccp_alpha","impurerity"])
    # df.to_csv("C:\\Users\kohei\Desktop\phenology_data\param_2022.csv")
    # print(node_counts)
    # print(depth)
    # print(train_scores)
    # print(test_scores)
    # print(df)
    
    
    ax_11=fig.add_subplot(121)
    ax_11.axis('tight')
    ax_11.axis('off')
    tab = ax_11.table(cellText=np.round(pd.DataFrame(path).values, decimals=5),
                loc='upper left',
                colLabels=dir(path),
                rowLabels=np.arange(len(path.ccp_alphas)),
                colColours =["#EEEEEE"] * 2,
                rowColours =["#EEEEEE"] * len(path.ccp_alphas))
    tab.auto_set_font_size(False)
    tab.set_fontsize(15)
    tab.scale(1,2)

    ax_21=fig.add_subplot(422)
    ax_21.plot(path.ccp_alphas[:-1], path.impurities[:-1], marker="o", drawstyle="steps-post")
    ax_21.set_xlabel("effective alpha")
    ax_21.set_ylabel("total impurity of leaves")
    ax_21.set_title("Total Impurity vs effective alpha for training set")

    ax_22=fig.add_subplot(424)
    ax_22.plot(path.ccp_alphas[:-1], node_counts, marker="o", drawstyle="steps-post")
    ax_22.set_xlabel("alpha")
    ax_22.set_ylabel("number of nodes")
    ax_22.set_title("Number of nodes vs alpha")

    ax_23=fig.add_subplot(426)
    ax_23.plot(path.ccp_alphas[:-1], depth, marker="o", drawstyle="steps-post")
    ax_23.set_xlabel("alpha")
    ax_23.set_ylabel("depth of tree")
    ax_23.set_title("Depth vs alpha")

    ax_24=fig.add_subplot(428)
    ax_24.set_xlabel("alpha")
    ax_24.set_ylabel("accuracy")
    ax_24.set_title("Accuracy vs alpha for training and testing sets")
    ax_24.plot(path.ccp_alphas[:-1], train_scores, marker="o", label="train", drawstyle="steps-post")
    ax_24.plot(path.ccp_alphas[:-1], test_scores, marker="o", label="test", drawstyle="steps-post")
    ax_24.legend()    
    
    fig.tight_layout()
    plt.show()
    return df

#分類評価指標表示
def output_graphs(clf, X_test, y_test):
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # Confusion Matrix 出力
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0, 0])
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, square=True, cbar=True, annot=True, fmt = "d",cmap='Blues_r')
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    # plt.savefig('C:\\Users\kohei\Desktop\phenology_data\\sklearn_confusion_matrix_2022.png',dpi = 400)

    # Feature Importance 出力
    importances = pd.DataFrame({'Importance':clf.feature_importances_}, index=FEATURES_ALL)
    importances_2 = importances.sort_values('Importance', ascending=False).head(10).sort_values('Importance', ascending=True)
    importances.sort_values('Importance', ascending=False).head(10).sort_values('Importance', ascending=True).plot.barh(ax=axes[0, 1], grid=True)
    # importances_2.to_csv("C:\\Users\kohei\Desktop\phenology_data\\CT_importance_2021.csv")

    # ROC曲線出力
    # RocCurveDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 0])
    # axes[1, 0].set_title('ROC(Receiver Operating Characteristic) Curve')

    # 適合率-再現率グラフ出力
    # PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 1])

    plt.show()


#決定木描画
CLASS_NAMES = ['Water', 'Hisi' , 'Kuromo']
def output_trees(clf, X_train, y_train):
    plt.figure(figsize=(18,7))
    plot_tree(clf, filled=True, feature_names=FEATURES_ALL, class_names=CLASS_NAMES, fontsize=9)
    plt.show()
    

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_hisi_ver2")
df_2021_06_hisi_ave = get_df_for_RF("2021_06_01_hisi_ver2.csv","2021_06_01_hisi_other_ver2.csv","hisi",2021)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_kuromo_ver2")
df_2021_06_kuromo_ave = get_df_for_RF("2021_06_01_kuromo.csv","2021_06_01_kuromo_other_ver2.csv","kuromo",2021)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_water_ver2")
df_2021_06_water_ave = get_df_for_RF("2021_06_01_water.csv","2021_06_01_water_other_ver2.csv","water",2021)

df_2021_06_ave = pd.concat([df_2021_06_hisi_ave,df_2021_06_kuromo_ave,df_2021_06_water_ave],axis = 0)
df_2021_06_ave_d = df_2021_06_ave.loc[:,FEATURES]
df_2021_06_ave_d = df_2021_06_ave_d.reset_index()
df_2021_06_ave_d = df_2021_06_ave_d.drop("index",axis = 1)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_hisi_ver2")
df_2021_08_05_hisi_ave = get_df_for_RF("2021_08_05_hisi_ver2.csv","2021_08_05_hisi_other_ver2.csv","hisi",2021)
df_2021_08_28_hisi_ave = get_df_for_RF("2021_08_28_hisi_ver2.csv","2021_08_28_hisi_other_ver2.csv","hisi",2021)
df_2021_08_30_hisi_ave = get_df_for_RF("2021_08_30_hisi_ver2.csv","2021_08_30_hisi_other_ver2.csv","hisi",2021)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_kuromo_ver2")
df_2021_08_05_kuromo_ave = get_df_for_RF("2021_08_05_kuromo.csv","2021_08_05_kuromo_other_ver2.csv","kuromo",2021)
df_2021_08_28_kuromo_ave = get_df_for_RF("2021_08_28_kuromo.csv","2021_08_28_kuromo_other_ver2.csv","kuromo",2021)
df_2021_08_30_kuromo_ave = get_df_for_RF("2021_08_30_kuromo.csv","2021_08_30_kuromo_other_ver2.csv","kuromo",2021)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_water_ver2")
df_2021_08_05_water_ave = get_df_for_RF("2021_08_05_water.csv","2021_08_05_water_other_ver2.csv","water",2021)
df_2021_08_28_water_ave = get_df_for_RF("2021_08_28_water.csv","2021_08_28_water_other_ver2.csv","water",2021)
df_2021_08_30_water_ave = get_df_for_RF("2021_08_30_water.csv","2021_08_30_water_other_ver2.csv","water",2021)

df_2021_08_hisi_ave = (df_2021_08_05_hisi_ave + df_2021_08_28_hisi_ave + df_2021_08_30_hisi_ave)/3
df_2021_08_kuromo_ave = (df_2021_08_05_kuromo_ave + df_2021_08_28_kuromo_ave + df_2021_08_30_kuromo_ave)/3
df_2021_08_water_ave = (df_2021_08_05_water_ave + df_2021_08_28_water_ave + df_2021_08_30_water_ave)/3

df_2021_08_ave = pd.concat([df_2021_08_hisi_ave,df_2021_08_kuromo_ave,df_2021_08_water_ave],axis = 0)
# df_2021_08_ave = df_2021_08_ave.interpolate()
df_2021_08_ave = df_2021_08_ave.dropna()
#print(df_2021_08_ave)

df_2021_08_ave_ans = df_2021_08_ave.loc[:,"ans"]
df_2021_08_ave_ans = df_2021_08_ave_ans.reset_index()
df_2021_08_ave_ans = df_2021_08_ave_ans.drop("index",axis = 1)

df_2021_08_ave_d = df_2021_08_ave.loc[:,FEATURES]
df_2021_08_ave_d = df_2021_08_ave_d.reset_index()
df_2021_08_ave_d = df_2021_08_ave_d.drop("index",axis = 1)

df_2021_08_06_dif_ave_d = df_2021_08_ave_d - df_2021_06_ave_d
df_2021_08_06_dif_ave_d = df_2021_08_06_dif_ave_d.set_axis([FEATURES_DIFF],axis = 1)
# print(df_2021_08_06_dif_ave_d)
df_2021_08_06_dif_ave = pd.concat([df_2021_08_06_dif_ave_d,df_2021_08_ave_ans],axis = 1)
# print(df_2021_08_06_dif_ave)
df_2021_08_06_dif_ave = df_2021_08_06_dif_ave.dropna()

df_2021 = pd.concat([df_2021_08_ave_d,df_2021_08_06_dif_ave],axis = 1)
df_2021 = df_2021.dropna()
# df_2021 = df_2021_08_ave
#print(df_2021)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi_ver2")
# df_2022_08_hisi_ave = get_df_for_RF("2022_08_10_hisi_ver2.csv","2022_08_10_hisi_other_ver2.csv","hisi",2022)


# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo_ver2")
# df_2022_08_kuromo_ave = get_df_for_RF("2022_08_10_kuromo.csv","2022_08_10_kuromo_other_ver2.csv","kuromo",2022)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2")
# df_2022_08_water_ave = get_df_for_RF("2022_08_10_water.csv","2022_08_10_water_other_ver2.csv","water",2022)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi_ver2")
# df_2022_06_04_hisi_ave = get_df_for_RF("2022_06_04_hisi_ver2.csv","2022_06_04_hisi_other_ver2.csv","hisi",2022)
# df_2022_06_09_hisi_ave = get_df_for_RF("2022_06_09_hisi_ver2.csv","2022_06_09_hisi_other_ver2.csv","hisi",2022)
# df_2022_06_19_hisi_ave = get_df_for_RF("2022_06_19_hisi_ver2.csv","2022_06_19_hisi_other_ver2.csv","hisi",2022)
# df_2022_06_29_hisi_ave = get_df_for_RF("2022_06_29_hisi_ver2.csv","2022_06_29_hisi_other_ver2.csv","hisi",2022)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo_ver2")
# df_2022_06_04_kuromo_ave = get_df_for_RF("2022_06_04_kuromo.csv","2022_06_04_kuromo_other_ver2.csv","kuromo",2022)
# df_2022_06_09_kuromo_ave = get_df_for_RF("2022_06_09_kuromo.csv","2022_06_09_kuromo_other_ver2.csv","kuromo",2022)
# df_2022_06_19_kuromo_ave = get_df_for_RF("2022_06_19_kuromo.csv","2022_06_19_kuromo_other_ver2.csv","kuromo",2022)
# df_2022_06_29_kuromo_ave = get_df_for_RF("2022_06_29_kuromo.csv","2022_06_29_kuromo_other_ver2.csv","kuromo",2022)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2")
# df_2022_06_04_water_ave = get_df_for_RF("2022_06_04_water.csv","2022_06_04_water_other_ver2.csv","water",2022)
# df_2022_06_09_water_ave = get_df_for_RF("2022_06_09_water.csv","2022_06_09_water_other_ver2.csv","water",2022)
# df_2022_06_19_water_ave = get_df_for_RF("2022_06_19_water.csv","2022_06_19_water_other_ver2.csv","water",2022)
# df_2022_06_29_water_ave = get_df_for_RF("2022_06_29_water.csv","2022_06_29_water_other_ver2.csv","water",2022)

# df_2022_08_ave = pd.concat([df_2022_08_hisi_ave,df_2022_08_kuromo_ave,df_2022_08_water_ave],axis = 0)
# df_2022_08_ave_d = df_2022_08_ave.loc[:,FEATURES]
# df_2022_08_ave_d = df_2022_08_ave_d.reset_index()
# df_2022_08_ave_d = df_2022_08_ave_d.drop("index",axis = 1)

# df_2022_06_hisi_ave = (df_2022_06_04_hisi_ave + df_2022_06_09_hisi_ave + df_2022_06_19_hisi_ave + df_2022_06_29_hisi_ave)/4
# df_2022_06_kuromo_ave = (df_2022_06_04_kuromo_ave + df_2022_06_09_kuromo_ave + df_2022_06_19_kuromo_ave + df_2022_06_29_kuromo_ave)/4
# df_2022_06_water_ave = (df_2022_06_04_water_ave + df_2022_06_09_water_ave + df_2022_06_19_water_ave + df_2022_06_29_water_ave)/4

# df_2022_06_ave = pd.concat([df_2022_06_hisi_ave,df_2022_06_kuromo_ave,df_2022_06_water_ave])
# #print(df_2022_06_ave)

# df_2022_06_ave_ans = df_2022_06_ave.loc[:,"ans"]
# df_2022_06_ave_ans = df_2022_06_ave_ans.reset_index()
# df_2022_06_ave_ans = df_2022_06_ave_ans.drop("index",axis = 1)

# df_2022_06_ave_d = df_2022_06_ave.loc[:,FEATURES]
# df_2022_06_ave_d = df_2022_06_ave_d.reset_index()
# df_2022_06_ave_d = df_2022_06_ave_d.drop("index",axis = 1)

# df_2022_08_06_dif_ave_d = df_2022_08_ave_d - df_2022_06_ave_d
# df_2022_08_06_dif_ave_d = df_2022_08_06_dif_ave_d.set_axis([FEATURES_DIFF],axis = 1)
# df_2022_08_06_dif_ave = pd.concat([df_2022_08_06_dif_ave_d,df_2022_06_ave_ans],axis = 1)
# df_2022_08_06_dif_ave = df_2022_08_06_dif_ave.dropna()

# df_2022 = pd.concat([df_2022_08_ave_d,df_2022_08_06_dif_ave],axis = 1)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi_ver2")
df_2022_07_01_hisi_ave = get_df_for_RF("2022_07_01_hisi_ver2.csv","2022_07_01_hisi_other_ver2.csv","hisi",2022)
df_2022_07_29_hisi_ave = get_df_for_RF("2022_07_29_hisi_ver2.csv","2022_07_29_hisi_other_ver2.csv","hisi",2022)
df_2022_07_31_hisi_ave = get_df_for_RF("2022_07_31_hisi_ver2.csv","2022_07_31_hisi_other_ver2.csv","hisi",2022)
os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo_ver2")
df_2022_07_01_kuromo_ave = get_df_for_RF("2022_07_01_kuromo.csv","2022_07_01_kuromo_other_ver2.csv","kuromo",2022)
df_2022_07_29_kuromo_ave = get_df_for_RF("2022_07_29_kuromo.csv","2022_07_29_kuromo_other_ver2.csv","kuromo",2022)
df_2022_07_31_kuromo_ave = get_df_for_RF("2022_07_31_kuromo.csv","2022_07_31_kuromo_other_ver2.csv","kuromo",2022)
os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2")
df_2022_07_01_water_ave = get_df_for_RF("2022_07_01_water.csv","2022_07_01_water_other_ver2.csv","water",2022)
df_2022_07_29_water_ave = get_df_for_RF("2022_07_29_water.csv","2022_07_29_water_other_ver2.csv","water",2022)
df_2022_07_31_water_ave = get_df_for_RF("2022_07_31_water.csv","2022_07_31_water_other_ver2.csv","water",2022)

df_2022_07_hisi_ave = (df_2022_07_01_hisi_ave + df_2022_07_29_hisi_ave + df_2022_07_31_hisi_ave)/3
df_2022_07_kuromo_ave = (df_2022_07_01_kuromo_ave + df_2022_07_29_kuromo_ave + df_2022_07_31_kuromo_ave)/3
df_2022_07_water_ave = (df_2022_07_01_water_ave + df_2022_07_29_water_ave + df_2022_07_31_water_ave)/3


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi_ver2")
df_2022_04_12_hisi_ave = get_df_for_RF("2022_04_12_hisi_ver2.csv","2022_04_12_hisi_other_ver2.csv","hisi",2022)
df_2022_04_20_hisi_ave = get_df_for_RF("2022_04_20_hisi_ver2.csv","2022_04_20_hisi_other_ver2.csv","hisi",2022)
df_2022_04_22_hisi_ave = get_df_for_RF("2022_04_22_hisi_ver2.csv","2022_04_22_hisi_other_ver2.csv","hisi",2022)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo_ver2")
df_2022_04_12_kuromo_ave = get_df_for_RF("2022_04_12_kuromo.csv","2022_04_12_kuromo_other_ver2.csv","kuromo",2022)
df_2022_04_20_kuromo_ave = get_df_for_RF("2022_04_20_kuromo.csv","2022_04_20_kuromo_other_ver2.csv","kuromo",2022)
df_2022_04_22_kuromo_ave = get_df_for_RF("2022_04_22_kuromo.csv","2022_04_22_kuromo_other_ver2.csv","kuromo",2022)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2")
df_2022_04_12_water_ave = get_df_for_RF("2022_04_12_water.csv","2022_04_12_water_other_ver2.csv","water",2022)
df_2022_04_20_water_ave = get_df_for_RF("2022_04_20_water.csv","2022_04_20_water_other_ver2.csv","water",2022)
df_2022_04_22_water_ave = get_df_for_RF("2022_04_22_water.csv","2022_04_22_water_other_ver2.csv","water",2022)


df_2022_07_ave = pd.concat([df_2022_07_hisi_ave,df_2022_07_kuromo_ave,df_2022_07_water_ave],axis = 0)
df_2022_07_ave_d = df_2022_07_ave.loc[:,FEATURES]
df_2022_07_ave_d = df_2022_07_ave_d.reset_index()
df_2022_07_ave_d = df_2022_07_ave_d.drop("index",axis = 1)
df_2022_07_ave_d = df_2022_07_ave_d.dropna()

df_2022_04_hisi_ave = (df_2022_04_12_hisi_ave + df_2022_04_20_hisi_ave + df_2022_04_22_hisi_ave)/3
df_2022_04_kuromo_ave = (df_2022_04_12_kuromo_ave + df_2022_04_20_kuromo_ave + df_2022_04_22_kuromo_ave)/3
df_2022_04_water_ave = (df_2022_04_12_water_ave + df_2022_04_20_water_ave + df_2022_04_22_water_ave)/3

df_2022_04_ave = pd.concat([df_2022_04_hisi_ave,df_2022_04_kuromo_ave,df_2022_04_water_ave])
#print(df_2022_06_ave)

df_2022_04_ave_ans = df_2022_04_ave.loc[:,"ans"]
df_2022_04_ave_ans = df_2022_04_ave_ans.reset_index()
df_2022_04_ave_ans = df_2022_04_ave_ans.drop("index",axis = 1)

df_2022_04_ave_d = df_2022_04_ave.loc[:,FEATURES]
df_2022_04_ave_d = df_2022_04_ave_d.reset_index()
df_2022_04_ave_d = df_2022_04_ave_d.drop("index",axis = 1)

df_2022_07_04_dif_ave_d = df_2022_07_ave_d - df_2022_04_ave_d
df_2022_07_04_dif_ave_d = df_2022_07_04_dif_ave_d.set_axis([FEATURES_DIFF],axis = 1)
df_2022_07_04_dif_ave = pd.concat([df_2022_07_04_dif_ave_d,df_2022_04_ave_ans],axis = 1)
df_2022_07_04_dif_ave = df_2022_07_04_dif_ave.dropna()

df_2022 = pd.concat([df_2022_07_ave_d,df_2022_07_04_dif_ave],axis = 1)
# df_2022 = df_2022_07_ave.interpolate()
df_2022 = df_2022.dropna()
df_2022 = df_2022.replace([np.inf, -np.inf], 0)

#df_2022.to_csv("C:\\Users\kohei\Desktop\phenology_data\\確認_2022.csv")

X_train, X_test, y_train, y_test = Data(df_2021.values)

path = get_ccp_alphas(X_train, y_train)

clfs = train_with_alphas(X_train, y_train, path.ccp_alphas)
train_scores = [clf.score(X_train, y_train) for clf in clfs[:-1]]
test_scores = [clf.score(X_test, y_test) for clf in clfs[:-1]]
output_prune_result(path, clfs[:-1], train_scores, test_scores)

#print(clfs[1].n_features_in_)

# df = output_prune_result(path,clfs[:-1],train_scores,test_scores)
# df.to_csv("C:\\Users\kohei\Desktop\phenology_data\\param_2022_ver2.csv")

#print(clfs[7].feature_importances_)

# output_graphs(clfs[5], X_test, y_test)

output_trees(clfs[7], X_train, y_train)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_all")
df_2021_06_all = get_df_for_CFT("2021_06_01_10m_all.csv","2021_06_01_20m_all.csv",2021)
df_geo_2021 = get_geo("2021_08_05_10m_all.csv")
df_2021_08_05_all = get_df_for_CFT("2021_08_05_10m_all.csv","2021_08_05_20m_all.csv",2021)
df_2021_08_28_all = get_df_for_CFT("2021_08_28_10m_all.csv","2021_08_28_20m_all.csv",2021)
df_2021_08_30_all = get_df_for_CFT("2021_08_30_10m_all.csv","2021_08_30_20m_all.csv",2021)

df_2021_08_ave = (df_2021_08_05_all + df_2021_08_28_all + df_2021_08_30_all)/3

df_2021_diff = df_2021_08_ave - df_2021_06_all
df_2021_diff = df_2021_diff.set_axis([FEATURES_DIFF],axis = 1)

df_2021_all = pd.concat([df_2021_08_ave,df_2021_diff],axis = 1)
df_2021_all = df_2021_all.interpolate()
# df_2021_all = df_2021_08_ave.interpolate()
df_2021_all = df_2021_all.fillna(0)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_all")
# df_2022_08_all = get_df_for_CFT("2022_08_10_10m_all.csv","2022_08_10_20m_all.csv",2022)
# df_geo_2022 = get_geo("2022_08_10_10m_all.csv")
# df_2022_06_04_all = get_df_for_CFT("2022_06_04_10m_all.csv","2022_06_04_20m_all.csv",2022)
# df_2022_06_09_all = get_df_for_CFT("2022_06_09_10m_all.csv","2022_06_09_20m_all.csv",2022)
# df_2022_06_19_all = get_df_for_CFT("2022_06_19_10m_all.csv","2022_06_19_20m_all.csv",2022)
# df_2022_06_29_all = get_df_for_CFT("2022_06_29_10m_all.csv","2022_06_29_20m_all.csv",2022)

# df_2022_06_ave = (df_2022_06_04_all + df_2022_06_09_all + df_2022_06_19_all + df_2022_06_29_all)/4

# df_2022_diff = df_2022_08_all - df_2022_06_ave
# df_2022_diff = df_2022_diff.set_axis([FEATURES_DIFF],axis = 1)

# df_2022_all = pd.concat([df_2022_08_all,df_2022_diff],axis = 1)
# df_2022_all = df_2022_all.fillna(0)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_all")
df_2022_07_01_all = get_df_for_CFT("2022_07_01_10m_all.csv","2022_07_01_20m_all.csv",2022)
df_2022_07_29_all = get_df_for_CFT("2022_07_29_10m_all.csv","2022_07_29_20m_all.csv",2022)
df_2022_07_31_all = get_df_for_CFT("2022_07_31_10m_all.csv","2022_07_31_20m_all.csv",2022)
# df_2022_09_04_all = get_df_for_CFT("2022_09_04_10m_all.csv","2022_09_04_20m_all.csv",2022)
# df_2022_09_12_all = get_df_for_CFT("2022_09_12_10m_all.csv","2022_09_12_20m_all.csv",2022)
df_geo_2022 = get_geo("2022_07_31_10m_all.csv")
df_2022_04_12_all = get_df_for_CFT("2022_04_12_10m_all.csv","2022_04_12_20m_all.csv",2022)
df_2022_04_20_all = get_df_for_CFT("2022_04_20_10m_all.csv","2022_04_20_20m_all.csv",2022)
df_2022_04_22_all = get_df_for_CFT("2022_04_22_10m_all.csv","2022_04_22_20m_all.csv",2022)

# df_2022_09_ave = (df_2022_09_12_all + df_2022_09_04_all)/2
df_2022_07_ave = (df_2022_07_01_all + df_2022_07_29_all + df_2022_07_31_all)/3
df_2022_04_ave = (df_2022_04_12_all + df_2022_04_20_all + df_2022_04_22_all)/3

df_2022_diff = df_2022_07_ave - df_2022_04_ave
df_2022_diff = df_2022_diff.set_axis([FEATURES_DIFF],axis = 1)

df_2022_all = pd.concat([df_2022_07_ave,df_2022_diff],axis = 1)
df_2022_all = df_2022_all.interpolate()
df_2022_all = df_2022_all.dropna()
df_2022_all = df_2022_all.replace([np.inf, -np.inf], 0)



# result = clfs[1].predict(df_2021_all)
# df_result = pd.DataFrame(data = result)
# df_result = pd.concat([df_result,df_geo_2021],axis = 1)
# df_result.to_csv("C:\\Users\\kohei\\Desktop\\phenology_data\\CT_result_2022_(7,4)_from_2022_(7,4)_NDCI_clfs1_ver2.csv")
