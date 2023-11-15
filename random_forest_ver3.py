# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:27:52 2022

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
from sklearn.svm import SVC

FEATURES = ["Blue","Green","Red","NIR","B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)",
           "B12(2190)",'sf1_ave','WAVI','NDAVI','NDVI','ave_123','MNDWI','NDMI']



#print(X)

#ccp_alphasの計算
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


#print(path.ccp_alphas)

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

    # Feature Importance 出力
    importances = pd.DataFrame({'Importance':clf.feature_importances_}, index=FEATURES)
    importances.sort_values('Importance', ascending=False).head(10).sort_values('Importance', ascending=True).plot.barh(ax=axes[0, 1], grid=True)

    # ROC曲線出力
    RocCurveDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 0])
    axes[1, 0].set_title('ROC(Receiver Operating Characteristic) Curve')

    # 適合率-再現率グラフ出力
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:,1], ax=axes[1, 1])

    plt.show()


#決定木描画
CLASS_NAMES = ['Water', 'Hisi' , 'Kuromo']
def output_trees(clf, X_train, y_train):
    plt.figure(figsize=(18,7))
    plot_tree(clf, filled=True, feature_names=FEATURES, class_names=CLASS_NAMES, fontsize=9)
    plt.show()
    """
    #viz = dtreeviz(
        clf,
        X_train, 
        y_train,
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        target_name='y', 
        fontname='DejaVu Sans' #fontname='Hiragino Sans'
    ) 
    display(viz)
"""

def rforesti(array):
    X_train = array[:,0:19]
    y_train = array[:,20]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3,random_state=1234)

    rf = RFC(n_estimators=10, max_depth=4, bootstrap=True, random_state=1234)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)

    accu = accuracy_score(y_test, y_pred)
    print('accuracy = {:>.4f}'.format(accu))
    
    # Feature Importance
    fti = rf.feature_importances_ 
    #print(fti)

    plt.figure(figsize = (10,6))
    plt.barh(y = range(len(fti)), width = fti)
    plt.yticks(ticks = range(len(FEATURES)), labels = FEATURES)
    plt.show()

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
                    
         
        #df_rgb = df_rgb.drop(q)#hisi
        #df_rgb = df_rgb.drop(range(625,628))#kuromo
        #df_rgb = df_rgb.drop(range(5653,5678))#water
        
        df_rgb_oth = pd.concat([df_rgb,df_oth],axis = 1)
        #df_rgb_oth_pheno[file] = df_rgb_oth.mean()
        
        df_sf = pd.DataFrame()
        df_sf["sf1_ave"] = (df_rgb["NIR"] - ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))/(df_rgb["NIR"] + ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3))
        df_sf["WAVI"] = (1.5*(df_rgb["NIR"] - df_rgb["Blue"]))/(0.5 + df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDAVI"] = (df_rgb["NIR"] - df_rgb["Blue"])/(df_rgb["NIR"] + df_rgb["Blue"])
        df_sf["NDVI"] = (df_rgb["NIR"] - df_rgb["Red"])/(df_rgb["NIR"] + df_rgb["Red"])
        df_sf["ave_123"] = ((df_rgb["Blue"] + df_rgb["Green"] + df_rgb["Red"])/3)
        df_sf["MNDWI"] = (df_rgb["Green"] - df_oth["B11(SWIR)"])/(df_rgb["Green"] + df_oth["B11(SWIR)"])
        df_sf["NDMI"] = (df_rgb["NIR"] - df_oth["B11(SWIR)"])/(df_rgb["NIR"] + df_oth["B11(SWIR)"])
        #df_sf["NDCI"] = (df_rgb["Red"] - df_oth["B5(705nm)"])/(df_rgb["Red"] + df_oth["B5(705nm)"])
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        data = pd.DataFrame(index=range(len(df)), columns=['ans'])
        data.fillna(c, inplace=True)
        
        df = pd.concat([df,data],axis = 1)
        df = df.dropna()
        
        return df

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi\\2022_08_hisi")
df_2022_08_hisi_ave = get_df_for_RF("2022_08_10_hisi.csv","2022_08_10_hisi_other.csv","hisi",2022)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo\\2022_08_kuromo")
df_2022_08_kuromo_ave = get_df_for_RF("2022_08_10_kuromo.csv","2022_08_10_kuromo_other.csv","kuromo",2022)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water\\2022_08_water")
df_2022_08_water_ave = get_df_for_RF("2022_08_10_water.csv","2022_08_10_water_other.csv","water",2022)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi\\2022_06_hisi")
df_2022_06_04_hisi_ave = get_df_for_RF("2022_06_04_hisi.csv","2022_06_04_hisi_other.csv","hisi",2022)
df_2022_06_09_hisi_ave = get_df_for_RF("2022_06_09_hisi.csv","2022_06_09_hisi_other.csv","hisi",2022)
df_2022_06_19_hisi_ave = get_df_for_RF("2022_06_19_hisi.csv","2022_06_19_hisi_other.csv","hisi",2022)
df_2022_06_29_hisi_ave = get_df_for_RF("2022_06_29_hisi.csv","2022_06_29_hisi_other.csv","hisi",2022)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo\\2022_06_kuromo")
df_2022_06_04_kuromo_ave = get_df_for_RF("2022_06_04_kuromo.csv","2022_06_04_kuromo_other.csv","kuromo",2022)
df_2022_06_09_kuromo_ave = get_df_for_RF("2022_06_09_kuromo.csv","2022_06_09_kuromo_other.csv","kuromo",2022)
df_2022_06_19_kuromo_ave = get_df_for_RF("2022_06_19_kuromo.csv","2022_06_19_kuromo_other.csv","kuromo",2022)
df_2022_06_29_kuromo_ave = get_df_for_RF("2022_06_29_kuromo.csv","2022_06_29_kuromo_other.csv","kuromo",2022)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water\\2022_06_water")
df_2022_06_04_water_ave = get_df_for_RF("2022_06_04_water.csv","2022_06_04_water_other.csv","water",2022)
df_2022_06_09_water_ave = get_df_for_RF("2022_06_09_water.csv","2022_06_09_water_other.csv","water",2022)
df_2022_06_19_water_ave = get_df_for_RF("2022_06_19_water.csv","2022_06_19_water_other.csv","water",2022)
df_2022_06_29_water_ave = get_df_for_RF("2022_06_29_water.csv","2022_06_29_water_other.csv","water",2022)

df_2022_08_ave = pd.concat([df_2022_08_hisi_ave,df_2022_08_kuromo_ave,df_2022_08_water_ave],axis = 0)
df_2022_08_ave_d = df_2022_08_ave.loc[:,FEATURES]
df_2022_08_ave_d = df_2022_08_ave_d.reset_index()
df_2022_08_ave_d = df_2022_08_ave_d.drop("index",axis = 1)

df_2022_06_hisi_ave = (df_2022_06_04_hisi_ave + df_2022_06_09_hisi_ave + df_2022_06_19_hisi_ave + df_2022_06_29_hisi_ave)/4
df_2022_06_kuromo_ave = (df_2022_06_04_kuromo_ave + df_2022_06_09_kuromo_ave + df_2022_06_19_kuromo_ave + df_2022_06_29_kuromo_ave)/4
df_2022_06_water_ave = (df_2022_06_04_water_ave + df_2022_06_09_water_ave + df_2022_06_19_water_ave + df_2022_06_29_water_ave)/4

df_2022_06_ave = pd.concat([df_2022_06_hisi_ave,df_2022_06_kuromo_ave,df_2022_06_water_ave])

df_2022_06_ave_ans = df_2022_06_ave.loc[:,"ans"]
df_2022_06_ave_ans = df_2022_06_ave_ans.reset_index()
df_2022_06_ave_ans = df_2022_06_ave_ans.drop("index",axis = 1)

df_2022_06_ave_d = df_2022_06_ave.loc[:,FEATURES]
df_2022_06_ave_d = df_2022_06_ave_d.reset_index()
df_2022_06_ave_d = df_2022_06_ave_d.drop("index",axis = 1)

df_2022_08_06_dif_ave_d = df_2022_08_ave_d - df_2022_06_ave_d
df_2022_08_06_dif_ave = pd.concat([df_2022_08_06_dif_ave_d,df_2022_06_ave_ans],axis = 1)
df_2022_08_06_dif_ave = df_2022_08_06_dif_ave.dropna()


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_hisi\\2021_06_hisi")
df_2021_06_hisi_ave = get_df_for_RF("2021_06_01_hisi.csv","2021_06_01_hisi_other.csv","hisi",2021)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_kuromo\\2021_06_kuromo")
df_2021_06_kuromo_ave = get_df_for_RF("2021_06_01_kuromo.csv","2021_06_01_kuromo_other.csv","kuromo",2021)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_water\\2021_06_water")
df_2021_06_water_ave = get_df_for_RF("2021_06_01_water.csv","2021_06_01_water_other.csv","water",2021)

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_hisi\\2021_08_hisi")
df_2021_08_05_hisi_ave = get_df_for_RF("2021_08_05_hisi.csv","2021_08_05_hisi_other.csv","hisi",2021)
df_2021_08_28_hisi_ave = get_df_for_RF("2021_08_28_hisi.csv","2021_08_28_hisi_other.csv","hisi",2021)
df_2021_08_30_hisi_ave = get_df_for_RF("2021_08_30_hisi.csv","2021_08_30_hisi_other.csv","hisi",2021)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_kuromo\\2021_08_kuromo")
df_2021_08_05_kuromo_ave = get_df_for_RF("2021_08_05_kuromo.csv","2021_08_05_kuromo_other.csv","kuromo",2021)
df_2021_08_28_kuromo_ave = get_df_for_RF("2021_08_28_kuromo.csv","2021_08_28_kuromo_other.csv","kuromo",2021)
df_2021_08_30_kuromo_ave = get_df_for_RF("2021_08_30_kuromo.csv","2021_08_30_kuromo_other.csv","kuromo",2021)


os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_water\\2021_08_water")
df_2021_08_05_water_ave = get_df_for_RF("2021_08_05_water.csv","2021_08_05_water_other.csv","water",2021)
df_2021_08_28_water_ave = get_df_for_RF("2021_08_28_water.csv","2021_08_28_water_other.csv","water",2021)
df_2021_08_30_water_ave = get_df_for_RF("2021_08_30_water.csv","2021_08_30_water_other.csv","water",2021)

df_2021_06_ave = pd.concat([df_2021_06_hisi_ave,df_2021_06_kuromo_ave,df_2021_06_water_ave],axis = 0)
df_2021_06_ave_d = df_2021_06_ave.loc[:,FEATURES]
df_2021_06_ave_d = df_2021_06_ave_d.reset_index()
df_2021_06_ave_d = df_2021_06_ave_d.drop("index",axis = 1)

df_2021_08_hisi_ave = (df_2021_08_05_hisi_ave + df_2021_08_28_hisi_ave + df_2021_08_30_hisi_ave)/3
df_2021_08_kuromo_ave = (df_2021_08_05_kuromo_ave + df_2021_08_28_kuromo_ave + df_2021_08_30_kuromo_ave)/3
df_2021_08_water_ave = (df_2021_08_05_water_ave + df_2021_08_28_water_ave + df_2021_08_30_water_ave)/3

df_2021_08_ave = pd.concat([df_2021_08_hisi_ave,df_2021_08_kuromo_ave,df_2021_08_water_ave],axis = 0)
df_2021_08_ave = df_2021_08_ave.dropna()

df_2021_08_ave_ans = df_2021_08_ave.loc[:,"ans"]
df_2021_08_ave_ans = df_2021_08_ave_ans.reset_index()
df_2021_08_ave_ans = df_2021_08_ave_ans.drop("index",axis = 1)

df_2021_08_ave_d = df_2021_08_ave.loc[:,FEATURES]
df_2021_08_ave_d = df_2021_08_ave_d.reset_index()
df_2021_08_ave_d = df_2021_08_ave_d.drop("index",axis = 1)

df_2021_08_06_dif_ave_d = df_2021_08_ave_d - df_2021_06_ave_d
df_2021_08_06_dif_ave = pd.concat([df_2021_08_06_dif_ave_d,df_2021_08_ave_ans],axis = 1)
df_2021_08_06_dif_ave = df_2021_08_06_dif_ave.dropna()


#print(df_2022_08_06_dif_ave)

def Data(array):
    X = array[:,0:19]
    Y = array[:,20]
   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=1234)
    return X_train, X_test, y_train, y_test

# kf = KFold(n_splits = 3, shuffle = True, random_state = 1234)
# def Kfold(array):
#     array_X = array.iloc[:,0:19]
#     array_y = array.iloc[:,20]
#     for train_index, test_index, in kf.split(array):
#         X_train = array_X.iloc[train_index]
#         X_test  = array_X.iloc[test_index]
#         y_train = array_y.iloc[train_index]
#         y_test  = array_y.iloc[test_index]
#         #print(X_train,X_test,y_train,y_test)
#         path = get_ccp_alphas(X_train, y_train)
#         clfs = train_with_alphas(X_train, y_train, path.ccp_alphas)
#         train_scores = [clf.score(X_train, y_train) for clf in clfs[:-1]]
#         test_scores = [clf.score(X_test, y_test) for clf in clfs[:-1]]
#         output_prune_result(path,clfs[:-1],train_scores,test_scores)
    
    

# CLASS_NAMES = ['Water', 'Hisi' , 'Kuromo']
# def output_trees(clf):
#     plt.figure(figsize=(18,7))
#     plot_tree(clf, filled=True, feature_names=FEATURES, class_names=CLASS_NAMES, fontsize=9)
#     plt.show()

# path = get_ccp_alphas(X_train, y_train)
# clfs = train_with_alphas(X_train, y_train, path.ccp_alphas)

# #スコア算出
# train_scores = [clf.score(X_train, y_train) for clf in clfs[:-1]]
# test_scores = [clf.score(X_test, y_test) for clf in clfs[:-1]]

# output_prune_result(path, clfs[:-1], train_scores, test_scores)

# output_trees(clfs[10], X_train, y_train)

df_2021 = pd.concat([df_2021_08_ave,df_2021_08_06_dif_ave],axis = 0)
df_2022 = pd.concat([df_2022_08_ave,df_2022_08_06_dif_ave],axis = 0)
#df_2021.to_csv("C:\\Users\kohei\Desktop\phenology_data\\forward_2021.csv")


#rint(df_2022)

#X_train, X_test, y_train, y_test = Data(df_2021.values)

# path = get_ccp_alphas(X_train, y_train)

# clfs = train_with_alphas(X_train, y_train, path.ccp_alphas)
# train_scores = [clf.score(X_train, y_train) for clf in clfs[:-1]]
# test_scores = [clf.score(X_test, y_test) for clf in clfs[:-1]]

# #print(clfs[1].n_features_in_)

# #df = output_prune_result(path,clfs[:-1],train_scores,test_scores)
# #df.to_csv("C:\\Users\kohei\Desktop\phenology_data\\param_2022.csv")

# #output_graphs(clfs[13], X_test, y_test)

# output_trees(clfs[17], X_train, y_train)

#Kfold(df_2021)

# rforesti(df_2021.values)
# rforesti(df_2022.values)

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
                              cv = 5,
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

#def CV(clf)


#RF(X_train, X_test, y_train, y_test)

# def SVM(X_train, X_test, y_train, y_test):
#     model = SVC()               # インスタンス生成
#     model.fit(X_train, y_train) # SVM実行
#     predicted = model.predict(X_test) # テストデーテへの予測実行
#     print(accuracy_score(y_test, predicted))

#SVM(X_train, X_test, y_train, y_test)





