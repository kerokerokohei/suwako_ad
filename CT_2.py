# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:58:58 2022

@author: kohei
"""

#from dtreeviz.trees import dtreeviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

"""
FEATURES = ['X0', 'X1', 'X2', 'X3', 'X4']
def make_df():
    X, y = make_classification(n_samples=1100, n_features=5, n_redundant=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = make_df()
#print(y_train)

#data = np.loadtxt("Training_data_spectral_features.csv", delimiter = ",")
"""
FEATURES = ['SF1','SF1_ave','NDWI','WAVI','NDAVI','ANS']
df_train_data = pd.read_csv("Training_data_spectral_features.csv")
array = df_train_data.values
X = array[:,0:4]
Y = array[:,5]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1234)
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

path = get_ccp_alphas(X_train, y_train)
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

clfs = train_with_alphas(X_train, y_train, path.ccp_alphas)

#print(clfs[3])

#スコア算出
train_scores = [clf.score(X_train, y_train) for clf in clfs[:-1]]
test_scores = [clf.score(X_test, y_test) for clf in clfs[:-1]]

print(train_scores)
print("accuracy")
print(test_scores)

#枝刈り単位の情報参照
def output_prune_result(path, clfs, train_scores, test_scores):
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig=plt.figure(figsize=(10,12))
    
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

output_prune_result(path, clfs[:-1], train_scores, test_scores)


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


#output_graphs(clfs[11], X_test, y_test)

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
output_trees(clfs[3], X_train, y_train)

