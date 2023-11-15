# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:18:41 2022

@author: kohei
"""

from sklearn.datasets import load_iris
from sklearn import tree
from csv import reader


with open('NDWI.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # print(csv_reader)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_sf1 = list(csv_reader)
    # list_of_rows_sf1_f = map(float, list_of_rows_sf1)
    # print(list_of_rows)
# print(list_of_rows_sf1)
with open('NDWI_ans.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # print(csv_reader)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_sf1_ans = list(csv_reader)
# print(list_of_rows_sf1_ans)
with open('area_estimation_test2_source.csv', 'r') as csv_file:
    csv_reader = reader(csv_file)
    # print(csv_reader)
    # Passing the cav_reader object to list() to get a list of lists
    list_of_rows_sf1_predict = list(csv_reader)

def main():
  # アヤメのデータを読み込む
  # iris = load_iris()
  import numpy as np
  # アヤメの素性(説明変数)はリスト
  # 順にがく片の幅，がく片の長さ，花弁の幅，花弁の長さ
  # print(iris.data)

  # アヤメの種類(目的変数)は3種類(3値分類)
  # print(iris.target)
  """
  with open('SF1_7_7.csv', 'r') as csv_file:
      csv_reader = reader(csv_file)
      # print(csv_reader)
      # Passing the cav_reader object to list() to get a list of lists
      list_of_rows_sf1 = list(csv_reader)
      # print(list_of_rows)
  # print(list_of_rows_sf1)
  with open('SF1_7_7_ans.csv', 'r') as csv_file:
      csv_reader = reader(csv_file)
      # print(csv_reader)
      # Passing the cav_reader object to list() to get a list of lists
      list_of_rows_sf1_ans = list(csv_reader)
  # print(list_of_rows_sf1_ans)
  """

  '''
    今回の内容と関係ありそうなパラメータ
    criterion = 'gini' or 'entropy' (default: 'gini')                        # 分割する際にどちらを使うか
    max_depth = INT_VAL or None (default: None)                              # 作成する決定木の最大深さ
    min_samples_split = INT_VAL (default: 2)                                 # サンプルを分割する際の枝の数の最小値
    min_samples_leaf = INT_VAL (default: 1)                                  # 1つのサンプルが属する葉の数の最小値
    min_weight_fraction_leaf = FLOAT_VAL (default: 0.0)                      # 1つの葉に属する必要のあるサンプルの割合の最小値
    max_leaf_nodes = INT_VAL or None (default: None)                         # 作成する葉の最大値(設定するとmax_depthが無視される)
    class_weight = DICT, LIST_OF_DICTS, 'balanced', or None (default: None)  # 各説明変数に対する重み
    presort = BOOL (default: False)                                          # 高速化のための入力データソートを行うか
  '''
  # モデルを作成
  clf = tree.DecisionTreeClassifier(max_depth = 3)
  clf = clf.fit(list_of_rows_sf1, list_of_rows_sf1_ans)

  # 作成したモデルを用いて予測を実行
  predicted = clf.predict(list_of_rows_sf1_predict)

  # 予測結果の出力(正解データとの比較)
  print('=============== predicted ===============')
  print(predicted)
  # print('============== correct_ans ==============')
  # print(iris.target)
  # print('=============== id_rate =================')
  # print(sum(predicted == iris.target) / len(iris.target))
  
  # 分類木に描けた結果のcsv
  np.savetxt("predict_result_test2.csv", predicted, delimiter =",",fmt ='% s')
  
  '''
    feature_namesには各説明変数の名前を入力
    class_namesには目的変数の名前を入力
    filled = Trueで枝に色が塗られる
    rounded = Trueでノードの角が丸くなる
  '''
  # 学習したモデルの可視化
  # これによって同じディレクトリにiris_model.dotが出力されるので中身をwww.webgraphviz.comに貼り付けたら可視化できる
  f = tree.export_graphviz(clf, out_file = 'test_sf1.dot' ,class_names = ('water','hisi','kuromo'),
                            filled = True, rounded = True)#class_names = (0,1,2)

if __name__ == '__main__':
  main()

#print("a")
#print(iris.target)




