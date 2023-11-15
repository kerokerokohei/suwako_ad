# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:45:49 2023

@author: kohei
"""
import pandas as pd
import os
import numpy as np

class MyParentClass:
    def __init__(self, arg1):
        self.arg1 = arg1

    def my_method(self):
        print("This is my parent method")

class MyClass(MyParentClass):
    def __init__(self, arg1, arg2):
        super().__init__(arg1)
        self.arg2 = arg2

    def my_method(self):
        super().my_method()
        print("This is my method")

# my_instance = MyClass("argument1", "argument2")
# # print(my_instance.arg1)
# print(my_instance.arg2)
# my_instance.my_method()

os.chdir("C:\\Users\kohei\Desktop\Analysis_Paper\Suwako_all")
# df_2021_06_01_all = get_df_for_CFT("2021_06_01_all_10m.csv","2021_06_01_all_20m.csv",2021)
df_geo = pd.read_csv("2021_06_01_all_10m.csv",encoding="shift-jis")
array = np.vstack([df_geo["xcoord"],df_geo["ycoord"]])
# df_geo.columns = ["xcoord","ycoord"]
df_geo = pd.DataFrame(data = array)
print(df_geo)
