# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:25:37 2022

@author: kohei
"""

import matplotlib
import pandas as pd
import numpy as np    
import os

def get_df_for_RF(file_1,file_2,surface_type,year):
   
        df_rgb = pd.read_csv(file_1,encoding="shift-jis",engine='python',usecols=[1,2,3,4])
        df_rgb.columns=["Blue","Green","Red","NIR"]
        
        if year == 2022:
            df_rgb = (df_rgb - 1000)/10000
        else:
            df_rgb = df_rgb/10000
        
        
        #file_2 = os.path.split(j)[1]
        df_oth = pd.read_csv(file_2,encoding="shift-jis",engine='python',usecols=[1,2,3,4,5,6])
        df_oth.columns=["B5(705nm)","B6(740nm)","B7(783nm)","B8A(865nm)","B11(SWIR)","B12(2190)"]
        if year == 2022:
            df_oth = (df_oth - 1000)/10000 #2022年用
        else:
            df_oth = df_oth/10000
        
        
        if surface_type == "hisi":
            c = 1
        elif surface_type == "kuromo":
            c = 2
        else:
            c = 0
        
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
        # df_sf["TCT_Wetness"] = 0.2578*df_rgb["Blue"] + 0.2305*df_rgb["Green"] + 0.0883*df_rgb["Red"] + 0.1071*df_rgb["NIR"] + (-0.7611)*df_oth["B11(SWIR)"] + (-0.5308)*df_oth["B12(2190)"]
        
        
        df = pd.concat([df_rgb_oth,df_sf],axis = 1)
        data = pd.DataFrame(index=range(len(df)), columns=['ans'])
        data.fillna(c, inplace=True)
        
        df = pd.concat([df,data],axis = 1)
        df = df.dropna()
        
        return df

os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_hisi_ver2")
df_2021_08_05_hisi_ave = get_df_for_RF("2021_08_05_hisi_ver2.csv","2021_08_05_hisi_other_ver2.csv","hisi",2021)
df_2021_08_28_hisi_ave = get_df_for_RF("2021_08_28_hisi_ver2.csv","2021_08_28_hisi_other_ver2.csv","hisi",2021)
df_2021_08_30_hisi_ave = get_df_for_RF("2021_08_30_hisi_ver2.csv","2021_08_30_hisi_other_ver2.csv","hisi",2021)

def align_data_frame(surface_type,year,*file_names):
    n = int(len(file_names)/2)
    df = {}
    df_sum = pd.DataFrame()
    for i in range(n):
        df[i] = pd.DataFrame()
        df[i] = get_df_for_RF(file_names[2 * i], file_names[2 * i + 1], surface_type, year)
        df_sum = df_sum.add(df[i], fill_value=0)
    df_ave = df_sum / n
    return df_ave
    
align_data_frame("hisi", 2021,"2021_08_05_hisi_ver2.csv","2021_08_05_hisi_other_ver2.csv","2021_08_30_hisi_ver2.csv","2021_08_30_hisi_other_ver2.csv")

    

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_kuromo_ver2")
# df_2021_08_05_kuromo_ave = get_df_for_RF("2021_08_05_kuromo.csv","2021_08_05_kuromo_other_ver2.csv","kuromo",2021)
# df_2021_08_28_kuromo_ave = get_df_for_RF("2021_08_28_kuromo.csv","2021_08_28_kuromo_other_ver2.csv","kuromo",2021)
# df_2021_08_30_kuromo_ave = get_df_for_RF("2021_08_30_kuromo.csv","2021_08_30_kuromo_other_ver2.csv","kuromo",2021)

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2021_water_ver2")
# df_2021_08_05_water_ave = get_df_for_RF("2021_08_05_water.csv","2021_08_05_water_other_ver2.csv","water",2021)
# df_2021_08_28_water_ave = get_df_for_RF("2021_08_28_water.csv","2021_08_28_water_other_ver2.csv","water",2021)
# df_2021_08_30_water_ave = get_df_for_RF("2021_08_30_water.csv","2021_08_30_water_other_ver2.csv","water",2021)

# df_2021_08_hisi_ave = (df_2021_08_05_hisi_ave + df_2021_08_28_hisi_ave + df_2021_08_30_hisi_ave)/3
# df_2021_08_kuromo_ave = (df_2021_08_05_kuromo_ave + df_2021_08_28_kuromo_ave + df_2021_08_30_kuromo_ave)/3
# df_2021_08_water_ave = (df_2021_08_05_water_ave + df_2021_08_28_water_ave + df_2021_08_30_water_ave)/3

# df_2021_08_ave = pd.concat([df_2021_08_hisi_ave,df_2021_08_kuromo_ave,df_2021_08_water_ave],axis = 0)   
# df_2021 = df_2021_08_ave
# df_2021 = df_2021.dropna()

# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_hisi_ver2")
# df_2022_07_01_hisi_ave = get_df_for_RF("2022_07_01_hisi_ver2.csv","2022_07_01_hisi_other_ver2.csv","hisi",2022)
# df_2022_07_29_hisi_ave = get_df_for_RF("2022_07_29_hisi_ver2.csv","2022_07_29_hisi_other_ver2.csv","hisi",2022)
# df_2022_07_31_hisi_ave = get_df_for_RF("2022_07_31_hisi_ver2.csv","2022_07_31_hisi_other_ver2.csv","hisi",2022)
# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_kuromo_ver2")
# df_2022_07_01_kuromo_ave = get_df_for_RF("2022_07_01_kuromo.csv","2022_07_01_kuromo_other_ver2.csv","kuromo",2022)
# df_2022_07_29_kuromo_ave = get_df_for_RF("2022_07_29_kuromo.csv","2022_07_29_kuromo_other_ver2.csv","kuromo",2022)
# df_2022_07_31_kuromo_ave = get_df_for_RF("2022_07_31_kuromo.csv","2022_07_31_kuromo_other_ver2.csv","kuromo",2022)
# os.chdir("C:\\Users\kohei\Desktop\phenology_data\\2022_water_ver2")
# df_2022_07_01_water_ave = get_df_for_RF("2022_07_01_water.csv","2022_07_01_water_other_ver2.csv","water",2022)
# df_2022_07_29_water_ave = get_df_for_RF("2022_07_29_water.csv","2022_07_29_water_other_ver2.csv","water",2022)
# df_2022_07_31_water_ave = get_df_for_RF("2022_07_31_water.csv","2022_07_31_water_other_ver2.csv","water",2022)

# df_2022_07_hisi_ave = (df_2022_07_01_hisi_ave + df_2022_07_29_hisi_ave + df_2022_07_31_hisi_ave)/3
# df_2022_07_kuromo_ave = (df_2022_07_01_kuromo_ave + df_2022_07_29_kuromo_ave + df_2022_07_31_kuromo_ave)/3
# df_2022_07_water_ave = (df_2022_07_01_water_ave + df_2022_07_29_water_ave + df_2022_07_31_water_ave)/3

# df_2022_07_ave = pd.concat([df_2022_07_hisi_ave,df_2022_07_kuromo_ave,df_2022_07_water_ave],axis = 0)
# df_2022 = df_2022_07_ave
# df_2022 = df_2022.replace([np.inf, -np.inf], 0)
# df_2022 = df_2022.dropna()






    