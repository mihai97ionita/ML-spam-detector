import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_rows", None, "display.max_columns", None)
import os

# Load all our dataset to merge them
df_p = pd.read_csv(".\\datasets\\Youtube01-Psy.csv")
df_k = pd.read_csv(".\\datasets\\Youtube02-KatyPerry.csv")
df_l = pd.read_csv(".\\datasets\\Youtube03-LMFAO.csv")
df_e = pd.read_csv(".\\datasets\\Youtube04-Eminem.csv")
df_s = pd.read_csv(".\\datasets\\Youtube05-Shakira.csv")
frame=[df_p,df_k,df_l,df_e,df_s]
df_merged = pd.concat(frame)
keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]
df_merged_keys = pd.concat(frame,keys=keys)
df_merged_keys.to_csv(".\\datasets\\YoutubeSpam.csv")
df= df_merged_keys #simplificare denumire
df

df_x = df['CONTENT'] # datele de intare
df_y = df['CLASS'].values #datele de iesire

df_x_p =df_p['CONTENT'] #Psy.csv
df_y_p = df_p['CLASS'].values #Psy.csv

df_x_k =df_k['CONTENT'] #KatyPerry.csv
df_y_k = df_k['CLASS'].values #KatyPerry.csv

df_x_l =df_l['CONTENT'] #LMFAO.csv
df_y_l = df_l['CLASS'].values #LMFAO.csv

df_x_e =df_e['CONTENT'] #Eminem.csv
df_y_e = df_e['CLASS'].values #Eminem.csv

df_x_s =df_s['CONTENT'] #Shakira.csv
df_y_s = df_s['CLASS'].values #Shakira.csv

from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()

x_data = Vectorizer.fit_transform(df_x)#all

import os
import pickle

#salvare CountVectorizer
saved_file = open("Vectorizer.pkl","wb")
pickle.dump(Vectorizer,saved_file)
saved_file.close()


x_data_p = CountVectorizer().fit_transform(df_x_p)#Psy.csv

x_data_k = CountVectorizer().fit_transform(df_x_k)#KatyPerry.csv

x_data_l = CountVectorizer().fit_transform(df_x_l)#LMFAO.csv

x_data_e = CountVectorizer().fit_transform(df_x_e)#Eminem.csv

x_data_s = CountVectorizer().fit_transform(df_x_s)#Shakira.csv

#verific daca am serializat Vectorizer
import pickle
saved_file = open("Vectorizer.pkl","rb")
Vectorizer_loaded=pickle.load(saved_file)
saved_file.close()
#print(Vectorizer_loaded.get_feature_names())
#print(len(Vectorizer.get_feature_names()))

# %%train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, df_y, train_size=0.70, random_state=0)
x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_data_p, df_y_p, train_size=0.70, random_state=0)
x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(x_data_k, df_y_k, train_size=0.70, random_state=0)
x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(x_data_l, df_y_l, train_size=0.70, random_state=0)
x_train_e, x_test_e, y_train_e, y_test_e = train_test_split(x_data_e, df_y_e, train_size=0.70, random_state=0)
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x_data_s, df_y_s, train_size=0.70, random_state=0)

# print("x train: ",x_train.shape)
# print("x test: ",x_test.shape)
# print("y train: ",y_train.shape)
# print("y test: ",y_test.shape)

