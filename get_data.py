# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 04:12:16 2019

@author: Vincent, Sean
"""
# 1. download all the data as zip
# 2. change the directory to the data location (make sure book is in a different dir)

import glob, os, pandas as pd
import random
os.chdir(r"D:/data_zip")     # path of categories
i =1
dff= pd.DataFrame()

for file in glob.glob("*.gz"):
    print(file)
    data_df = pd.read_json(file, lines=True, compression='infer')    # read json file
    print(data_df.shape)
    data_df=data_df.drop_duplicates(subset='reviewText')
    print(data_df.shape)
    data_df = data_df.sample(n = 140)           # draw 500 random samples
    data_df['Category'] = file.replace('_5.json.gz','')
    i+=1
    dff=dff.append(data_df)
    print("---------")

dff.reset_index(inplace=True , drop = True)
# large file implementation
# computer couldnt handle the big files, so taking them in chunks
os.chdir(r"D:/data_zip/large_data")     # path of categories
d = pd.DataFrame()
for file in glob.glob("*.gz"):
    print(file)
    for chunk in pd.read_json(file, chunksize=200000,lines=True, compression='infer'):
        print(type(chunk))
        df = chunk
        print(df.shape)
        df=df.drop_duplicates(subset='reviewText')
        print(df.shape)
        df = df.sample(n = 140)
        df['Category'] = file.replace('_5.json.gz','')
        d = d.append(df)
        print("---------")
        break
data=pd.concat([dff,d], ignore_index = True)
data.to_csv(r"AMAZON.csv",index=False)#,columns=['overall','verified','reviewTime','reviewerID','asin','style','reviewerName','reviewText','summary','unixReviewTime','vote','	image','Category'])
