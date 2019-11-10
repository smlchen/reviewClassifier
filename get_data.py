# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 04:12:16 2019

@author: Vincent, Sean
"""
# 1. download all the data
# 2. change the directory to the data location (make sure book is in a different dir)
import glob, os, pandas as pd
os.chdir(r"/Users/john-/Documents/UC Davis/ECS 171/GroupProject/data")     # path of categories
i =1
dff= pd.DataFrame()

for file in glob.glob("*.json"):
    print(file)
    data_df = pd.read_json(file, lines=True)    # read json file
    data_df = data_df.sample(n = 500)           # draw 500 random samples
    data_df['Category'] = file
    i+=1
    dff=dff.append(data_df)
    
dff.reset_index(inplace=True , drop = True)
# some bug where the books json data crashes the bug
# quick bug fix for fast rollout but will work on a better solution

df=pd.DataFrame()
for chunk in pd.read_json(r"/Users/john-/Documents/UC Davis/ECS 171/GroupProject/data/Books/Books_5.json", chunksize=200000,lines=True):
    df = df.append(chunk)
    df = df.sample(n = 500)
    df['Category'] = 'Books_5.json'
    df.reset_index(inplace=True , drop = True)
    break
data=pd.concat([dff,df], ignore_index = True)
    
data.to_csv(r"AMAZON.csv",index=False,columns=['overall','verified','reviewTime','reviewerID',	'asin','style',	'reviewerName','reviewText','summary','unixReviewTime','vote','	image','Category'])
