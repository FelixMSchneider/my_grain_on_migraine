#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import pandas as pd
import pylab as plt
import numpy as np

from nsl_user import NslUser
nsluser = NslUser(10)
nsluser.get_all_user_id()
all_users=nsluser.df_all_userID.to_numpy()[:,0]


import time
import os


import json


def dump_faillist(faillist):
    with open("faillist.json", 'w') as f:
        json.dump(faillist, f, indent=2) 

def read_faillist():
    if os.path.isfile("faillist.json"):
        with open("faillist.json", 'r') as f:
            faillist = json.load(f)
    else:
        faillist=[]
    return faillist



faillist=read_faillist()

for i,user_id in enumerate(all_users):
    if user_id in faillist:
        print((user_id ,"already exist in faillist.json")
        continue
    if os.path.isfile("ALL_USER_DF.csv"):
        if user_id in pd.read_csv("ALL_USER_DF.csv")["user_id"].to_numpy():
            print(user_id ,"already exist in ALL_USER_DF.csv")
            continue

    t1=time.time()
    print(user_id)
    try:
        nsluser.user_id=user_id
        nsluser.get_user_attributes()
        nsluser.feature_dict={}
        nsluser.get_migraine_days_sql()
        nsluser.get_temp_and_air()
        nsluser.get_spectrum()
        nsluser.get_spec_peak()
    except:
        faillist.append(user_id)        
        print("error while loading data from DB")
        dump_faillist(faillist)
        continue
    try:    
        df_spec=pd.DataFrame(nsluser.spec_dict)
        df_spec.to_csv("./STORAGE/UID_"+str(user_id)+"_spec.csv")
        
        df_hd=pd.DataFrame(nsluser.feature_dict["migraine_days"])
        df_hd.set_index("event_time", inplace=True)
        df_hd.to_csv("./STORAGE/UID_"+str(user_id)+"_HD.csv")
            
        df1=pd.DataFrame(nsluser.feature_dict["temperature"])
        df1.set_index("event_time", inplace=True) 
        df2=pd.DataFrame(nsluser.feature_dict["airpressure"])
        df2.set_index("event_time", inplace=True)
        df_weather=df1.join(df2)
        df_weather.to_csv("./STORAGE/UID_"+str(user_id)+"_weather.csv")
    except:
        faillist.append(user_id)        
        print("error while joining data frames")
        dump_faillist(faillist)
        continue    
    try:
        user_df=pd.DataFrame()
        for attkey in ['user_id','freq_specmax', 'T_specmax', 'specmax', 'specmax_outstanding', 'specmax_ratio', 'first_name', 'gender']:
            user_df[attkey]=[(vars(nsluser)[attkey])]
    except:
        faillist.append(user_id)        
        print("error while compiling user_df")
        dump_faillist(faillist)
        continue   
    try:
        if i%1==0:
            if os.path.isfile("ALL_USER_DF.csv"):
                user_df.to_csv("ALL_USER_DF.csv", mode="a", index=False, header=False)
            else:
                user_df.to_csv("ALL_USER_DF.csv",index=False)
    except:
        faillist.append(user_id)        
        print("error while appending to ALL_USER_DF.csv")
        dump_faillist(faillist)
        continue
    t2=time.time()
    ptime=round(t2-t1,2)
    print("    processing time:", ptime)

