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


def dump_faildict(faildict):
    with open("faildict.json", 'w') as f:
        json.dump(faildict, f, indent=2) 

def read_faildict():
    if os.path.isfile("faildict.json"):
        with open("faildict.json", 'r') as f:
            faildict = json.load(f)
    else:
        faildict={}
    return faildict



faildict=read_faildict()

for i,user_id in enumerate(all_users):
    user_id=int(user_id)
    if str(user_id) in faildict.keys():
        print(user_id ,"already exist in faildict.json")
        continue

    if os.path.isfile("ALL_USER_DF.csv"):
        if user_id in pd.read_csv("ALL_USER_DF.csv")["user_id"].to_numpy():
            print(user_id ,"already exist in ALL_USER_DF.csv")
            continue

    t1=time.time()
    print(user_id)
    try:
        nsluser.clear_user_attributes()
        nsluser.user_id=user_id
    except:
        faildict[user_id]="EC 1"
        print("error while initializing user")
        dump_faildict(faildict)
        continue
    try:
        nsluser.get_migraine_days_sql()
    except:
        faildict[user_id]="EC 2"
        print("error while get migraine_days")
        dump_faildict(faildict)
        continue
    try:
        nsluser.get_temp_and_air()
    except:
        faildict[user_id]="EC 3"
        print("error while loading weather data")
        dump_faildict(faildict)
        continue
    try:
        nsluser.get_spectrum()
    except:
        faildict[user_id]="EC 4"
        print("error while calculating the spectrum")
        dump_faildict(faildict)
        continue
    try:
        nsluser.get_spec_peak()
    except:
        faildict[user_id]="EC 5"
        print("error while calculating the peak of spektrum")
        dump_faildict(faildict)
        continue
    try:
        nsluser.get_user_attributes()
    except:
        faildict[user_id]="EC 6"
        print("error while getting user info")
        dump_faildict(faildict)
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
        faildict[user_id]="EC 7"
        print("error while joining data frames")
        dump_faildict(faildict)
        continue    
    try:
        user_df=pd.DataFrame()
        for attkey in ['user_id','freq_specmax', 'T_specmax', 'specmax', 'specmax_outstanding', 'specmax_ratio', 'first_name', 'gender']:
            user_df[attkey]=[(vars(nsluser)[attkey])]
    except:
        faildict[user_id]="EC 8"
        print("error while compiling user_df")
        dump_faildict(faildict)
        continue   
    try:
        if i%1==0:
            if os.path.isfile("ALL_USER_DF.csv"):
                user_df.to_csv("ALL_USER_DF.csv", mode="a", index=False, header=False)
            else:
                user_df.to_csv("ALL_USER_DF.csv",index=False)
    except:
        faildict[user_id]="EC 9"
        print("error while appending to ALL_USER_DF.csv")
        dump_faildict(faildict)
        continue
    t2=time.time()
    ptime=round(t2-t1,2)
    print("    processing time:", ptime)

