'''
module to get user features for nsl_timeline database
'''


import sqlalchemy as db
import os
import pandas as pd
import pylab as plt
import numpy as np
import datetime


class NslUser:
    def __init__(self,user_id=11):
        self.user_id  = user_id
        self.HOST     = 'localhost' # '127.0.0.1'
        self.PORT     = '5432'
        self.USERNAME = 'postgres'
        self.PASSWORD = 'postgres'
        self.DB       = 'nsl_timeline'
        self.feature_dict={}
        self.cs = f"postgresql://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DB}"
        self.engine = db.create_engine(self.cs, echo=False)
        self.featurelist=['migraine_days','temperature', 'airpressure']
        self.user_attributes=["first_name", "gender", "T_specmax", "specmax_outstanding", "specmax", "specmax_ratio"]

    def __str__(self):

        return_str =        "userId     = " + str(self.user_id) 

        for attr in self.user_attributes:
            if attr in dir(self):
                return_str += "\n"+attr+" = " + str(vars(self)[attr]) 
    
        if len(self.feature_dict.keys()) > 0:
            return_str += "\n\nFeatures in feature_dict:" 
        
        for feature in self.feature_dict.keys():
            return_str += "\n"+ feature 

        return return_str

    def __repr__(self):
        return self.__str__()

    def _get_migraine_days(self,df):
        '''
        get days at which the usere head headache
        called by get_feature(feature) if feature = migraine_days
        input:
        df: pandas.DataFrame created by get_feature function
        return:
        self.timeaxis   Timestamp : timeaxis 
        self.iaxis      integer   : time in days starting at 0
        self.migdays:   binary    : 0: no headache 1: headache

        '''
        starttime= df["from_days"][0]- datetime.timedelta(1)
        stoptime = list(df["to_days"])[-1] + datetime.timedelta(1)
        #print(starttime, stoptime)

        newtime=starttime

        dt=datetime.timedelta(1) # 1 day

        timeaxis=[]
        iaxis=[]
        i=0
        md=0
        mem=0
        mds=[]
        while newtime<=stoptime:
            #print(newtime, md)
            if mem==0:
                md=0
            else:
                md=1
    
            newtime=starttime + i*dt

            if datetime.datetime(newtime.year, newtime.month, newtime.day) in list(pd.to_datetime(df["from_days"])):
                md=1
                #keep mem parameter to one if bis-day is not the same as von-day
                if datetime.datetime(newtime.year, newtime.month, newtime.day) not in list(pd.to_datetime(df["to_days"])):
                    mem=1
                else:
                    mem=0
            # as soon as newday is a bis-day set mem back to 0  --> the next day will be no migraine day        
            if mem==1 and datetime.datetime(newtime.year, newtime.month, newtime.day) in list(pd.to_datetime(df["to_days"])):
                mem=0
        
            mds.append(md)
            timeaxis.append(newtime)
            iaxis.append(i)
            i+=1
        self.timeaxis=timeaxis
        self.iaxis=iaxis
        self.migdays=mds

    def get_feature(self, feature):
        '''
        send a request to the database in order to retrieve features from featurelist
        features can be 
        "migraine_days", "temperature", "airpressure" 

        return:
        df: pandas DataFrame
             columns['event_time', feature] 
        '''

        feature_key=feature
        if feature=="migraine_days":
            feature_key="headacheType"

        if feature not in self.featurelist:
            print("feature not supported")
            print("choose feature from", self.featurelist)
            return None
        query= """
            SELECT 
                   *
            from 
                quantity q join factor f 
                    on q.user_id = f.user_id
                       and q.server_factor_id = f.server_factor_id
            where
                 global_id = '""" +str(feature_key)+"""' and
                 q.deleted_at is null and f.user_id="""+str(self.user_id)+"""
            order by lower(q.event_localtime_range) 
            """
        result = self.engine.execute(query).fetchall()
        df=pd.DataFrame(result)
        if feature=="migraine_days":
            df["from"] = df["event_time_range"].apply(lambda x: x.lower)
            df["to"]   = df["event_time_range"].apply(lambda x: x.upper)
            df["from"] = df["from"].apply(lambda x: pd.Timestamp(x))
            df["to"]   = df["to"].apply(lambda x: pd.Timestamp(x))
            #df=df.sort_values("from")
            df["from_days"] = df["from"].apply(lambda x: datetime.datetime(x.year, x.month,x.day))
            df["to_days"]   = df["to"].apply(lambda x: datetime.datetime(x.year, x.month,x.day))
            self._get_migraine_days(df)
            dfm=pd.DataFrame()
            dfm["event_time"]=self.timeaxis
            dfm["iaxis"]=self.iaxis
            dfm["migraine_days"]=self.migdays
            self.feature_dict[feature]=dfm[["event_time", "iaxis",feature]]
 
        else:
            df["event_time"] = df["event_time_range"].apply(lambda x: x.lower)
            df[feature] = df["value"]
            #df.set_index("event_time")
            self.feature_dict[feature]=df[["event_time", feature]]

    def get_temp_and_air(self):
        query='''
        with CTE_TABLE_1 AS
        (
        SELECT q.user_id as user_id,lower(event_time_range) as etr,value as airpressure from
        quantity q join factor f on q.user_id = f.user_id and q.server_factor_id = f.server_factor_id 
        where global_id in ('airpressure') and q.deleted_at is null and q.user_id= '''+str(self.user_id)+''' 
        order by lower(event_time_range)
        )
        
        , CTE_TABLE_2 AS
        (
        SELECT q.user_id,lower(event_time_range) as etr,value as temperature from
        quantity q join factor f on q.user_id = f.user_id and q.server_factor_id = f.server_factor_id 
        where global_id in ('temperature') and q.deleted_at is null and q.user_id= '''+str(self.user_id)+'''
        order by lower(event_time_range)
        )


        SELECT ct1.user_id, ct1.etr::date, ct1.airpressure, ct2.temperature FROM
        CTE_TABLE_1 ct1 JOIN CTE_TABLE_2 ct2 on ct1.etr = ct2.etr
        '''
        result = self.engine.execute(query).fetchall()
        df=pd.DataFrame(result)
        df["event_time"] = df["etr"]
        for feature in "airpressure", "temperature":
            df2=pd.DataFrame()
            df2[feature] = df[feature]
            self.feature_dict[feature]=df[["event_time", feature]]


    def get_migraine_days_sql(self):
        query='''   
        with CTE1 as (
        SELECT event_time_range, value from
                quantity q join factor f on q.user_id = f.user_id and q.server_factor_id = f.server_factor_id
                where global_id in ('headacheType') and q.deleted_at is null and q.user_id= '''+str(self.user_id)+'''
                order by lower(event_time_range))
        , CTE2 as
            (        
            select 
             generate_series(lower(event_time_range)::date, upper(event_time_range)::date, '1 day') as hd, value from CTE1
            )
        , CTE3 as 
            (
            select min(hd) as min_hd, max(hd) as max_hd from CTE2
            )

        , CTE_timeline as (
                select
                generate_series(min_hd::date, max_hd::date, '1 day') as tl from CTE3
                        )

        select  tl,case when value is null then 0 else value end as value_ 
        from CTE_timeline cte_t left join CTE2 c2 on c2.hd = cte_t.tl order by cte_t.tl;
        '''
        result = self.engine.execute(query).fetchall()
        df=pd.DataFrame(result)
        df["event_time"] = df["tl"]
        df["migraine_days"] = df["value_"]
        t0=df["event_time"][0]
        df["iaxis"]=df["event_time"].apply(lambda x: (x-t0).total_seconds()/60/60/24)
        self.feature_dict["migraine_days"]=df[["event_time", "iaxis","migraine_days"]]


    def plot_feature(self, feature):
        if feature not in self.feature_dict.keys():
            df=self.get_feature(feature)
        if feature not in self.featurelist:
            print("feature not supported")
            print("choose feature from", featurelist)
            return None
        df=self.get_feature(feature)
        df=df[["event_time", feature]]
        df.set_index("event_time", inplace=True)
        df.plot()
        plt.show()

    def get_sex_from_firstname(self):
        import requests
        import json
        request="https://api.genderize.io/?name=" + self.first_name
        a=requests.request("GET", request)
        gdict=a.text
        d = json.loads(gdict)
        self.gender=str(d["gender"])     

    def clear_user_attributes(self):
        for attr in self.user_attributes:
            if attr in dir(self):
                #vars(self)[attr]=None
                vars(self).pop(attr) 
        self.feature_dict={}

    def get_user_attributes(self):
        query="""
        select user_state from  user_data where user_id="""+str(self.user_id)+""";
        """
        result = self.engine.execute(query).fetchall()
        user_dict=result[0][0]
        
        try:
            first_name=user_dict["data"]["FIRST_NAME"]["value"]
            self.first_name=first_name.split()[0]
        except:
            self.first_name=None
        
        try:
            gender=user_dict["data"]["GENDER"]["value"]
        except:
            gender="not_defined"



        if gender:
            self.gender=gender.lower() 

        if self.first_name and self.gender not in ["male", "female"]:
            try:
                print("get gender from first name")
                self.get_sex_from_firstname()
            except:
                print("not successful to get gender from first_name")
                pass
    
    def get_all_user_id(self):
        if os.path.isfile("./all_user_ID.csv"):
            self.df_all_userID=pd.read_csv("./all_user_ID.csv", index_col=0)            
        else:
            query = '''
            SELECT user_id from factor;
            '''
            result = self.engine.execute(query).fetchall()
            df_uid=pd.DataFrame(result)
            filgz=df_uid["user_id"]>0
            df_uid=df_uid[filgz].drop_duplicates()
            df_uid=df_uid.reset_index()
            df_uid=df_uid.drop("index", axis=1)
            self.df_all_userID=df_uid

    def get_spectrum(self, data=None, iaxis=None, zeropad=100000, returnspec=False):
        
        if not data:
            try:
                data=self.feature_dict["migraine_days"]["migraine_days"].to_numpy()
                iaxis=self.feature_dict["migraine_days"]["iaxis"].to_numpy()
            except:
                print("migraine_days could not be loaded from feature_dict")
                return None

            
            # neglegt migraine days of type 3 and 4
        TF=(data==3)+(data==4)
        data[TF]=0
        data=np.heaviside(data,0)
        X=iaxis
        Y=data
        zeropad=zeropad-len(data)
        # zeropadding to increase resolution in frequency domain
        Yzp=np.r_[Y, np.zeros(zeropad)]
        #Yzp=Yzp*np.hanning(len(Y))
    
        # Fourier Transformation
        from numpy import fft
        freq=fft.fftfreq(len(Yzp), 1)
        spec=fft.fft(Yzp)
        # select periods 0<T<20years
        
        T=np.divide(1,freq, where=np.abs(freq)>0)
        
        gz= ((T>0)* (T<1000))
        freq=freq[gz]
        spec=spec[gz]
        self.spec_dict={}
        self.spec_dict["freq"]=freq[1:]
        self.spec_dict["spec"]=spec[1:]/len(data)
        if returnspec:
            return freq[1:], spec[1:]/len(data)

    def plotspec(self):
        try:
            isinstance(self.spec_dict,dict)
        except:
            print("spec_dict not defined")
            self.get_spectrum()
        self.get_spec_peak()
        plt.plot([1/self.freq_specmax,1/self.freq_specmax], [-0.1*self.specmax, 1.1*self.specmax], "r--")
        plt.plot(1/self.spec_dict["freq"],np.abs(self.spec_dict["spec"]))
        plt.plot([0,100], [0.85*self.specmax,0.85*self.specmax])
        plt.xlim(0,100)

    def get_spec_peak(self):
        try:
            isinstance(self.spec_dict,dict)
        except:
            print("spec_dict not defined")
            self.get_spectrum()
        freq=self.spec_dict["freq"]
        spec=self.spec_dict["spec"]

        sp_range=((freq>1/200) * (freq <1/5)) 
        spec_=np.abs(spec[sp_range])
        freq_=freq[sp_range]
        T_=1/freq_
        sm=spec_==spec_.max()
        if spec_[sm][0] > 0:
            self.freq_specmax=freq_[sm][0]
            self.T_specmax=round(1/self.freq_specmax,1)
            self.specmax=spec_[sm][0]
            self.specmax_outstanding = not np.any(np.abs(T_[np.where(spec_>self.specmax*0.85)[0]]-1/self.freq_specmax) > 0.5)
            fil=(T_>self.T_specmax+1) + (T_<self.T_specmax-1)
            self.specmax_ratio=round((spec_[fil]/self.specmax).max(),3)
        else:
            self.specmax_outstanding = False
            self.freq_specmax=np.nan
            self.T_specmax=np.nan
            self.specmax=spec_[sm][0]
            self.specmax_ratio=np.nan

if __name__ == "__main__":

    from get_features import nsl_user
    nsluser=nsl_user(user_id=10)
    nsluser.get_user_data()
    print(nsluser)

