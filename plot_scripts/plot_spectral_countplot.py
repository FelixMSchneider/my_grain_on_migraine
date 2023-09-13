import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np
import datetime as dt


df=pd.read_csv("../CSVs/ALL_USER_DF_get_sex_from_first_name.csv")
df_per=df#df[df["specmax_outstanding"]==True]

#df_per=df_per.replace("weiblich", "female")
#df_per=df_per.replace("männlich", "male")
#df_per=df_per.replace("intersex", "male")
#df_per=df_per.replace("None", "male")



#def get_sex_from_firstname(first_name):
#    import gender_guesser.detector as gender_dec
#    d = gender_dec.Detector()
#    sex=d.get_gender(first_name.title())
#    return sex
#
#import numpy as np 
#fil=  (df_per["gender"]=="not_defined")*  (np.invert(df_per["first_name"].isna()))
#
#df_per.loc[fil,"gender"]= df_per[fil]["first_name"].apply(lambda x: get_sex_from_firstname(x))





fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212, sharex=ax1)

df_all=df_per#[fil]

sns.histplot(data=df_all[df_all["gender"]=="female"], x="T_specmax",color="C0", binwidth=1, ax=ax1 ,label="female")
sns.histplot(data=df_all[df_all["gender"]=="male"], x="T_specmax",  color="C1", binwidth=1, ax=ax2 ,label="male")

l1=ax1.legend()
l1.set_title("gender")
l2=ax2.legend()
l2.set_title("gender")
ax1.set_ylim(0,1250)
ax2.set_ylim(0,350)
ax2.set_xlim(0,60)

ax1.set_xlabel("")
ax2.set_xlabel("T$_{max}$ [Days]")
plt.tight_layout()


plt.savefig("../Figures/countplot.png")

