import pandas as pd
import seaborn as sns
import pylab as plt
import numpy as np
import datetime as dt


df=pd.read_csv("../CSVs/ALL_USER_DF.csv")
df_per=df#df[df["specmax_outstanding"]==True]

df_per=df_per.replace("weiblich", "female")
df_per=df_per.replace("männlich", "male")
df_per=df_per.replace("intersex", "male")
df_per=df_per.replace("None", "male")



df_per1=df_per[df_per["gender"]=="male"]
df_per2=df_per[df_per["gender"]=="female"]

df_per=pd.concat([df_per2,df_per1])

fil=(df_per["gender"] == "male").to_numpy() + ( df_per["gender"] == "female" ).to_numpy()

df_per["specmax_ratio_2"]=df_per["specmax_ratio"].apply(lambda x: 1.0/x)

sns.scatterplot(data=df_per[fil], x="T_specmax", y="specmax_ratio_2" , hue="gender", alpha=0.6)
plt.xlabel("T [Days]")
plt.ylabel("Specmax / Spec")
plt.xlim(0,60);
plt.ylim(0.9,3);

plt.savefig("../Figures/spectral_classification_DB.png")
