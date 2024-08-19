#%%
import pandas as pd 

#%%
# importando arquivos 
circuits = pd.read_csv("..\\Data\\circuits.csv")
# %%
circuits.head()
# %%
circuits.isna().sum()
# %%
