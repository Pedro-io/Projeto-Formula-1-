#%%
import pandas as pd 
#%%
# importando arquivos 
circuits = pd.read_csv("..\\Data\\circuits.csv")
races = pd.read_csv("..\\Data\\races.csv")
results = pd.read_csv("..\\Data\\results.csv")
drivers = pd.read_csv("..\\Data\\drivers.csv")
seasons = pd.read_csv("..\\Data\\seasons.csv")
sprints = pd.read_csv("..\\Data\\sprint_results.csv")
lap_times = pd.read_csv("..\\Data\\lap_times.csv")
# %%
#Coletando informações gerais sobre os dataframes

# %%

circuits.info()

#%%
races.info()
#
#%%
results.info()
#%%
# %%
drivers.info()
# %%
# %%
seasons.info()#não vai ser usada -- IRRELEVANTE

# %%
sprints.info()

# %%
lap_times.info()
# %%
