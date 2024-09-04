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


#%%
import os
import pandas as pd

# Define o caminho para a pasta onde os arquivos CSV estão localizados
folder_path = "..\\Data\\"

# Lista todos os arquivos no diretório especificado
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop através de cada arquivo CSV na pasta
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    
    # Lê o arquivo CSV em um DataFrame
    df = pd.read_csv(file_path)
    
    # Exibe o nome do arquivo e as primeiras linhas do DataFrame
    print(f"Informações sobre o arquivo: {file_name}")
    df.info()  # Exibe as primeiras 5 linhas do DataFrame
    print("\n")  # Adiciona uma linha em branco para separar as saídas

# %%
