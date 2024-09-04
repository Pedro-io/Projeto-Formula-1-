#%%
import pandas as pd 

#%%
results = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Projeto-Formula-1-\Data\results.csv")

results.info()
# %%
results.isna().sum()
# %%
results['rank'].value_counts()
# %%
results.head()
# %%
races = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Projeto-Formula-1-\Data\races.csv")

races.info()
# %%
races_test = races[races['year'] >= 2022]

races_test.head(30)
# %%

results_filter = results[results['raceId']>= 1074]

# %%
results_filter
# %%
results_filter['rank'].value_counts()
# %%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import tree




#%%
results_filter.info()
# %%
df = results_filter

# Converter colunas que são objetos mas representam números para numéricas
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['milliseconds'] = pd.to_numeric(df['milliseconds'], errors='coerce')
df['fastestLap'] = pd.to_numeric(df['fastestLap'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
df['fastestLapSpeed'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')

# Remover colunas não numéricas ou irrelevantes
df = df.drop(columns=['number', 'positionText', 'time', 'fastestLapTime'])

# Remover linhas com valores NaN
df = df.dropna()

# Separar as variáveis dependentes (X) e independentes (y)
X = df.drop(columns=['position'])
y = df['position']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
clf = DecisionTreeClassifier()

# Treinar o modelo
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)

# Avaliar a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# %%
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, rounded=True)
plt.show()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Gerar a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Visualizar a matriz de confusão usando seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusão')
plt.show()

# %%
import joblib

# Salvar o modelo treinado
joblib.dump(clf, 'modelo_arvore_decisao.pkl')

# %%
# Carregar o modelo treinado
clf = joblib.load('modelo_arvore_decisao.pkl')

# %%
# Filtrar o DataFrame results onde raceId está entre 900 e 1073
nova_df = results[(results['raceId'] >= 900) & (results['raceId'] <= 1073)]

# Converter colunas que são objetos mas representam números para numéricas
nova_df['position'] = pd.to_numeric(nova_df['position'], errors='coerce')
nova_df['milliseconds'] = pd.to_numeric(nova_df['milliseconds'], errors='coerce')
nova_df['fastestLap'] = pd.to_numeric(nova_df['fastestLap'], errors='coerce')
nova_df['rank'] = pd.to_numeric(nova_df['rank'], errors='coerce')
nova_df['fastestLapSpeed'] = pd.to_numeric(nova_df['fastestLapSpeed'], errors='coerce')

# Remover colunas não numéricas ou irrelevantes
nova_df = nova_df.drop(columns=['number', 'positionText', 'time', 'fastestLapTime'])

# Remover linhas com valores NaN
nova_df = nova_df.dropna()

# Separar as variáveis dependentes (X) e a variável alvo (y)
X_novo = nova_df.drop(columns=['position'])
y_novo = nova_df['position']

# %%

# Fazer previsões com a nova base de dados
y_novo_pred = clf.predict(X_novo)



# %%
from sklearn.metrics import accuracy_score

# Calcular a nova acurácia
nova_acuracia = accuracy_score(y_novo, y_novo_pred)
print(f'Nova acurácia: {nova_acuracia:.2f}')


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Gerar a matriz de confusão
nova_cm = confusion_matrix(y_novo, y_novo_pred)

# Visualizar a matriz de confusão usando seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(nova_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Nova Matriz de Confusão')
plt.show()

# %%
