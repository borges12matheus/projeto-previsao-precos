#Importando as bibliotecas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessamento import dados_encoded
import pickle


#Realizando a separação dos dados
dados_model = dados_encoded
x = dados_model.drop('price',axis=1)
y = dados_model.price

#Verificando o desbalanceamento da variável target  Y
print(f"Skewness: {skew(y):.2f}")
print(f"Kurtosis: {kurtosis(y):.2f}")

#Visualizando graficamente
sns.histplot(y, kde=True)
plt.title("Distribuição da Variável Alvo (y)")
plt.show()

#Transformando a variável target para uma base logarítmica
y_transformed = np.log(y)

sns.histplot(y_transformed, kde=True)
plt.title("Distribuição da Variável Alvo (y)")
plt.show()

#Separando os dados de teste e treino
x_train, x_test, y_train, y_test = train_test_split(x,y_transformed, test_size=0.2, random_state=42)

# Treinando o nosso modelo
reg = LinearRegression().fit(x_train, y_train)

# Fazendo a previsão com o modelo anterior
y_pred = reg.predict(x_test)

# Transformando de volta a variável y
y_pred = np.exp(y_pred)

#Definindo os dados de teste e treino
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Criando o regressor e treinando o modelo
regTree = tree.DecisionTreeRegressor(random_state=10, max_depth=10)
regTree = regTree.fit(x_train, y_train)

# Fazendo a previsão
y_predTree = regTree.predict(x_test)

#Dividindo os dados de treino e teste, usando a variável target original sem transformação
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Treinando o modelo
regr = RandomForestRegressor(max_depth= 50, random_state= 0)
regr = regr.fit(x_train,y_train)

#Verificar as variáveis mais importantes do modelo
feature_importance_ = regr.feature_importances_
print(feature_importance_)

#Realizando a predição
y_predRF = regr.predict(x_test)

#Comparando os modelos já treinados
fig, ax = plt.subplots(figsize= (8,8))
ax.scatter(y_test, y_pred, alpha=0.8, label= 'Regress')
ax.scatter(y_test, y_predTree, alpha=0.8, label= 'Árvore')
ax.scatter(y_test, y_predRF, alpha=0.8, label= 'Random Forest')
ax.plot([0,500],[0,500])
ax.legend()

plt.show()

#Salvando modelo para produção
with open('LH_CD_MATHEUS-BORGES.pkl', 'wb') as arquivo:
    pickle.dump(regr, arquivo)