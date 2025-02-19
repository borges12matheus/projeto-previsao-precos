import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from treinamento import x,y

#Vamos avaliar os modelos de forma comparativa
#Dividindo os dados de treino e teste, usando a variável target original sem transformação
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

modelos = {
    'Regressão Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Árvore de Decisão': tree.DecisionTreeRegressor(),
    'Random Forest Tunned': RandomForestRegressor()  
}

resultados = []
#Realizar o treinamento dos modelos e calcular as métricas, salvando na lista de resiu
for name, modelo in modelos.items():
    modelo = modelo.fit(x_train, y_train)
    y_pred_modelo = modelo.predict(x_test)

    mae = mean_absolute_error(y_test,  y_pred_modelo)
    mape = mean_absolute_percentage_error(y_test,  y_pred_modelo)
    rmse = root_mean_squared_error(y_test,  y_pred_modelo)
    r2 = modelo.score(x_test, y_test)

    resultados.append([name, mae, mape, rmse, r2])

#Cria uma tabela para armazenar as métricas dos modelos
df_resultados = pd.DataFrame(resultados, columns=['Modelo', 'MAE', 'MAPE','RMSE', 'R²'])
df_resultados.sort_values(by="RMSE", ascending=True, inplace=True)  # Ordenar pelo melhor RMSE

print(df_resultados)
