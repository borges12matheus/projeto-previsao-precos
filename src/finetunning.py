from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from treinamento import x, y

#Separando os dados de teste e treino
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# #Fazendoo tunning do modelo Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }

# regr = RandomForestRegressor(random_state=0)
# grid_search = GridSearchCV(regr, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(x_train, y_train)

# print("Melhores Hiperparâmetros:", grid_search.best_params_)

#Treinando o modelo
regr_tunning = RandomForestRegressor(max_depth= 30, max_features= 'sqrt', min_samples_leaf= 1,
                                     min_samples_split= 2, n_estimators= 300, random_state= 0)
regr_tunning = regr_tunning.fit(x_train,y_train)

#Verificar as variáveis mais importantes do modelo
regr_tunning.feature_importances_

# Fazendo a predição do modelo otimizado por grid search cross-validation
y_predRF_cv = regr_tunning.predict(x_test)
