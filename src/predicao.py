import pandas as pd
import pickle

with open(r'D:\Data Science\Indicium\models\LH_CD_MATHEUS-BORGES.pkl', 'rb') as arquivo:
    regr = pickle.load(arquivo)
    
novo_dado = {
    'id': 2595,
    'nome': 'Skylit Midtown Castle',
    'host_id': 2845,
    'host_name': 'Jennifer',
    'bairro_group': 'Manhattan',
    'bairro': 'Midtown',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type': 'Entire home/apt',
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'ultima_review': '2019-05-21',
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355
}

# Criar DataFrame
df_novo = pd.DataFrame([novo_dado])
df_novo.head()

#Criar coluna nome_caracteres
df_novo['nome_caracteres'] = df_novo.nome.str.len()

# Remover colunas irrelevantes (ajuste conforme necessário)
df_novo = df_novo.drop(['id', 'host_id', 'host_name', 'nome', 'bairro', 'ultima_review'], axis= 1)

# Criar nova feature: preco_por_noite (se aplicável)
df_novo['preco_por_noite'] = df_novo.minimo_noites

# One-hot encoding para variáveis categóricas
df_novo = pd.get_dummies(df_novo, columns=['bairro_group', 'room_type'],dtype= int)

# Ajustar colunas para que tenham a mesma estrutura do conjunto de treinamento
colunas_modelo = ['latitude', 'longitude', 'minimo_noites', 'numero_de_reviews',
       'reviews_por_mes', 'calculado_host_listings_count',
       'disponibilidade_365', 'nome_caracteres', 'preco_por_noite',
       'bairro_group_Bronx', 'bairro_group_Brooklyn', 'bairro_group_Manhattan',
       'bairro_group_Queens', 'bairro_group_Staten Island',
       'room_type_Entire home/apt', 'room_type_Private room',
       'room_type_Shared room'] 
for col in colunas_modelo:
    if col not in df_novo.columns:
        df_novo[col] = 0  # Adiciona colunas ausentes com valor 0

# Reordenar colunas na mesma ordem do treinamento
df_novo = df_novo[colunas_modelo]

# Exibir DataFrame transformado
df_novo.head()

# Aplicando os novos dados para o modelo
nova_pred = regr.predict(df_novo)
print('A sugestão de preço em dolars para este imóvel é: U$', nova_pred[0])