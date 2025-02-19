#Importando os dados do dataset
import pandas as pd
dados = pd.read_csv(r'D:\Data Science\Indicium\data\teste_indicium_precificacao.csv')

#Excluindo toda linha vazia que exista no database
dados = dados.dropna(how= 'all')

#Exluindo a coluna 'ultima review' pois não gera impacto significativo para a análise
dados = dados.drop('ultima_review', axis= 1)

#Substituindo os valores nulos por 0 da coluna numero_de_reviews
dados.numero_de_reviews = dados.numero_de_reviews.fillna(0)

#Substituindo os valores nulos por 0 da coluna reviews_por_mes
dados.reviews_por_mes = dados.reviews_por_mes.fillna(0)

#Excluindo a coluna 'nome' e bairro, pois não geram impacto para nossa análise. O bairro já está representado por bairro_group
dados = dados.drop(['nome', 'bairro', 'id','host_id','host_name'],axis= 1)

#Calculando os quartis
Q1 = dados.price.quantile(0.25)
Q2 = dados.price.quantile(0.50)
Q3 = dados.price.quantile(0.75)
IQR = Q3 - Q1
print('Q1:', Q1)
print('Q2:', Q2)
print('Q3:', Q3)
print('IQR:', IQR)

#Calculando os limites inferior e superior
limite_inferior = Q1 - 1.5*IQR
limite_superior = Q3 + 1.5*IQR
print('Limite Inferior:', limite_inferior)
print('Limite Superior:', limite_superior)

mediana_por_bairro = dados.groupby('bairro_group').price.median().to_dict()
print(mediana_por_bairro)

#Substituindo os valores iguais a 0 pela mediana de cada bairro
dados.loc[dados.price <= 0,'price'] = dados.bairro_group.map(mediana_por_bairro)

#Quantidade de valores acima do limite superior calculado
outliers = dados.loc[dados.price > limite_superior]

#Separando os dados dos anúncios de luxo
dados = dados.loc[dados.price <= limite_superior]

#Vamos criar a variável preço por noite que será a divisão do preço pelo minimo de noites exigidas pelo anunciante
dados['preco_por_noite'] = dados.price/dados.minimo_noites

#Transformando as variáveis categóricas em numéricas
dados_encoded = pd.get_dummies(dados, columns= ['bairro_group','room_type'],dtype= int)
print(dados_encoded)

