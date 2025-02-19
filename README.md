# **🏡 Projeto de Previsão de Preços de Imóveis para Aluguel**  
📌 *Análise, Modelagem e Predição de Preços de Imóveis utilizando Machine Learning*  

---

## **📌 Descrição do Projeto**  
Este projeto tem como objetivo a criação de um modelo de Machine Learning capaz de prever o preço de imóveis para aluguel com base em características como **localização, tipo de acomodação, número de reviews e disponibilidade no ano**.  

A abordagem utilizada inclui **análise exploratória dos dados (EDA), engenharia de features, treinamento de modelos de regressão e avaliação do desempenho preditivo**.  

---

## **📊 Conjunto de Dados**  
O dataset contém informações detalhadas sobre imóveis disponíveis para aluguel, incluindo:  
- 📍 **Localização** (latitude, longitude, bairro)  
- 🏠 **Tipo de imóvel** (quarto privado, casa/apartamento inteiro, entre outros)  
- 🔢 **Quantidade de noites mínimas**  
- ⭐ **Número de avaliações e avaliações por mês**  
- 📆 **Disponibilidade anual**  
- 💲 **Preço do aluguel** *(variável alvo para previsão)*  

---

## **📂 Estrutura do Projeto**  

```bash
📂 projeto-previsao-precos
│── 📂 data                  # Conjunto de dados (CSV, JSON, etc.)
│── 📂 notebooks             # Jupyter notebooks para análise exploratória
│── 📂 src                   # Código-fonte do projeto
│   ├── preprocessamento.py  # Funções de limpeza e transformação dos dados
│   ├── treinamento.py       # Treinamento dos modelos
│   ├── finetunning.py       # Otimização do modelo
│   ├── avaliacao.py         # Avaliação dos modelos
│   ├── predicao.py          # Código para novas previsões
│── 📂 models                # Modelo treinado e salvo
│── requirements.txt         # Lista de bibliotecas necessárias
│── README.md                # Documentação do projeto
```

---

## **📌 Tecnologias Utilizadas**  
- 📜 **Linguagem:** Python 3.9+  
- 📚 **Bibliotecas:**  
  - `pandas` → Manipulação de dados  
  - `numpy` → Cálculos numéricos  
  - `scikit-learn` → Algoritmos de machine learning  
  - `matplotlib` / `seaborn` → Visualização de dados  
  - `joblib` → Salvamento e carregamento do modelo  
  - `RandomForestRegressor` / `DecisionTreeRegressor` → Modelos avançados de regressão  
- ⚡ **Ferramentas Adicionais:**  
  - Jupyter Notebook  
  - Google Colab (opcional)  
  - Streamlit / Flask (para API de previsão)  

---

## **🔍 Etapas do Projeto**
### **1️⃣ Análise Exploratória dos Dados (EDA)**
✔ Tratamento de valores nulos e outliers  
✔ Transformação de variáveis categóricas  
✔ Análise de correlação entre variáveis  
✔ Criação de novas features  

### **2️⃣ Pré-processamento dos Dados**
✔ Normalização e escalonamento de variáveis  
✔ Transformação de variáveis categóricas em numéricas  
✔ Engenharia de features (ex: `preco_por_noite`)  

### **3️⃣ Treinamento dos Modelos**
✔ Modelos testados:  
   - **Regressão Linear**  
   - **Árvore de Decisão**  
   - **Random Forest**  
   - **Random Forest Tunado**  

✔ Técnicas utilizadas:  
   - **Divisão treino/teste** (80%/20%)  
   - **Cross-validation** para otimização de hiperparâmetros  

### **4️⃣ Avaliação dos Modelos**
✔ **Métricas utilizadas:**  
   - **MAE (Erro Médio Absoluto)**  
   - **MAPE (Erro Percentual Médio Absoluto)**  
   - **RMSE (Raiz do Erro Quadrático Médio)**  
   - **R² (Coeficiente de Determinação)**  

📊 **Resultados das Métricas:**  
|          Modelo            |    MAE    |   MAPE   |    RMSE   |     R²   |
|----------------------------|-----------|----------|-----------|----------|
|  **Random Forest Tunned**  | 0.761683  | 0.008853 | 4.147637  | 0.996287 |
|  **Random Forest**         | 0.772605  | 0.008837 | 4.265349  | 0.996073 |
|  **Árvore de Decisão**     | 1.120523  | 0.011729 | 6.563222  | 0.990703 |
|  **Regressão Linear**      | 30.643937 | 0.294602 | 41.661007 | 0.625390 |

📌 **Conclusão:** O modelo **Random Forest com tunning** apresentou os **melhores resultados**, com **baixo erro e alto R²**, sendo escolhido como modelo final.

---

## **🚀 Como Executar o Projeto**
### **1️⃣ Clonar o Repositório**
```bash
git clone https://github.com/seu-usuario/projeto-previsao-precos.git
cd projeto-previsao-precos
```

### **2️⃣ Criar um Ambiente Virtual (opcional)**
```bash
python -m venv venv
source venv/bin/activate  # Para Linux/macOS
venv\Scripts\activate     # Para Windows
```

### **3️⃣ Instalar Dependências**
```bash
pip install -r requirements.txt
```

### **4️⃣ Rodar os Notebooks**
Abra o Jupyter Notebook e explore os arquivos na pasta `notebooks/`.  
```bash
jupyter notebook
```

### **5️⃣ Fazer Previsão com Novo Dado**
Após treinar o modelo, é possível carregar um novo dado e fazer previsões:

```python
import joblib
import pandas as pd

# 🔹 Carregar modelo salvo
modelo = joblib.load('models/modelo_random_forest.pkl')

# 🔹 Criar um novo exemplo de imóvel para prever o preço
novo_dado = pd.DataFrame([{
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type_Entire home/apt': 1,
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355
}])

# 🔹 Fazer previsão
previsao = modelo.predict(novo_dado)
print(f"Preço previsto: ${previsao[0]:.2f}")
```

---

## **📌 Próximos Passos**
✅ Melhorar o **tuning de hiperparâmetros** para o Random Forest  
✅ Implementar um **modelo com XGBoost** para comparação  
✅ Criar uma **API com Flask ou FastAPI** para expor o modelo  
✅ Construir um **dashboard interativo com Streamlit**  

---

## **📜 Licença**
Este projeto é de código aberto sob a licença **MIT**.

---

## **💬 Contato**
📧 **Email:** borgesmatheus1201@email.com  
🐍 **GitHub:** [borges12matheus](https://github.com/borges12matheus)  
🔗 **LinkedIn:** [matheusborges12](https://www.linkedin.com/in/matheusborges12/)  
