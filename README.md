# Spaceship Titanic — CAIXAVERSO 2025 + Kaggle

Este repositório contém o trabalho de Machine Learning 1 (CAIXAVERSO 2025) desenvolvido sobre o dataset Spaceship Titanic, que também é tema de uma competição no Kaggle. O objetivo é prever se um(a) passageiro(a) foi “Transported” (variável alvo booleana `Transported`). Aproveitei o mesmo projeto para, além de entregar o trabalho da disciplina, participar da competição.

- Competição: [Spaceship Titanic (Kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)
- Ambiente: Jupyter Notebook (Python)
- Notebook principal: `trabalho.ipynb`
- Alvo: `Transported` (True/False)
- Dados:
  - `train.csv` — usado para EDA, pré-processamento e treino/validação
  - `test.csv` — usado apenas para gerar submissões ao Kaggle (não entra no treino)

Observação: no desenvolvimento do trabalho usei apenas o `train.csv`. O `test.csv` será utilizado apenas para gerar o arquivo de submissão.

---

## Dataset

O dataset traz informações de passageiros(as) de uma nave espacial, incluindo variáveis demográficas, de cabine e de gastos a bordo. Principais colunas:

- Identificação e viagem: `PassengerId`, `HomePlanet`, `Cabin` (no formato Deck/Num/Side), `Destination`
- Perfil: `Age`, `VIP`, `CryoSleep`
- Gastos a bordo: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- `Name`
- Alvo: `Transported` (apenas em `train.csv`)

No arquivo `test.csv`, a coluna `Transported` não está presente e deverá ser prevista pelo modelo para submissão.

---

## Abordagem

O fluxo seguido no notebook `trabalho.ipynb`:

1) Exploração e carga
- Leitura do `train.csv` e inspeção inicial de colunas e tipos.

2) Limpeza e engenharia de atributos
- Cabin
  - Preenchimento de ausentes com `U/0/U`.
  - Divisão de `Cabin` em três novas colunas: `Deck`, `Num`, `Side`.
  - Conversão de `Num` para numérico e imputação com mediana.
- Preenchimento de ausentes
  - Numéricas: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `Num` => mediana.
  - Categóricas: `HomePlanet`, `CryoSleep`, `VIP`, `Deck`, `Side` => moda.
- Remoção de colunas não utilizadas
  - `PassengerId`, `Name`, `Cabin`, `Destination`.
- Encoding
  - Conversão de booleanos para tipo `bool` consistente (`CryoSleep`, `VIP`, `Transported`).
  - One-Hot Encoding forçado para: `HomePlanet`, `Deck`, `Side` (com `drop_first=True`).
  - Verificação final para garantir ausência de colunas com `dtype=object`.

3) Separação treino/validação
- `train_test_split` com `test_size=0.2` e `random_state=42`.
- Shapes:
  - Treino: 6954 amostras e 21 features
  - Validação: 1739 amostras e 21 features

4) Escalonamento
- `StandardScaler` aplicado apenas para a Regressão Logística (modelos lineares se beneficiam).
- Random Forest usa os dados não escalados.

5) Modelagem
- Modelo 1: Regressão Logística (baseline, com dados escalados)
  - `LogisticRegression(max_iter=1000, random_state=42)`
  - Acurácia: 0.7832
- Modelo 2: Random Forest (base, sem escalonamento)
  - `RandomForestClassifier(random_state=42)`
  - Acurácia: 0.7872
- Otimização de hiperparâmetros (GridSearchCV)
  - Espaço de busca reduzido para execução rápida:
    - `n_estimators: [100]`
    - `max_depth: [10, 20]`
    - `cv=2`, `scoring='accuracy'`, `n_jobs=-1`
  - Melhor configuração: `{'n_estimators': 100, 'max_depth': 10}`
  - Random Forest Otimizado — Acurácia: 0.7953

---

## Resultados

- Baseline (Regressão Logística, com StandardScaler): 0.7832 de acurácia
- Random Forest (base): 0.7872 de acurácia
- Random Forest (otimizado via GridSearchCV):
  - Melhores hiperparâmetros: `n_estimators=100`, `max_depth=10`
  - Acurácia: 0.7953

Interpretação:
- O modelo de Random Forest otimizado apresentou o melhor desempenho, atingindo cerca de 79.5% de acurácia no conjunto de validação. Para este problema com classes relativamente balanceadas, a acurácia é uma métrica adequada como primeiro critério de comparação.

---

## Conclusões do trabalho

- O pipeline de dados (imputação mediana/moda, engenharia de `Cabin` em `Deck/Num/Side`, One-Hot nas categóricas, remoção de campos textuais não preditivos) mostrou-se consistente e replicável.
- Modelos baseados em árvores (Random Forest) performaram levemente melhor que o baseline linear (Regressão Logística) sem necessidade de normalização.
- A otimização pontual de hiperparâmetros trouxe ganho adicional, indicando espaço para buscas mais amplas (ex.: `min_samples_leaf`, `max_features`, `class_weight`).
- O modelo e o fluxo são facilmente produtizáveis: desde que os dados recebam o mesmo pré-processamento, o estimador pode ser aplicado para predição e geração de submissões ao Kaggle.

Contexto acadêmico:
- Este projeto foi desenvolvido como entrega do módulo Machine Learning 1 do CAIXAVERSO 2025. Por utilizar um dataset de competição, também foi possível aproveitar o mesmo trabalho para participar do desafio no Kaggle.

---

## Como reproduzir

Pré-requisitos:
- Python 3.9+ (testado com Python 3.12.6)
- Jupyter (ou VS Code com suporte a notebooks)

Dependências principais (instale via pip):
- pandas, numpy, scikit-learn, jupyter
- (opcionais) matplotlib, seaborn, xgboost/lightgbm/catboost para experimentos futuros

Exemplo:
```bash
pip install pandas numpy scikit-learn jupyter
```

Organização de dados:
- O notebook original faz leitura em `../Datasets/train.csv`. Para facilitar, recomenda-se colocar os dados dentro do repositório e ajustar o caminho no notebook.
- Estrutura sugerida:
```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── trabalho.ipynb
└── README.md
```
- Se usar `data/train.csv`, atualize no notebook o caminho na célula de leitura (ex.: `pd.read_csv('data/train.csv')`).

Execução:
- Abra o `trabalho.ipynb` e execute as células em ordem.
- Ao final do treino/validação, você verá os resultados de acurácia e relatórios de classificação na saída do notebook.

---

## Geração de submissão para o Kaggle

O `test.csv` deve passar pelo mesmo pipeline de pré-processamento aplicado ao `train.csv`:
- Divisão de `Cabin` em `Deck/Num/Side` (com os mesmos tratamentos de ausentes).
- Imputação com a mesma lógica (mediana/moda).
- Remoção de colunas não utilizadas (`PassengerId`, `Name`, `Cabin`, `Destination`).
- One-Hot Encoding com o mesmo esquema de colunas (atenção ao `drop_first=True` e à compatibilidade de colunas entre treino e teste).
- Aplicar o modelo final (Random Forest otimizado).
- Gerar `submission.csv` com:
  - `PassengerId` (do `test.csv`, preservado em uma cópia antes de dropar colunas)
  - `Transported` (predição booleana)

Dicas:
- Guarde a lista exata de colunas de treino após o One-Hot para reordenar/alinhar as colunas do `test.csv`.
- Use `DataFrame.reindex(columns=colunas_de_treino, fill_value=0)` antes de `predict`.

---

## Limitações e próximos passos

- Validação
  - O GridSearch usou `cv=2` para ser mais ágil; valores maiores (ex.: 5 ou 10) podem dar uma estimativa mais estável do desempenho.
- Métricas
  - Explorar outras métricas (AUC, F1, Recall) e curvas (ROC/PR), especialmente se houver alguma assimetria de custos ou mudança no balanceamento entre classes.
- Modelos
  - Testar Gradient Boosting (XGBoost, LightGBM, CatBoost) e calibrar probabilidade.
- Engenharia de atributos
  - Gastos agregados (ex.: `TotalSpent`), indicadores binários (gastou/não gastou), e interações simples podem aumentar a capacidade preditiva.
- Pipeline
  - Empacotar o pré-processamento em um `sklearn.Pipeline`/`ColumnTransformer` para reuso e produção.

---

## Reprodutibilidade

- Seeds e aleatoriedade
  - `random_state=42` foi utilizado onde aplicável (train/test split, modelos).
- Versões
  - Notebook executado com Python 3.12.6; versões de bibliotecas podem afetar resultados.

---

## Licença

Defina a licença do projeto (por exemplo, MIT). Se não houver um arquivo `LICENSE`, considere adicioná-lo.

---

## Agradecimentos

- CAIXAVERSO 2025 — Módulo Machine Learning 1
- Comunidade Kaggle e autores do dataset Spaceship Titanic
- Bibliotecas open-source: pandas, numpy, scikit-learn, entre outras