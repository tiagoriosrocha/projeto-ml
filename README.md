# Análise e Classificação de Contratos Inteligentes com Foco em Recall

Este projeto tem como objetivo aplicar, comparar e combinar diversas técnicas de Machine Learning para a classificação de contratos inteligentes. As features são extraídas por análise estática do código fonte. Um foco particular é a maximização da métrica de **recall** para a classe de interesse, visando identificar corretamente o maior número possível de contratos relevantes dessa classe.

## Dados Fonte

Os dados utilizados são provenientes de um dataset de contratos inteligentes, onde as features representam métricas quantitativas obtidas por análise estática.

*   **Arquivo de Dados:** `smart-contract-dataset.csv`
*   **Descrição Detalhada das Features:** Consulte o arquivo `Unveiling vulnerable smart contracts: Toward profiling vulnerable smart
contracts using genetic algorithm and generating benchmark dataset.pdf`  na pasta do projeto para uma descrição completa das features e da origem do dataset.

## Estrutura do Projeto

Foram sendo criados Notebooks a medida que os estudos foram avançando, ou seja, é possível compreender a sequência de testes e análise que foram realizados ao longo do desenvolvimento deste projeto. O projeto está organizado nos seguintes arquivos principais:

*   `smart-contract-dataset.csv`: O dataset principal.
*   `Unveiling vulnerable smart contracts: Toward profiling vulnerable smart
contracts using genetic algorithm and generating benchmark dataset.pdf`: Documentação de referência sobre os dados.  
*   `1_testes_gerais.ipynb`: Notebook inicial para ter uma visão geral dos dados e testes introdutórios.
*   `2_svm_reducao_pca.ipynb`: Notebook com o uso de SVM.
*   `3_svm_reducao_giny.ipynb`: Notebook com uso de SVM com redução de dimensionalidade usando Gini.
*   `4_decisiontree_randomforest_pca_pipeline.ipynb`: Notebook com árvores de decisão e random forest.
*   `5_decisiontree_randomforest_somente_15.ipynb`: Notebook com árvores de decisão e random forest para apenas 15 feactures segundo Gini.
*   `6_Pipelines_comentado.ipynb`: Notebook dedicado à construção, treinamento e avaliação de **pipelines de classificadores individuais**.
*   `7_Ensembles_comentado.ipynb`: Notebook focado na implementação e avaliação de **modelos de ensemble**, utilizando os melhores classificadores individuais como base.

## Metodologia Geral Aplicada

Ambos os notebooks seguem um fluxo de trabalho que inclui:

1.  **Carregamento e Preparação dos Dados:**
    *   Leitura do dataset `smart-contract-dataset.csv`.
    *   Renomeação de colunas para facilitar o manuseio.
    *   Separação das features (X) e da variável alvo (y).
    *   Divisão estratificada dos dados para garantir a representatividade das classes nos diferentes conjuntos:
        *   `X_train_val`, `y_train_val`: Conjunto combinado para treinamento e validação cruzada dos modelos finais.
        *   `X_test`, `y_test`: Conjunto de teste, reservado para a avaliação final e imparcial dos modelos.
        *   `X_train`, `y_train` e `X_val`, `y_val`: Subconjuntos de `X_train_val`, usados no `6_Pipelines_comentado.ipynb` para uma avaliação preliminar e mais rápida dos modelos individuais.

## Detalhamento dos Notebooks

### 1. `6_Pipelines_comentado.ipynb`: Avaliação de Classificadores Individuais

Este notebook foca na implementação e avaliação de uma variedade de algoritmos de classificação de forma individual. O objetivo é entender o desempenho base de cada modelo e identificar candidatos promissores para ensembles.

*   **Modelos Avaliados:**
    *   Naive Bayes Gaussiano
    *   Árvores de Decisão (`DecisionTreeClassifier`)
    *   Regressão Logística (`LogisticRegression`)
    *   Support Vector Classifier Linear (`LinearSVC`)
    *   Support Vector Classifier com kernel RBF (`SVC`)
    *   Random Forest (`RandomForestClassifier`)
    *   AdaBoost (`AdaBoostClassifier`)
    *   XGBoost (`XGBClassifier`)
    *   LightGBM (`LGBMClassifier`)
    *   CatBoost (`CatBoostClassifier`)
*   **Fluxo de Trabalho Específico:**
    *   Para cada classificador, um pipeline de pré-processamento customizado é definido, incorporando os transformadores e técnicas de seleção de features descritos acima.
    *   Os hiperparâmetros para cada modelo são definidos (geralmente valores padrão ou heurísticos neste estágio).
    *   **Treinamento e Avaliação Preliminar:** Os modelos são treinados no subconjunto `X_train` e avaliados no subconjunto `X_val`.
    *   **Métricas de Avaliação:** Relatório de Classificação (incluindo precisão, recall, F1-score para ambas as classes) e Matriz de Confusão são gerados para cada modelo.

### 2. `7_Ensembles_comentado.ipynb`: Construção e Avaliação de Modelos Ensemble

Este notebook utiliza os insights e os pipelines dos melhores modelos individuais (especificamente Random Forest, XGBoost, LightGBM e CatBoost) para construir e avaliar modelos de ensemble, com o objetivo de melhorar a performance e a robustez da classificação.

*   **Modelos Base para Ensemble:**
    *   `RandomForestClassifier`
    *   `XGBClassifier`
    *   `LGBMClassifier`
    *   `CatBoostClassifier`
    *   (Os pipelines de pré-processamento para estes modelos são redefinidos/reutilizados neste notebook).
*   **Técnicas de Ensemble Implementadas:**
    *   **`VotingClassifier`:**
        *   Combina as predições dos quatro modelos base.
        *   Utiliza `voting='soft'`, que se baseia na média das probabilidades preditas pelos classificadores base (requer que os modelos base implementem `predict_proba`).
    *   **`StackingClassifier`:**
        *   Utiliza os quatro modelos base no primeiro nível.
        *   Emprega uma `LogisticRegression` como meta-estimador (nível final) para aprender a combinar as predições dos modelos base.
        *   A validação cruzada interna (`cv=5`) é usada para gerar as predições "out-of-fold" dos modelos base, que servem de entrada para o meta-estimador.
*   **Fluxo de Trabalho Específico:**
    *   **Validação Cruzada Robusta:** Ambos os ensembles (`VotingClassifier` e `StackingClassifier`) são avaliados usando `cross_validate` com `StratifiedKFold(n_splits=5)` no conjunto completo `X_train_val`.
        *   Métricas Coletadas na CV: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision` (PR AUC).
        *   Uma função auxiliar `print_cv_results` é usada para exibir a média e o desvio padrão dessas métricas.
    *   **Treinamento Final e Avaliação no Conjunto de Teste:**
        *   Após a validação cruzada, os ensembles são treinados em todo o conjunto `X_train_val`.
        *   O desempenho final e imparcial é medido no conjunto de teste `X_test`.
        *   Relatórios de Classificação e Matrizes de Confusão são gerados para o conjunto de teste.