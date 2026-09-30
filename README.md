# Machine learning com dados de energia renovável

## Objetivo
Este trabalho tem duas partes, e em cada uma comparei três algoritmos:
1. Classificação: prever se um empreendimento é Solar, Eólico ou Hidráulico usando a potência e a localização.
2. Regressão: prever a radiação solar de cada hora em Petrolina (PE) usando temperatura, umidade, nuvens, vento e hora do dia.

## Dados
| Tarefa | Fonte | Período / escopo | Arquivo |
|---|---|---|---|
| 1 | SIGA — ANEEL (API CKAN, sem token) | Cadastro de empreendimentos, até 1200 linhas por sigla (UFV, EOL, UHE, PCH, CGH) | `aneel_classificacao_orange.csv` |
| 2 | Open-Meteo, API histórica (sem token) | 01/04/2025 a 30/06/2025, das 7h às 17h, fuso America/Recife, lat −9,39 e lon −40,50 | `meteo_regressao_orange.csv` |

Limitações dos dados: a consulta da ANEEL traz no máximo 1200 linhas por sigla, então não é uma amostra aleatória (a Hidráulica veio quase completa). O cadastro mistura empreendimentos em fases diferentes, tem coordenadas iguais a zero e muitas usinas Solar de cerca de 1 kW, e a potência outorgada não é energia gerada. Os dados da Open-Meteo vêm de modelo (reanálise) e não de um painel fotovoltaico.

## Como executar
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook Aula_APIs_Energia_Renovavel.ipyn   # Kernel > Restart & Run All
```
As primeiras células consultam as APIs e geram os dois CSVs de novo. Não precisa de token.

## Metodologia
- Tarefa 1: split 80/20 estratificado com `random_state=42`. Modelos: KNN (com `StandardScaler` dentro de um Pipeline), Árvore de Decisão e Random Forest. Usei a média macro nas métricas e mostrei o F1 weighted como apoio.
- Tarefa 2: split temporal, com as primeiras 80% das horas para treino e as últimas 20% para teste, sem embaralhar. Modelos: Regressão Linear (com `StandardScaler`), Random Forest e Gradient Boosting. Métricas: MAE, MSE e R².

## Resultados

### Tarefa 1: classificação (split 80/20 estratificado, seed 42, 776 exemplos de teste, média macro)
| Algoritmo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---|---|---|---|---|
| KNN (k=5) | 0,965 | 0,966 | 0,964 | 0,965 | 0,965 |
| Árvore de Decisão | 0,937 | 0,937 | 0,933 | 0,934 | 0,936 |
| Random Forest | 0,976 | 0,977 | 0,974 | 0,975 | 0,975 |

Baseline (sempre a classe maior): accuracy 0,381 e F1 macro 0,184.

Análise extra (seção 1.5 do notebook), sem Solar de até 1 kW e sem coordenadas zero (3.075 das 3.876 linhas, com 446 de Solar):

| Algoritmo | F1 macro (original) | F1 macro (filtrada) | Accuracy (filtrada) |
|---|---|---|---|
| KNN (k=5) | 0,965 | 0,939 | 0,958 |
| Árvore de Decisão | 0,934 | 0,903 | 0,937 |
| Random Forest | 0,975 | 0,955 | 0,967 |

### Tarefa 2: regressão (split temporal 80/20 sem embaralhar, 800 horas de treino e 201 de teste)
| Algoritmo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Regressão Linear | 145,205 | 30034,201 | 0,360 |
| Random Forest | 66,399 | 7210,084 | 0,846 |
| Gradient Boosting | 67,082 | 7444,700 | 0,841 |

Baseline (média do treino): MAE 209,0 e R² −0,330. A radiação média cai de 497,8 W/m² no treino para 373,4 no teste, e o período de teste tem mais nuvens. Random Forest sem a coluna `hora`: MAE 131,4 e R² 0,337.

## Conclusões
Tarefa 1: o Random Forest foi o melhor (acurácia 0,976 e F1 macro 0,975, contra 0,381 do baseline), mas a diferença para o KNN (0,965) é pequena e pode ser acaso, porque usei só um split. A Solar foi a classe mais confundida: os 159 exemplos de Solar de até 1 kW do teste foram todos acertados, e os 12 erros do Random Forest ficaram entre as 81 Solar acima de 1 kW (7 classificadas como Hidráulica e 5 como Eólica). A amostra tem problemas (62,8% das linhas Solar com até 1 kW e 47 linhas com coordenadas zero), mas tirar esses casos diminuiu o F1 macro em só 2 a 3 pontos e a ordem dos modelos não mudou. As coordenadas repetidas da Solar (46%) merecem atenção, mas só 28 linhas são duplicatas exatas.

Tarefa 2: o Random Forest (MAE 66,4 W/m² e R² 0,846) e o Gradient Boosting (67,1 e 0,841) ficaram quase iguais e foram bem melhores que a Regressão Linear (145,2 e 0,360), que não segue a forma de sino da radiação. A hora foi a variável mais importante: sem ela o R² do Random Forest cai de 0,846 para 0,337. Estimar radiação não é o mesmo que prever geração de energia, porque a geração depende da eficiência do painel e do inversor, da temperatura, da inclinação, de sombra e de perdas, e W/m² é potência em um instante e não energia (kWh).
