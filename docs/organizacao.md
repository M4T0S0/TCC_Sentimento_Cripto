# Organização do repositório

Este arquivo explica como as pastas do repositório foram organizadas e por quê.

## Visão geral

```
tcc-sentimento-cripto/
├── README.md            apresentação do trabalho e das etapas
├── requirements.txt     bibliotecas usadas
├── .gitignore           o que fica fora do Git
├── docs/
│   ├── decisoes.md      decisões, números oficiais e justificativas de cada etapa
│   └── organizacao.md   este arquivo
├── src/tcc/             código comum a todas as etapas
├── notebooks/           um notebook por etapa
├── results/             saídas oficiais de cada notebook
└── data/                preços e textos (só no computador, fora do Git)
```

A ideia é separar três coisas: o código reutilizável (`src/tcc/`), a execução de cada experimento (`notebooks/`) e o que cada experimento produziu (`results/`). Os notebooks ficam curtos, porque a lógica fica nos módulos, e qualquer resultado pode ser conferido sem rodar nada de novo.

## Etapas e nomes

Cada etapa tem um número e um nome curto. O notebook e a pasta de resultados usam o mesmo nome, então `notebooks/05_com_sentimento.ipynb` grava em `results/05_com_sentimento/`.

| Etapa | Notebook e pasta | O que faz |
|---|---|---|
| 01 | `01_alvo` | Compara prever o preço com prever o retorno |
| 02 | `02_lstm` | Busca dos hiperparâmetros da LSTM com Optuna |
| 03 | `03_transformer` | Busca dos hiperparâmetros do Transformer com Optuna |
| 04 | `04_sentimento` | Limpeza dos textos e sentimento diário com o CryptoBERT |
| 05 | `05_com_sentimento` | LSTM e Transformer com e sem sentimento |
| 06 | `06_controle` | Controle com sentimento sintético |
| 07 | `07_walk_forward` | Repetição da comparação em três regimes de mercado |
| 08 | `08_tabelas_figuras` | Tabelas e figuras do texto do TCC |
| 09 | `09_segmentos` | Sentimento separado por veículo, assunto, força e engajamento |
| 10 | `10_validacao_manual` | Comparação do CryptoBERT com rótulos manuais |

O período (2022-01-01 a 2023-12-31) e o alvo não aparecem no nome das pastas. Eles ficam no `config.json` de cada pasta de resultados, junto com os demais parâmetros da rodada.

As etapas dependem umas das outras na ordem dos números. A 03 lê as previsões da 02, a 05 lê os hiperparâmetros da 02 e da 03 e o sentimento da 04, a 06 e a 07 reaproveitam a 05, a 08 só lê os resultados de 01 a 07, a 09 usa as notas da 04 e a versão financeira da 05, e a 10 usa as notas da 04.

## Código (`src/tcc/`)

| Módulo | Conteúdo |
|---|---|
| `config.py` | Moedas, sementes, divisão 70/15/15, hiperparâmetros padrão e caminhos |
| `dados.py` | Download dos preços do Yahoo Finance e montagem das entradas |
| `modelo.py` | LSTM, Transformer e treino com parada antecipada |
| `avaliacao.py` | MAPE, RMSE, acerto de direção e teste de Diebold-Mariano |
| `experimento.py` | Treino, teste, resumo por moeda e comparação entre versões |
| `otimizacao.py` | Busca de hiperparâmetros com Optuna, sem usar o período de teste |
| `textos.py` | Limpeza das notícias e dos tweets e separação por moeda |
| `sentimento.py` | CryptoBERT, média diária e preenchimento dos dias sem texto |
| `controles.py` | Sinal sintético usado no controle da etapa 06 |
| `walkforward.py` | Retreino em blocos de meses e rótulo de regime |
| `segmentos.py` | Separação do sentimento por tipo de texto (etapa 09) |
| `validacao.py` | Amostra, planilha e concordância da validação manual (etapa 10) |

Todos os notebooks começam da mesma forma: encontram a pasta do projeto, no computador ou no dataset do Kaggle, e importam `tcc`. No Kaggle o dataset é só de leitura, então as saídas vão para `/kaggle/working/results/` e depois são copiadas para `results/`.

## Resultados (`results/`)

Os mesmos nomes de arquivo se repetem entre as etapas, para facilitar a leitura:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | Parâmetros da rodada (período, moedas, sementes, hiperparâmetros) |
| `previsoes.csv` | Uma linha por moeda, modelo, versão, semente e dia de teste |
| `resumo.csv` | MAPE, desvio entre sementes, RMSE, direção e p contra o naive |
| `mape.csv` | Tabela de MAPE por moeda, pronta para leitura |
| `diebold_mariano.csv` | Comparações de Diebold-Mariano da etapa |
| `hiperparametros.json` | Melhor conjunto encontrado pelo Optuna (etapas 02 e 03) |
| `trials.csv` | Todos os trials do Optuna (etapas 02 e 03) |
| `*.png` | Figura de conferência da etapa |

A etapa 04 tem duas subpastas com o sentimento diário de cada moeda (`BTC.csv`, `ETH.csv`...), uma para cada forma de preencher os dias sem texto:
- `ffill/`: repete o último valor da própria moeda.
- `copia_btc/`: usa o valor do BTC no mesmo dia e, se ele também faltar, repete o último valor.

A etapa 08 separa `tabelas/` (t1 a t7, em CSV) e `figuras/` (f1 a f6, em PNG), e junta todas as tabelas em `tabelas.xlsx`, uma por aba. São esses os arquivos usados no texto do TCC.

Todos os números de `results/` vêm da rodada oficial no Kaggle. Quando uma etapa é refeita, a pasta inteira é substituída.

## O que fica fora do Git

| Item | Motivo |
|---|---|
| `data/textos/` | Notícias (`cryptonews.csv`) e tweets brutos, publicados por terceiros |
| `data/validacao/` | Planilha da validação manual, que contém os textos |
| `results/04_sentimento/nota_por_texto.csv` | Contém o texto original de cada notícia e tweet |
| `data/precos/` | Baixado do Yahoo Finance na primeira execução |
| `*.zip` | Pacotes que o Kaggle gera para download |
| `__pycache__/`, `.ipynb_checkpoints/` | Arquivos temporários do Python e do Jupyter |

Para refazer a etapa 04, basta colocar os dois arquivos de texto em `data/textos/`.

## Fluxo de trabalho

1. O código é editado no computador, na pasta do repositório.
2. A pasta inteira sobe para o Kaggle como Dataset (New Version a cada mudança).
3. O notebook da etapa é importado no Kaggle e roda com GPU e internet ligadas.
4. O zip de `/kaggle/working/results/` é baixado e extraído em `results/`.
5. Os números oficiais e a decisão da etapa são registrados em `docs/decisoes.md`.
6. A mudança é enviada ao GitHub com `git add -A`, `git commit` e `git push`.

Uma etapa nova segue o mesmo padrão: um notebook `NN_nome.ipynb`, uma pasta `results/NN_nome/` com os mesmos nomes de arquivo, uma linha no README e uma seção em `decisoes.md`.
