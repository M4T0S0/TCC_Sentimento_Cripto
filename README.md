# Predição de criptoativos com análise de sentimentos

TCC de Engenharia de Computação, UTFPR Pato Branco.
*Predição de séries temporais de criptoativos utilizando análise de sentimentos no modelo Transformer.*
Orientador: Prof. Dr. Ives Rene Venturini Pola.

## Etapas

| # | Notebook | Pergunta |
|---|---|---|
| 01 | `01_lstm_preco_vs_retorno` | A LSTM deve prever o preço ou o retorno? (LSTM padrão, hiperparâmetros fixos à mão) |
| 02 | `02_optuna_lstm` | Quais hiperparâmetros da LSTM, com alvo retorno? (Optuna) |
| 03 | `03_optuna_transformer` | O Transformer, com alvo retorno e Optuna, supera a LSTM? |
| 04 | `04_sentimento_cryptobert` | Limpeza dos textos e sentimento diário por moeda (CryptoBERT) |
| 05 | `05_sentimento` | O sentimento (ffill ou BTC) melhora a LSTM e o Transformer? |
| 06 | `06_controle_sintetico` | As redes usariam um sentimento que antecipasse o mercado? (controle) |
| 07 | `07_walk_forward` | A conclusão vale em todos os regimes de mercado (queda, lateral e alta)? |
| 08 | `08_tabelas_figuras` | Tabelas e figuras finais para o texto (só lê `results/`) |

## Estrutura

```
src/tcc/
  config.py        moedas, sementes, divisão e hiperparâmetros
  dados.py         preços do Yahoo Finance e montagem das entradas
  modelo.py        LSTM, Transformer e treino
  avaliacao.py     MAPE, RMSE, direção e Diebold-Mariano
  experimento.py   treino, teste, resumo e comparação (Diebold-Mariano)
  otimizacao.py    busca de hiperparâmetros (Optuna)
  textos.py        limpeza das notícias e tweets e separação por moeda
  sentimento.py    CryptoBERT, média diária e preenchimento
  controles.py     sentimento sintético para os controles
  walkforward.py   retreino em blocos de meses e rótulo de regime
notebooks/         um notebook por etapa
data/precos/       OHLCV diário salvo na primeira execução
data/textos/       cryptonews.csv e dataset_52-person-...csv (brutos, fora do Git)
results/           saídas de cada notebook
```

## Como rodar

- **Kaggle:** suba esta pasta como Dataset, abra o notebook, ligue a Internet (para baixar os preços) e a GPU.
- **PC:** `pip install -r requirements.txt` e abra `notebooks/`.

## Dados

- **Preços:** baixados do Yahoo Finance na primeira execução (`data/precos/`).
- **Textos:** os arquivos brutos de notícias (`cryptonews.csv`) e tweets (`dataset_52-person-from-2021-02-05_2023-06-12_21-34-17-266_with_sentiment.csv`) não ficam no repositório. Para refazer a etapa 04, coloque-os em `data/textos/`.
- **Resultados:** `results/` guarda as saídas oficiais de cada etapa. As notas texto a texto (`nota_por_texto.csv`) ficam de fora, porque contêm os textos originais.
