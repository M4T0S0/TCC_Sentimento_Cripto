# Sentimento e previsão de criptoativos

Código do meu Trabalho de Conclusão de Curso em Engenharia de Computação na UTFPR, campus Pato Branco, orientado pelo Prof. Dr. Ives Rene Venturini Pola.

O trabalho testa se o sentimento de notícias e tweets, medido com o CryptoBERT, melhora a previsão do preço de fechamento diário de BTC, ETH, SOL, BNB e DOGE feita por uma LSTM e por um Transformer. O período vai de janeiro de 2022 a dezembro de 2023, e toda previsão é comparada com o modelo ingênuo, que repete o preço de ontem.

Em resumo, nenhum modelo superou o ingênuo, e o sentimento não melhorou a previsão. O sentimento acompanha o retorno do mesmo dia, mas não antecipa o do dia seguinte. As decisões e os números de cada etapa estão em `docs/decisoes.md`.

## Etapas

Cada etapa é um notebook em `notebooks/` e grava suas saídas na pasta de mesmo nome em `results/`.

1. `01_alvo`: compara prever o preço com prever o retorno.
2. `02_lstm`: busca dos hiperparâmetros da LSTM com Optuna.
3. `03_transformer`: o mesmo para o Transformer.
4. `04_sentimento`: limpa os textos e calcula o sentimento diário de cada moeda.
5. `05_com_sentimento`: treina as duas redes com e sem sentimento.
6. `06_controle`: verifica se as redes aproveitariam um sentimento que antecipasse o preço.
7. `07_walk_forward`: repete a comparação em queda, lateralização e alta.
8. `08_tabelas_figuras`: gera as tabelas e figuras do texto.

O código comum às etapas fica em `src/tcc/`.

## Como rodar

No Kaggle, suba a pasta como Dataset, importe o notebook e ligue a GPU e a internet. No computador, instale as dependências com `pip install -r requirements.txt`.

Os preços são baixados do Yahoo Finance na primeira execução. Os textos brutos não estão no repositório porque não fui eu quem os publicou; para refazer a etapa 4, coloque `cryptonews.csv` e o arquivo de tweets em `data/textos/`.

## Texto do TCC

Os arquivos, tabelas e figuras deste repositório são referenciados no documento final do TCC, que também será disponibilizado aqui.
