# Decisões

## 01 · Alvo: preço ou retorno

**Decisão:** a rede prevê o **retorno** de amanhã, e o preço sai de hoje × exp(retorno).

**Observação:** esta etapa usa uma LSTM padrão, com hiperparâmetros fixos escolhidos à mão (`HP_LSTM` em `config.py`: janela 14, 64 neurônios, 1 camada, dropout 0,1, lr 0,001, batch 32, até 100 épocas com parada pela validação). O objetivo não é achar a melhor rede, só comparar os dois alvos em condições iguais. A busca de hiperparâmetros com Optuna vem na etapa seguinte, já com o alvo escolhido aqui.

**Experimento:** `notebooks/01_lstm_preco_vs_retorno.ipynb`, com a mesma LSTM, os mesmos dados OHLCV e só o alvo trocado.
- 2022-01-01 a 2023-12-31, divisão 70/15/15.
- Teste de 13/09 a 31/12/2023 (110 dias).
- 5 sementes.

| Moeda | Naive | LSTM preço | LSTM retorno |
|---|---|---|---|
| BTC | 1,45 | 1,81 | 1,46 |
| ETH | 1,65 | 1,97 | 1,70 |
| SOL | 4,02 | 5,15 | 4,05 |
| BNB | 1,53 | 2,02 | 1,61 |
| DOGE | 2,14 | 2,43 | 2,16 |
| **Média** | **2,16** | **2,68** | **2,20** |

MAPE em %, média de 5 sementes. Rodada oficial no Kaggle, em 27/10/2026 (`results/01_lstm_ohlcv_2022-01-01_2023-12-31/resumo.csv`).

**Evidência:**
1. **Preço é pior que o naive em todas as moedas.** A diferença é significativa em 4 de 5 (Diebold-Mariano, p ≤ 0,01; DOGE p = 0,13).
2. **Retorno empata com o naive.** A diferença não é significativa em BTC, ETH, SOL e DOGE (p de 0,08 a 0,82). Em BNB fica um pouco pior (p = 0,01).
3. **Retorno é mais estável.** O desvio entre sementes fica entre 0,004 e 0,026 ponto, contra 0,05 a 0,35 do preço.
4. **A diferença não vem de extrapolação.** O preço do teste fica 100% dentro da faixa do treino nas 5 moedas.
5. **O retorno empata porque quase não corrige o naive.** A correção média prevista fica entre 0,17% e 0,42% ao dia, enquanto o movimento real médio fica entre 1,46% e 4,10%. O acerto de direção fica entre 41% e 53%, o que é acaso. Ou seja, a LSTM com OHLCV não achou padrão e ficou perto do naive. Isso é o comportamento seguro: ela não piora o piso.

**Por quê:**
- O preço em nível não é estacionário. Com ele, a rede aprende sobretudo a repetir o valor de ontem e soma ruído à previsão.
- O retorno oscila em torno de zero. Quando a rede não acha padrão, ela prevê perto de zero, o que equivale ao naive. Assim ela não fica pior que o piso.

**Peso: alto.** Com alvo em preço, qualquer comparação posterior (com ou sem sentimento, LSTM ou Transformer) começa em desvantagem contra o naive, por um motivo que não tem nada a ver com a pergunta do TCC.

## 02 · Hiperparâmetros da LSTM (Optuna)

**Decisão:** a LSTM passa a usar janela de 14 dias, 64 neurônios, 2 camadas, dropout 0,31, taxa de aprendizado 0,0076 e batch 64 (`results/02_optuna_lstm_retorno_2022-01-01_2023-12-31/melhores_hp.json`).

**Resultado da busca:**
- Nenhum dos 30 trials ficou abaixo de 1. O melhor ficou em 1,007 e a mediana em 1,013.
- Por moeda, só SOL ficou levemente abaixo de 1 (0,998).
- Os 10 melhores trials repetem 2 camadas e janela de 14 dias.

**Teste** (MAPE em %, 5 sementes, 13/09 a 31/12/2023, rodada oficial no Kaggle em 27/10/2026):

| Moeda | Naive | LSTM padrão | LSTM Optuna |
|---|---|---|---|
| BTC | 1,45 | 1,46 | 1,46 |
| ETH | 1,65 | 1,70 | 1,67 |
| SOL | 4,02 | 4,05 | 4,04 |
| BNB | 1,53 | 1,61 | 1,62 |
| DOGE | 2,14 | 2,16 | 2,15 |
| **Média** | **2,16** | **2,20** | **2,19** |

**Evidência:**
1. **O Optuna melhora pouco:** de 2,20% para 2,19% na média. A LSTM continua sem vencer o naive em nenhuma moeda.
2. **Diebold-Mariano contra o naive:** BTC, SOL e DOGE ficam empatados (p de 0,16 a 0,50). ETH (p = 0,04) e BNB (p = 0,01) ficam um pouco piores que o naive.
3. **A rede Optuna corrige ainda menos o naive:** a correção média prevista fica entre 0,07% e 0,37% ao dia, contra 0,17% a 0,42% da LSTM padrão. O acerto de direção fica entre 42% e 48%.

**Leitura:** a busca confirma o resultado da etapa 01. Com só OHLCV, a melhor LSTM encontrada é a que se afasta pouco do naive. A falta de ganho não vem de hiperparâmetros mal escolhidos: é a informação de entrada que não antecipa o retorno de amanhã. Esse é o ponto de partida para testar se o sentimento acrescenta essa informação.

**Peso: médio.** Os hiperparâmetros mudam os números em centésimos, mas não a conclusão.

**Como foi feito** (`notebooks/02_optuna_lstm.ipynb`):
- **Alvo:** retorno, escolhido na etapa 01.
- **Onde a busca acontece:** só nos dias antes do teste. A validação cruzada temporal tem 5 blocos na segunda metade desse período. Cada bloco treina com todo o passado (os últimos 15% dele param o treino) e prevê o bloco seguinte. O teste da etapa 01 (13/09 a 31/12/2023) não entra na busca.
- **Critério:** MAPE da rede ÷ MAPE do naive. Abaixo de 1, a rede é melhor que o naive. A divisão deixa moedas com volatilidades diferentes na mesma escala.
- **Um conjunto para as 5 moedas** (média do critério entre elas). Um conjunto por moeda tenderia a sobreajustar a cada série.
- **Espaço de busca:** janela {7, 14, 21, 30}, neurônios {16, 32, 64, 128}, camadas {1, 2}, dropout 0–0,4, taxa de aprendizado 0,0001–0,01 (log), batch {16, 32, 64}.
- **Busca:** 30 trials com o amostrador TPE, semente 42.
- **Teste final:** o melhor conjunto roda no mesmo teste da etapa 01, com 5 sementes, ao lado da LSTM padrão e do naive.

## 03 · Hiperparâmetros do Transformer (Optuna)

**Decisão:** o Transformer passa a usar janela de 30 dias, d_modelo 16, 4 cabeças, 3 camadas, dropout 0,16, taxa de aprendizado 0,0034 e batch 16 (`results/03_optuna_transformer_retorno_2022-01-01_2023-12-31/melhores_hp.json`).

**Resultado da busca:**
- Nenhum dos 30 trials ficou abaixo de 1. O melhor ficou em 1,003 e a mediana em 1,011.
- Os 10 melhores trials preferem rede pequena (d_modelo 16 em 9 de 10) e janela longa (21 ou 30 dias em 9 de 10).

**Teste** (MAPE em %, 5 sementes, 13/09 a 31/12/2023, rodada oficial no Kaggle em 27/10/2026):

| Moeda | Naive | LSTM Optuna | Transformer padrão | Transformer Optuna |
|---|---|---|---|---|
| BTC | 1,45 | 1,46 | 1,46 | 1,46 |
| ETH | 1,65 | 1,67 | 1,67 | 1,67 |
| SOL | 4,02 | 4,04 | 4,06 | 4,03 |
| BNB | 1,53 | 1,62 | 1,66 | 1,63 |
| DOGE | 2,14 | 2,15 | 2,15 | 2,16 |
| **Média** | **2,16** | **2,19** | **2,20** | **2,19** |

A LSTM Optuna é a da etapa 02, com as mesmas previsões.

**Evidência:**
1. **O Transformer também não vence o naive.** Pelo Diebold-Mariano, BTC, ETH, SOL e DOGE ficam empatados (p de 0,10 a 0,76). BNB fica um pouco pior (p = 0,01).
2. **Transformer e LSTM empatam.** O Diebold-Mariano entre os dois não é significativo em nenhuma moeda (p de 0,12 a 0,78, em `dm_transformer_vs_lstm.csv`).
3. **O Optuna deixa o Transformer mais estável.** O desvio entre sementes cai de até 0,09 ponto, no padrão, para no máximo 0,03.
4. **O comportamento é o mesmo da LSTM.** A correção média prevista sobre o naive fica entre 0,07% e 0,43% ao dia. O acerto de direção fica entre 37% e 53%.

**Leitura:** com alvo retorno e só OHLCV, a arquitetura não muda o resultado. As duas redes ficam presas no piso do naive, porque a entrada não antecipa o retorno de amanhã. Para o TCC, isso deixa uma base limpa: se o sentimento melhorar alguma das redes, o ganho vem da informação nova, e não da arquitetura ou dos hiperparâmetros.

**Peso: médio.** Os hiperparâmetros só mudam os números em centésimos. O que importa aqui é a base sem sentimento para as duas arquiteturas.

**Como foi feito** (`notebooks/03_optuna_transformer.ipynb`):
- **Protocolo:** o mesmo da etapa 02, com alvo retorno, validação cruzada temporal em 5 blocos antes do teste, critério MAPE ÷ naive, um conjunto para as 5 moedas e 30 trials com TPE, semente 42.
- **Arquitetura:** cada dia da janela passa por uma camada linear até d_modelo, recebe uma posição aprendida e entra no encoder do Transformer (pré-normalização, camada interna de 2 × d_modelo). A previsão é lida no último dia.
- **Espaço de busca:** janela {7, 14, 21, 30}, d_modelo {16, 32, 64}, cabeças {2, 4}, camadas 1 a 3, dropout 0–0,4, taxa de aprendizado 0,0001–0,01 (log), batch {16, 32, 64}.
- **Transformer padrão** (`HP_TRANSFORMER` em `config.py`, escolhido à mão): janela 14, d_modelo 32, 4 cabeças, 2 camadas, dropout 0,1, taxa de aprendizado 0,0005, batch 32.

## 04 · Sentimento diário com CryptoBERT

**Decisão:**
- O sentimento de cada moeda é a média diária das notas do CryptoBERT (`ElKulako/cryptobert`), separada em duas colunas: notícias (`SentimentoNoticia`) e tweets (`SentimentoTweet`).
- A nota de cada texto é P(bullish) − P(bearish), de −1 a +1.
- As colunas `TemNoticia` e `TemTweet` (0/1) dizem se o dia teve texto da própria moeda.
- Os dias sem texto são preenchidos de dois jeitos, e os dois seguem para a etapa seguinte:
  - **ffill:** repete o último valor da própria moeda.
  - **btc:** usa o valor do BTC no mesmo dia (`CopiaBTCNoticia`/`CopiaBTCTweet` = 1); se o BTC também não tiver, repete o último valor. O BTC usa só o ffill.
- Nos dois, antes do primeiro texto o valor é a média dos dias de treino.
- As duas fontes são usadas, decisão do autor.
- Tabelas finais em `results/04_sentimento_cryptobert_2022-01-01_2023-12-31/sentimento_ffill/` e `sentimento_btc/` (`{MOEDA}_sentimento.csv`).

**Textos** (2022-01-01 a 2023-12-31, rodada oficial no Kaggle em 27/10/2026): 20.203 textos diferentes.

| Moeda | Notícias | Dias com notícia | Tweets | Dias com tweet |
|---|---|---|---|---|
| BTC | 8.805 | 715 (97,9%) | 5.855 | 503 (68,9%) |
| ETH | 3.426 | 657 (90,0%) | 2.696 | 382 (52,3%) |
| SOL | 876 | 468 (64,1%) | 366 | 163 (22,3%) |
| BNB | 513 | 372 (51,0%) | 538 | 184 (25,2%) |
| DOGE | 654 | 425 (58,2%) | 332 | 106 (14,5%) |

- As notícias acabam em 19/12/2023 e os tweets em 12/06/2023. Os tweets não cobrem o teste (13/09 a 31/12/2023).
- Um texto entra numa moeda quando cita o nome ou o código dela (`PADROES_MOEDA` em `config.py`). Um texto que cita duas moedas entra nas duas.

**Preenchimento** (dias de 730 com valor próprio / copiado do BTC / repetido):

| Moeda | Notícia ffill | Notícia btc | Tweet ffill | Tweet btc |
|---|---|---|---|---|
| BTC | 715 / 0 / 15 | igual | 503 / 0 / 227 | igual |
| ETH | 657 / 0 / 73 | 657 / 58 / 15 | 382 / 0 / 348 | 382 / 134 / 214 |
| SOL | 468 / 0 / 262 | 468 / 247 / 15 | 163 / 0 / 567 | 163 / 342 / 225 |
| BNB | 372 / 0 / 358 | 372 / 343 / 15 | 184 / 0 / 546 | 184 / 321 / 225 |
| DOGE | 425 / 0 / 305 | 425 / 290 / 15 | 106 / 0 / 624 | 106 / 397 / 227 |

Depois de 12/06/2023 não há tweet de nenhuma moeda, então nos dois métodos o tweet fica constante em todo o teste.

**Notas:**
- **O CryptoBERT é otimista.** Nas notícias, 54,5% das notas são bullish (> 1/3), 38,8% neutras e só 6,6% bearish (< −1/3). Nos tweets, 52,0%, 44,9% e 3,1%. A nota média é +0,30 nas notícias e +0,34 nos tweets, por isso 0 não é neutro.
- **A classificação é reproduzível.** As notas batem com uma rodada anterior independente do CryptoBERT (correlação 0,999999 nos 19.843 textos em comum; diferença máxima de 0,05, efeito da GPU).

**Sentimento x retorno** (Spearman, só nos dias com texto, em `correlacao_retorno.csv`):

| | Com o retorno de hoje | Com o retorno de amanhã |
|---|---|---|
| Notícias | 0,11 a 0,17, significativa nas 5 moedas | −0,07 a 0,03, nenhuma significativa |
| Tweets | 0,04 a 0,15, significativa em BTC e BNB | −0,02 a 0,10, nenhuma significativa |

**Leitura:** o sentimento reflete o que o preço fez no mesmo dia, mas não antecipa o dia seguinte. Isso é o esperado num mercado eficiente e indica que o ganho do sentimento nas redes deve ser pequeno. A etapa seguinte testa isso com as duas redes.

**Peso: alto.** É a variável que o TCC testa.

**Como foi feito** (`notebooks/04_sentimento_cryptobert.ipynb`, `src/tcc/textos.py`, `src/tcc/sentimento.py`):
- **Notícias** (`cryptonews.csv`): título + ". " + texto, com a data da coluna `date`.
- **Tweets** (`dataset_52-person-...csv`): `full_text`, com a data de `created_at`. O arquivo veio com a codificação quebrada (UTF-8 lido como latin-1) e com entidades HTML. Os dois foram desfeitos trecho a trecho, e os bytes perdidos (emojis incompletos) foram removidos.
- **Limpeza:** espaços e quebras de linha viram um espaço. Textos vazios e repetidos dentro da mesma fonte saem.
- **CryptoBERT:** até 128 tokens por texto, o que cobre um tweet ou um título com resumo.
- **Preenchimento:** o valor anterior (*forward fill*) e o valor do BTC no mesmo dia usam só informação disponível até hoje. Interpolação linear usaria o valor de um dia futuro e vazaria informação.
- **Variante btc:** a ideia é que o BTC puxa o mercado. O sentimento do BTC no mesmo dia pode ser um substituto melhor que o valor velho da própria moeda.

## 05 · O sentimento melhora a previsão?

**Resultado:** não. Com os mesmos hiperparâmetros, acrescentar o sentimento não muda o erro da LSTM nem do Transformer, em nenhuma das duas formas de preenchimento.

**Teste** (MAPE em %, 5 sementes, 13/09 a 31/12/2023, rodada oficial no Kaggle em 27/10/2026):

| Moeda | Naive | LSTM fin. | LSTM ffill | LSTM btc | Transf. fin. | Transf. ffill | Transf. btc |
|---|---|---|---|---|---|---|---|
| BTC | 1,45 | 1,46 | 1,46 | 1,46 | 1,46 | 1,45 | 1,45 |
| ETH | 1,65 | 1,67 | 1,69 | 1,69 | 1,67 | 1,67 | 1,67 |
| SOL | 4,02 | 4,04 | 4,04 | 4,04 | 4,03 | 4,05 | 4,06 |
| BNB | 1,53 | 1,62 | 1,62 | 1,62 | 1,63 | 1,64 | 1,62 |
| DOGE | 2,14 | 2,15 | 2,14 | 2,14 | 2,16 | 2,18 | 2,16 |
| **Média** | **2,16** | **2,19** | **2,19** | **2,19** | **2,19** | **2,20** | **2,19** |

fin. = só OHLCV; ffill e btc = OHLCV + sentimento da etapa 04 com cada preenchimento. No BTC, ffill e btc são a mesma tabela.

**Evidência:**
1. **Nenhuma diferença significativa com o sentimento.** Diebold-Mariano contra a versão financeira do mesmo modelo, nas 20 comparações (2 redes × 2 preenchimentos × 5 moedas): p de 0,06 a 0,99, nenhum abaixo de 0,05 (`dm_sentimento_vs_financeiro.csv`). Os mais próximos são Transformer BTC (p = 0,06) e Transformer BNB com btc (p = 0,06), ambos a favor do sentimento, com ganho de 0,01 a 0,02 ponto.
2. **O sentimento não leva ao piso do naive para baixo.** Nenhuma versão vence o naive. BNB continua pior que o naive em todas (p ≤ 0,02).
3. **As redes quase não usam o sentimento.** A correção média sobre o naive passa de 0,17% para 0,19% ao dia na LSTM, e de 0,22% para 0,22–0,24% no Transformer. O acerto de direção continua entre 37% e 53%.
4. **ffill e btc empatam.** Copiar o BTC nos dias vazios não muda a conclusão.
5. **A rodada é reproduzível.** As versões financeiras repetem exatamente os números das etapas 02 e 03.

**Leitura:** o resultado confirma a etapa 04. O sentimento do CryptoBERT acompanha o retorno do mesmo dia, mas não o de amanhã. Uma rede que prevê amanhã não tem o que aproveitar dele. A conclusão vale para as duas arquiteturas e para os dois preenchimentos.

**Peso: alto.** É a pergunta principal do TCC.

**Como foi feito** (`notebooks/05_sentimento.ipynb`):
- **Hiperparâmetros:** os melhores das etapas 02 (LSTM) e 03 (Transformer), iguais nas três versões. Assim, a única diferença entre as versões é a entrada.
- **Entrada com sentimento:** OHLCV em retorno + SentimentoNoticia, SentimentoTweet, TemNoticia e TemTweet do mesmo dia. A janela vê o sentimento até hoje e prevê o retorno de amanhã. As colunas CopiaBTC não entram, para que ffill e btc só difiram no preenchimento.
- **Escala:** MinMax ajustada só no treino, como nas outras etapas.
- **Comparação:** Diebold-Mariano (correção HLN, erro absoluto) na previsão média das 5 sementes.

## 06 · Controle: as redes usariam um sentimento útil?

**Resultado:** sim. Quando o sinal antecipa o retorno de amanhã, as duas redes o aproveitam, e o ganho cresce com a força do sinal. Isso mostra que a falta de ganho da etapa 05 vem do sentimento real, que não antecipa o mercado, e não das redes.

**Sinal sintético:** `Sinal = tanh(ρ·z + √(1−ρ²)·ruído)`, com z = retorno padronizado (média e desvio só do treino) e ruído com semente 42. Ele entra como uma coluna a mais no OHLCV.

| Versão | O que o sinal conhece | Spearman com o retorno de amanhã no teste |
|---|---|---|
| amanha_1.0 | retorno de amanhã, sem ruído | 1,00 |
| amanha_0.5 | retorno de amanhã + ruído | 0,27 a 0,41 |
| amanha_0.25 | retorno de amanhã + mais ruído | 0,12 a 0,24 |
| amanha_0.1 | quase só ruído | 0,06 a 0,16 |
| hoje | retorno de hoje (como o sentimento real) | −0,11 a −0,01 |

Sentimento real (etapa 04): −0,07 a 0,10, nenhuma correlação significativa.

**Teste** (MAPE médio das 5 moedas em %, 5 sementes, 13/09 a 31/12/2023, rodada oficial no Kaggle em 27/10/2026; por moeda em `tabela_mape.csv`):

| Versão | LSTM | Transformer | Acerto de direção (LSTM / Transf.) |
|---|---|---|---|
| naive | 2,16 | 2,16 | — |
| financeiro | 2,19 | 2,19 | 44% / 44% |
| hoje | 2,19 | 2,18 | 46% / 47% |
| amanha_1.0 | **0,60** | **0,48** | 97% / 96% |
| amanha_0.5 | 2,13 | 2,11 | 59% / 58% |
| amanha_0.25 | 2,17 | 2,17 | 52% / 51% |
| amanha_0.1 | 2,19 | 2,18 | 44% / 49% |

**Evidência:**
1. **Com o sinal perfeito, as redes erram de 3 a 5 vezes menos** que o naive em todas as moedas. A melhora é significativa nas 10 comparações (Diebold-Mariano, p < 0,001). O Transformer aproveita um pouco mais que a LSTM (0,48% contra 0,60%).
2. **O ganho cai rápido com o ruído.**
   - Com ρ = 0,5 (correlação de 0,27 a 0,41), a média cai de 2,19% para 2,11–2,13%, e as redes ficam abaixo do naive na média. A direção sobe para 58–59%, mas só uma das 10 comparações é significativa por moeda (Transformer SOL, p = 0,01).
   - Com ρ = 0,25 ou 0,1, o resultado volta ao da versão financeira.
3. **O sinal de hoje não ajuda.** Ele se comporta como o sentimento real da etapa 05: MAPE igual ao financeiro e direção perto de 50%.
4. **Diferenças pequenas e "significativas" aparecem até sem informação.** No Transformer, colunas sem informação sobre amanhã (hoje e amanha_0.1) deram 4 diferenças com p < 0,05, todas de 0,01 a 0,03 ponto. Então os p ≈ 0,06 da etapa 05 estão no nível do ruído, e não indicam ganho real do sentimento.

**Leitura:**
- A arquitetura funciona: as duas redes aprendem a usar uma entrada que antecipa o retorno.
- Para dar ganho visível, o sentimento precisaria ter correlação de pelo menos uns 0,3 com o retorno de amanhã. O sentimento real do CryptoBERT tem correlação perto de 0 com o amanhã e só acompanha o mesmo dia, como o controle "hoje". Isso explica o resultado da etapa 05 e é coerente com a hipótese de mercado eficiente.

**Peso: alto.** É o que separa "o sentimento não ajuda" de "o modelo não consegue usar o sentimento".

**Como foi feito** (`notebooks/06_controle_sintetico.ipynb`, `src/tcc/controles.py`):
- Mesmos hiperparâmetros das etapas 02 e 03, mesmo alvo (retorno), mesmo teste e 5 sementes.
- A versão financeira é a mesma da etapa 05 (mesmas previsões).
- O sinal de amanhã entra na linha de hoje, ou seja, a janela que termina em t vê um sinal que conhece r(t+1). Isso é proposital: é o controle positivo, um sentimento que "adivinha" o mercado.

## 07 · Walk-forward por regime de mercado

**Resultado:** a conclusão das etapas 02 a 05 vale em todos os regimes.
- Nenhuma rede vence o naive em nenhum regime, com ou sem sentimento.
- O sentimento não melhora as redes em nenhum regime.

**Como foi feito** (`notebooks/07_walk_forward.ipynb`, `src/tcc/walkforward.py`):
- **Período:** 609 dias previstos, de 01/05/2022 a 31/12/2023, em 10 blocos de 2 meses.
- **Treino:** em cada bloco, a rede é retreinada com todo o passado (os últimos 15% dele param o treino, e as escalas usam só o treino) e prevê o bloco.
- **Redes e versões:** as mesmas da etapa 05 (hiperparâmetros das etapas 02 e 03, 5 sementes): financeiro, sentimento_ffill e sentimento_btc.
- **Regimes:** Queda 2022 (mai–dez/2022, com Terra/Luna e FTX), Lateral 2023 (jan–set/2023) e Alta fim de 2023 (out–dez/2023).

**MAPE ÷ MAPE do naive** (média das 5 moedas, rodada oficial no Kaggle em 27/10/2026; abaixo de 1 = melhor que o naive):

| Regime | Naive (MAPE %) | LSTM fin. | LSTM ffill | LSTM btc | Transf. fin. | Transf. ffill | Transf. btc |
|---|---|---|---|---|---|---|---|
| Queda 2022 | 3,37 | 1,023 | 1,032 | 1,028 | 1,028 | 1,056 | 1,030 |
| Lateral 2023 | 2,04 | 1,023 | 1,029 | 1,027 | 1,028 | 1,033 | 1,033 |
| Alta fim de 2023 | 2,38 | 1,005 | 1,007 | 1,007 | 1,008 | 1,009 | 1,006 |
| **Todos** | **2,62** | **1,021** | **1,028** | **1,025** | **1,026** | **1,042** | **1,028** |

MAPE por moeda e regime em `tabela_mape.csv`.

**Evidência:**
1. **Nenhuma versão vence o naive** em nenhuma das 90 combinações de regime, moeda, rede e versão (Diebold-Mariano, p < 0,05). Nos 20 meses, as redes ficam de 2% a 4% acima do erro do naive. Em 2 ou 3 das 5 moedas, a diferença é significativa.
2. **O sentimento não melhora em nenhum regime.** Diebold-Mariano contra a versão financeira:
   - Nos 3 regimes (60 comparações): 57 empates, 1 melhora (LSTM ETH com ffill na queda de 2022, p = 0,03) e 2 pioras (BNB com ffill na queda de 2022, nas duas redes).
   - Nos 20 meses juntos (20 comparações): 18 empates e 2 pioras (BNB com ffill, nas duas redes).
3. **O btc evita a piora da BNB.** A BNB tem notícia em só 51% dos dias. Com o ffill, o valor velho se repete por muito tempo e atrapalha a rede na queda de 2022. Copiando o BTC no dia, a BNB não piora em nenhum regime. Mesmo assim, o btc também não melhora nenhuma moeda.
4. **O regime muda o tamanho do erro, mas não a conclusão.** A queda de 2022 tem o maior erro (naive 3,37%) e a lateral de 2023 o menor (2,04%). As redes ficam presas ao naive nos três. O acerto de direção fica entre 46% e 51%.

**Leitura:** o resultado das etapas 05 e 06 não vem do período de teste escolhido. Em quedas, laterais e altas, o sentimento diário do CryptoBERT não leva informação sobre o retorno de amanhã que as redes possam usar.

**Ressalva:** o preenchimento dos primeiros dias sem texto usa a média do treino da divisão 70/15/15, que termina depois do primeiro bloco. Isso afeta só os dias antes do primeiro texto de cada moeda (até fevereiro de 2022, um valor constante). Ele não carrega informação sobre o teste.

**Peso: alto.** Responde à crítica mais comum de testes com um único período.

## 08 · Tabelas e figuras finais

O notebook `notebooks/08_tabelas_figuras.ipynb` não treina nada. Ele só lê os resultados oficiais das etapas 01 a 07 e gera o material do texto em `results/08_tabelas_figuras/`. Todas as tabelas também saem juntas em `tabelas_tcc.xlsx`, uma por aba.

| Arquivo | Conteúdo | Etapa |
|---|---|---|
| `t1_alvo_preco_vs_retorno` | MAPE com alvo preço e com alvo retorno, e p contra o naive | 01 |
| `t2_hiperparametros` | Hiperparâmetros finais da LSTM e do Transformer | 02, 03 |
| `t3_sentimento_cobertura_correlacao` | Textos, cobertura e Spearman com o retorno de hoje e de amanhã | 04 |
| `t4_resultado_principal` e `t4b_mape_resumido` | MAPE, desvio, RMSE, direção, p contra o naive e p contra a versão financeira | 05 |
| `t5_controle_sintetico` | MAPE e direção para cada força do sinal sintético | 06 |
| `t6_walk_forward_regimes` | MAPE ÷ naive por regime e contagem de DM significativos | 07 |
| `f1_precos_divisao` | Preços de 2022–2023 com a divisão 70/15/15 | — |
| `f2_sentimento_vs_retorno` | Spearman do sentimento com o retorno de hoje e de amanhã | 04 |
| `f3_razao_naive_com_sem_sentimento` | MAPE ÷ naive por moeda, com e sem sentimento | 05 |
| `f4_controle_sintetico` | MAPE ÷ naive pela correlação do sinal com amanhã, com o sentimento real marcado | 06 |
| `f5_walk_forward_regimes` | MAPE ÷ naive por regime | 07 |
| `f6_exemplo_btc_teste` | Previsões do Transformer no BTC (15/10 a 30/11/2023) contra o real e o naive | 05 |

## Conclusão geral

1. **O alvo retorno é o correto** (etapa 01). O alvo preço fica pior que o naive em todas as moedas.
2. **Só com OHLCV, nenhuma rede vence o naive** (etapas 02 e 03). O Optuna não muda isso, e LSTM e Transformer empatam.
3. **O sentimento do CryptoBERT reflete o retorno do mesmo dia, não o de amanhã** (etapa 04).
4. **Somar o sentimento não melhora a LSTM nem o Transformer**, com nenhum dos dois preenchimentos (etapa 05).
5. **As redes conseguiriam usar um sentimento útil** (etapa 06). Com um sinal que antecipa o retorno, o erro cai até 4 vezes abaixo do naive. O ganho só aparece com correlação de uns 0,3 ou mais com o amanhã, e o sentimento real fica perto de 0.
6. **A conclusão vale nos três regimes de mercado** de maio/2022 a dezembro/2023 (etapa 07).

Em resumo, a falta de ganho vem da informação, e não do modelo: o sentimento diário público chega ao preço no mesmo dia. Isso é coerente com a hipótese de mercado eficiente na forma semiforte.
