"""CryptoBERT: nota de cada texto, média diária por fonte e preenchimento dos dias sem texto."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, experimento

FONTES = {'tweet': 'SentimentoTweet', 'noticia': 'SentimentoNoticia'}
COLUNAS = ['SentimentoTweet', 'SentimentoNoticia', 'TemTweet', 'TemNoticia']


def classificar(textos, lote=64):
    """Nota = P(bullish) − P(bearish), de −1 a +1."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizador = AutoTokenizer.from_pretrained(config.MODELO_SENTIMENTO)
    modelo = AutoModelForSequenceClassification.from_pretrained(config.MODELO_SENTIMENTO).to(dispositivo).eval()
    rotulo = {r.lower(): i for i, r in modelo.config.id2label.items()}       # bearish / neutral / bullish
    notas = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            entrada = tokenizador(textos[i:i + lote], padding=True, truncation=True,
                                  max_length=config.MAX_TOKENS, return_tensors='pt').to(dispositivo)
            p = torch.softmax(modelo(**entrada).logits, -1).cpu().numpy()
            notas.extend(p[:, rotulo['bullish']] - p[:, rotulo['bearish']])
    return np.array(notas, dtype=float)


def diario(longa, moeda, inicio, fim):
    """Média das notas do dia, por fonte. Dia sem texto fica vazio."""
    dias = pd.date_range(inicio, fim, freq='D', name='Date')
    tab = longa[longa['moeda'] == moeda].groupby(['data', 'fonte'])['nota'].mean().unstack('fonte')
    return tab.reindex(index=dias, columns=list(FONTES)).rename(columns=FONTES)


def preencher(tab, btc=None):
    """Tem* = 1 quando o dia tem texto da própria moeda.
    Com `btc` (tabela diária do BTC sem preencher), o dia vazio recebe antes o valor do BTC no mesmo dia (CopiaBTC* = 1).
    O que continuar vazio repete o último valor (só usa o passado).
    Antes do primeiro texto: média dos dias de treino da divisão 70/15/15."""
    fim_treino = experimento.cortes(len(tab) - 1)[0] + 1        # as entradas começam no 2º dia
    tab, extras = tab.copy(), []
    for col, nome in zip(FONTES.values(), ['Tweet', 'Noticia']):
        tab[f'Tem{nome}'] = tab[col].notna().astype(int)
        if btc is not None:
            do_btc = btc[col].reindex(tab.index)
            tab[f'CopiaBTC{nome}'] = (tab[col].isna() & do_btc.notna()).astype(int)
            tab[col] = tab[col].fillna(do_btc)
            extras.append(f'CopiaBTC{nome}')
        tab[col] = tab[col].ffill().fillna(tab[col].iloc[:fim_treino].mean())
    return tab[COLUNAS + extras]


def carregar(moeda, metodo, inicio, fim, colunas=COLUNAS):
    """Tabela diária da etapa 04 (`metodo` = 'ffill' ou 'btc')."""
    pasta = Path('results') / f'04_sentimento_cryptobert_{inicio}_{fim}' / f'sentimento_{metodo}'
    for base in [config.RAIZ, config.GRAVAVEL]:
        arquivo = base / pasta / f'{moeda}_sentimento.csv'
        if arquivo.exists():
            return pd.read_csv(arquivo, index_col='Date', parse_dates=True)[colunas]
    raise FileNotFoundError(f'{pasta}/{moeda}_sentimento.csv não encontrado. Rode a etapa 04.')
