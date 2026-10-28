"""Preços diários e montagem das entradas da rede."""
import numpy as np
import pandas as pd

from .config import PASTA_PRECOS, RAIZ

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def precos(moeda, inicio, fim):
    """OHLCV diário (UTC) do Yahoo Finance, guardado em data/precos/ para não baixar de novo."""
    nome = f'{moeda}-USD.csv'
    dias = pd.date_range(inicio, fim, freq='D')
    for arquivo in [RAIZ / 'data' / 'precos' / nome, PASTA_PRECOS / nome]:
        if arquivo.exists():
            df = pd.read_csv(arquivo, index_col='Date', parse_dates=True)[OHLCV]
            if dias.difference(df.index).empty:
                return df.loc[inicio:fim]
    import yfinance as yf
    df = yf.download(f'{moeda}-USD', start=inicio, end=dias[-1] + pd.Timedelta(days=1),
                     auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[OHLCV].astype(float)
    df.index = pd.to_datetime(df.index).tz_localize(None).rename('Date')
    faltando = dias.difference(df.index)
    if len(faltando):
        raise ValueError(f'{moeda}: faltam {len(faltando)} dias entre {inicio} e {fim}')
    PASTA_PRECOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PASTA_PRECOS / nome)
    return df


def entradas(df, alvo, extras=None):
    """Mesma informação (OHLCV) em duas escalas.
    preco:   OHLCV em nível; alvo = fechamento.
    retorno: OHLCV em variação log (%) em relação ao fechamento/volume de ontem; alvo = retorno do fechamento.
    extras:  colunas a mais do mesmo dia (ex.: sentimento), juntadas pela data."""
    if alvo == 'preco':
        X = df[OHLCV].copy()
        y = df['Close']
    elif alvo == 'retorno':
        ontem = df['Close'].shift(1)
        X = pd.DataFrame({c: 100 * np.log(df[c] / ontem) for c in ['Open', 'High', 'Low', 'Close']})
        X['Volume'] = np.log(df['Volume'] / df['Volume'].shift(1))
        y = X['Close']
    else:
        raise ValueError("alvo deve ser 'preco' ou 'retorno'")
    if extras is not None:
        X = X.join(extras.reindex(X.index))
    return X.iloc[1:], y.iloc[1:]                     # o 1º dia não tem "ontem"


def janelas(X, y, tamanho):
    """Entrada: dias t-L+1..t. Alvo: dia t+1."""
    fins = range(tamanho - 1, len(X) - 1)
    return (np.stack([X[i - tamanho + 1:i + 1] for i in fins]).astype(np.float32),
            np.array([y[i + 1] for i in fins], dtype=np.float32))
