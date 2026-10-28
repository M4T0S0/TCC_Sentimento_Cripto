"""Walk-forward: a rede é retreinada a cada bloco de meses com todo o passado e prevê o bloco seguinte."""
import pandas as pd

from . import config, dados, experimento


def blocos(datas, inicio, meses):
    """Pares (i_inicio, i_fim) das linhas de cada bloco de teste."""
    datas = pd.DatetimeIndex(datas)
    cortes = pd.date_range(inicio, datas[-1] + pd.Timedelta(days=1), freq=f'{meses}MS')
    i = [datas.searchsorted(c) for c in cortes] + [len(datas)]
    return [(a, b) for a, b in zip(i[:-1], i[1:]) if b > a]


def regime(datas):
    datas = pd.DatetimeIndex(datas)
    rotulo = pd.Series('', index=range(len(datas)))
    for nome, (a, b) in config.REGIMES.items():
        rotulo[(datas >= a) & (datas <= b)] = nome
    return rotulo.values


def rodar(moeda, alvo, inicio, fim, hp, sementes=config.SEMENTES, extras=None,
          inicio_teste=config.INICIO_WALK_FORWARD, meses=config.MESES_POR_BLOCO):
    """Previsões de todos os blocos, uma linha por (semente, dia). Em cada bloco, os últimos 15% do passado param o treino."""
    df = dados.precos(moeda, inicio, fim)
    X, y = dados.entradas(df, alvo, extras)
    fechamento = df['Close'].loc[X.index].values
    linhas = []
    for n, (a, b) in enumerate(blocos(X.index, inicio_teste, meses), 1):
        i_val = int(a * (1 - config.FRAC_VALIDACAO))
        for semente in sementes:
            dia, ontem, previsto = experimento.treinar_e_prever(X.values, y.values, fechamento, alvo, hp, semente, i_val, a, b)
            linhas.append(pd.DataFrame({'moeda': moeda, 'bloco': n, 'semente': semente, 'data': X.index[dia],
                                        'real': fechamento[dia], 'naive': ontem, 'previsto': previsto}))
    prev = pd.concat(linhas, ignore_index=True)
    return prev.assign(regime=regime(prev['data']))
