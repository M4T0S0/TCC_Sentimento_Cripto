"""Treino e teste das redes em uma divisão cronológica treino / validação / teste."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import avaliacao, config, dados, modelo


def cortes(n):
    """Índices onde começam a validação e o teste (70 / 15 / 15)."""
    return int(n * (1 - config.FRAC_VALIDACAO - config.FRAC_TESTE)), int(n * (1 - config.FRAC_TESTE))


def treinar_e_prever(X, y, fechamento, alvo, hp, semente, i_val, i_teste, i_fim):
    """Treina com as linhas [0, i_val), para pela validação [i_val, i_teste) e prevê o preço das linhas [i_teste, i_fim).
    As escalas usam só as linhas de treino. Devolve (linhas previstas, preço de ontem, preço previsto)."""
    L = hp['janela']
    escala_x = MinMaxScaler().fit(X[:i_val])
    escala_y = MinMaxScaler().fit(y[:i_val].reshape(-1, 1))
    Xj, yj = dados.janelas(escala_x.transform(X[:i_fim]), escala_y.transform(y[:i_fim].reshape(-1, 1)).ravel(), L)
    dia = np.arange(L, i_fim)                                    # linha prevista por cada janela
    tr, va, te = dia < i_val, (dia >= i_val) & (dia < i_teste), dia >= i_teste
    rede = modelo.treinar(Xj[tr], yj[tr], Xj[va], yj[va], hp, semente)
    saida = escala_y.inverse_transform(modelo.prever(rede, Xj[te]).reshape(-1, 1)).ravel()
    ontem = fechamento[dia[te] - 1]
    previsto = ontem * np.exp(saida / 100) if alvo == 'retorno' else saida
    return dia[te], ontem, previsto


def rodar(moeda, alvo, inicio, fim, hp=config.HP_LSTM, sementes=config.SEMENTES, extras=None):
    """Previsões do período de teste, uma linha por (semente, dia). `extras`: colunas a mais (ex.: sentimento)."""
    df = dados.precos(moeda, inicio, fim)
    X, y = dados.entradas(df, alvo, extras)
    fechamento = df['Close'].loc[X.index].values
    i_val, i_teste = cortes(len(X))
    linhas = []
    for semente in sementes:
        dia, ontem, previsto = treinar_e_prever(X.values, y.values, fechamento, alvo, hp, semente, i_val, i_teste, len(X))
        linhas.append(pd.DataFrame({'moeda': moeda, 'alvo': alvo, 'semente': semente, 'data': X.index[dia],
                                    'real': fechamento[dia], 'naive': ontem, 'previsto': previsto}))
    return pd.concat(linhas, ignore_index=True)


def faixa_do_treino(moeda, inicio, fim):
    """% dos dias de teste com preço fora da faixa [mínimo, máximo] vista no treino."""
    close = dados.precos(moeda, inicio, fim)['Close'].iloc[1:]
    i_val, i_teste = cortes(len(close))
    treino, teste = close.iloc[:i_val], close.iloc[i_teste:]
    return {'moeda': moeda, 'treino': f'{treino.index[0].date()} a {treino.index[-1].date()}',
            'teste': f'{teste.index[0].date()} a {teste.index[-1].date()}',
            'min_treino': treino.min(), 'max_treino': treino.max(),
            'teste_fora_da_faixa_%': float(((teste < treino.min()) | (teste > treino.max())).mean() * 100)}


def resumir(prev, chave='alvo'):
    """Por moeda e `chave` (uma coluna ou lista): MAPE médio ± desvio entre sementes, naive, direção e DM contra o naive."""
    chaves = [chave] if isinstance(chave, str) else list(chave)
    linhas = []
    for valores, g in prev.groupby(['moeda', *chaves], sort=False):
        por_semente = g.groupby('semente').apply(lambda s: avaliacao.mape(s.real, s.previsto))
        media = g.groupby('data')[['real', 'naive', 'previsto']].mean()          # previsão média das sementes
        dm, p = avaliacao.diebold_mariano(media.previsto - media.real, media.naive - media.real)
        linhas.append({**dict(zip(['moeda', *chaves], valores)),
                       'MAPE_%': por_semente.mean(), 'desvio_sementes': por_semente.std(),
                       'MAPE_naive_%': avaliacao.mape(media.real, media.naive),
                       'RMSE': avaliacao.rmse(media.real, media.previsto),
                       'acerto_direcao_%': avaliacao.acerto_direcao(media.real, media.previsto, media.naive),
                       'DM_vs_naive': dm, 'p_vs_naive': p, 'dias_teste': len(media)})
    return pd.DataFrame(linhas)


def comparar(prev, base, chave, outros, por=('moeda',)):
    """Diebold-Mariano de cada valor de `outros` contra `base` (na coluna `chave`), na previsão média das sementes.
    DM < 0: erra menos que a base."""
    media = prev.groupby([*por, chave, 'data'])[['real', 'previsto']].mean()
    linhas = []
    for grupo in prev[list(por)].drop_duplicates().itertuples(index=False):
        b = media.loc[(*grupo, base)]
        for outro in outros:
            o = media.loc[(*grupo, outro)]
            dm, p = avaliacao.diebold_mariano(o.previsto - o.real, b.previsto - b.real)
            linhas.append({**dict(zip(por, grupo)), chave: outro, 'contra': base, 'DM': dm, 'p': p})
    return pd.DataFrame(linhas)
