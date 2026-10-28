"""Sentimento sintético para os controles: um sinal que conhece, em parte, o retorno de hoje ou de amanhã."""
import numpy as np
import pandas as pd

from . import dados, experimento


def sinal(moeda, inicio, fim, rho, quando='amanha', semente=42):
    """Sinal = tanh(ρ·z + √(1−ρ²)·ruído), com z = retorno padronizado (média e desvio só do treino).
    quando='amanha': z é o retorno de amanhã (o que a rede tenta prever). quando='hoje': retorno de hoje."""
    _, r = dados.entradas(dados.precos(moeda, inicio, fim), 'retorno')
    treino = r.iloc[:experimento.cortes(len(r))[0]]
    z = (r - treino.mean()) / treino.std()
    if quando == 'amanha':
        z = z.shift(-1).fillna(0)                       # o último dia não tem amanhã (nunca entra numa janela)
    ruido = np.random.default_rng(semente).standard_normal(len(z))
    return pd.DataFrame({'Sinal': np.tanh(rho * z + np.sqrt(1 - rho ** 2) * ruido)}, index=z.index)
