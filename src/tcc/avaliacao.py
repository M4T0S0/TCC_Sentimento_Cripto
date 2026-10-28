"""Métricas de erro e teste de Diebold-Mariano."""
import numpy as np
from scipy import stats


def mape(real, previsto):
    return float(np.mean(np.abs(previsto - real) / real) * 100)


def rmse(real, previsto):
    return float(np.sqrt(np.mean((previsto - real) ** 2)))


def acerto_direcao(real, previsto, ontem):
    """% dos dias em que o modelo acertou se o preço sobe ou cai em relação a hoje."""
    return float(np.mean(np.sign(previsto - ontem) == np.sign(real - ontem)) * 100)


def diebold_mariano(erro_a, erro_b):
    """DM com correção de Harvey-Leybourne-Newbold (horizonte 1), perda = erro absoluto.
    Estatística < 0: A erra menos que B. Devolve (estatística, p)."""
    d = np.abs(erro_a) - np.abs(erro_b)
    n = len(d)
    if d.var() == 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(d.var() / n) * np.sqrt((n - 1) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))
