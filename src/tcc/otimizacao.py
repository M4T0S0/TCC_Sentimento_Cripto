"""Busca de hiperparâmetros com Optuna, sem tocar no período de teste."""
import numpy as np
import optuna

from . import avaliacao, config, dados, experimento


def espaco(trial, arquitetura='LSTM'):
    if arquitetura == 'Transformer':
        return dict(arquitetura='Transformer',
                    janela=trial.suggest_categorical('janela', [7, 14, 21, 30]),
                    d_modelo=trial.suggest_categorical('d_modelo', [16, 32, 64]),
                    cabecas=trial.suggest_categorical('cabecas', [2, 4]),
                    camadas=trial.suggest_int('camadas', 1, 3),
                    dropout=trial.suggest_float('dropout', 0.0, 0.4),
                    lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                    batch=trial.suggest_categorical('batch', [16, 32, 64]),
                    epocas=config.HP_TRANSFORMER['epocas'], paciencia=config.HP_TRANSFORMER['paciencia'])
    return dict(janela=trial.suggest_categorical('janela', [7, 14, 21, 30]),
                hidden=trial.suggest_categorical('hidden', [16, 32, 64, 128]),
                camadas=trial.suggest_int('camadas', 1, 2),
                dropout=trial.suggest_float('dropout', 0.0, 0.4),
                lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                batch=trial.suggest_categorical('batch', [16, 32, 64]),
                epocas=config.HP_LSTM['epocas'], paciencia=config.HP_LSTM['paciencia'])


def erro_relativo(X, y, fechamento, alvo, hp, semente=42):
    """Validação cruzada temporal nos dias antes do teste: cada dobra treina com o passado
    (os últimos 15% dele param o treino) e prevê o bloco seguinte. Devolve MAPE da rede ÷ MAPE do naive."""
    _, i_teste = experimento.cortes(len(X))
    blocos = np.linspace(int(i_teste * 0.5), i_teste, config.DOBRAS_CV + 1).astype(int)
    razoes = []
    for a, b in zip(blocos[:-1], blocos[1:]):
        i_val = int(a * (1 - config.FRAC_VALIDACAO))
        dia, ontem, previsto = experimento.treinar_e_prever(X, y, fechamento, alvo, hp, semente, i_val, a, b)
        real = fechamento[dia]
        razoes.append(avaliacao.mape(real, previsto) / avaliacao.mape(real, ontem))
    return float(np.mean(razoes))


def buscar(alvo, inicio, fim, trials=config.TRIALS_OPTUNA, moedas=config.MOEDAS, arquitetura='LSTM'):
    """Um único conjunto de hiperparâmetros para todas as moedas (média do erro relativo das moedas)."""
    series = {}
    for moeda in moedas:
        df = dados.precos(moeda, inicio, fim)
        X, y = dados.entradas(df, alvo)
        series[moeda] = (X.values, y.values, df['Close'].loc[X.index].values)

    def objetivo(trial):
        hp = espaco(trial, arquitetura)
        erros = {m: erro_relativo(*s, alvo, hp) for m, s in series.items()}
        for m, e in erros.items():
            trial.set_user_attr(m, e)
        return float(np.mean(list(erros.values())))

    estudo = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=trials)
    return estudo, espaco(optuna.trial.FixedTrial(estudo.best_params), arquitetura)
