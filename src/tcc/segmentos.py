"""Sentimento por segmento: o sentimento de cada tipo de texto (veículo, assunto, força, engajamento) em separado."""
import pandas as pd

from . import experimento

FORTE = 0.5                     # |nota| a partir da qual o texto é "forte"


def marcar(longa, fim_treino):
    """Uma coluna booleana por segmento. O limite de curtidas usa só os tweets do treino."""
    noticia, tweet = longa['fonte'] == 'noticia', longa['fonte'] == 'tweet'
    corte = longa.loc[tweet & (longa['data'] <= fim_treino), 'curtidas'].quantile(0.75)
    seg = {f'noticia_{v}': noticia & (longa['veiculo'] == v) for v in sorted(longa.loc[noticia, 'veiculo'].dropna().unique())}
    seg |= {f'assunto_{a}': noticia & (longa['assunto'] == a) for a in sorted(longa.loc[noticia, 'assunto'].dropna().unique())}
    seg |= {'texto_forte': longa['nota'].abs() >= FORTE, 'texto_fraco': longa['nota'].abs() < FORTE,
            'tweet_popular': tweet & (longa['curtidas'] >= corte), 'tweet_comum': tweet & (longa['curtidas'] < corte)}
    return pd.DataFrame(seg)


def diario(longa, mascara, moeda, inicio, fim):
    """Média do dia do segmento (vazio sem texto)."""
    dias = pd.date_range(inicio, fim, freq='D', name='Date')
    d = longa[mascara & (longa['moeda'] == moeda)]
    return d.groupby('data')['nota'].mean().reindex(dias)


def preencher(serie):
    """Mesmo preenchimento da etapa 04 (ffill): Tem = 1 no dia com texto, dia vazio repete o último valor,
    antes do primeiro texto vale a média dos dias de treino."""
    fim_treino = experimento.cortes(len(serie) - 1)[0] + 1
    tab = pd.DataFrame({'Tem': serie.notna().astype(int)})
    tab['Sentimento'] = serie.ffill().fillna(serie.iloc[:fim_treino].mean())
    return tab[['Sentimento', 'Tem']]
