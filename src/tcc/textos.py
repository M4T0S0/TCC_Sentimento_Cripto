"""Limpeza das notícias e dos tweets e separação dos textos por moeda."""
import html
import re
from pathlib import Path

import pandas as pd

from . import config

_QUEBRADO = re.compile(r'[\xc0-\xff][\x80-\xbf]+')            # UTF-8 lido como latin-1 ('â\x80\x99')


def achar(nome):
    """Procura o arquivo em data/textos/ do projeto e nos datasets do Kaggle."""
    achados = [config.RAIZ / 'data' / 'textos' / nome, *sorted(Path('/kaggle/input').glob(f'**/{nome}'))]
    for arquivo in achados:
        if arquivo.exists():
            return arquivo
    raise FileNotFoundError(f'{nome} não encontrado em data/textos/ nem em /kaggle/input')


def _consertar(m):
    trecho = m.group(0)
    for tentativa in (trecho, trecho.replace('ã', 'Ã', 1)):     # o arquivo veio em minúsculas e estragou alguns 'Ã'
        try:
            return tentativa.encode('latin-1').decode('utf-8')
        except UnicodeDecodeError:
            pass
    return ''                                                     # bytes perdidos (emoji incompleto)


def limpar(textos):
    """Desfaz a codificação quebrada e as entidades HTML ('&amp;') e junta espaços e quebras de linha."""
    textos = textos.fillna('').map(lambda t: html.unescape(_QUEBRADO.sub(_consertar, t)))
    return textos.str.replace(r'\s+', ' ', regex=True).str.strip()


def noticias():
    df = pd.read_csv(achar(config.ARQ_NOTICIAS))
    return pd.DataFrame({'fonte': 'noticia', 'data': pd.to_datetime(df['date'], errors='coerce').dt.normalize(),
                         'texto': limpar(df['title'].fillna('') + '. ' + df['text'].fillna(''))})


def tweets():
    df = pd.read_csv(achar(config.ARQ_TWEETS))
    return pd.DataFrame({'fonte': 'tweet', 'data': pd.to_datetime(df['created_at'], format='%m/%d/%Y', errors='coerce'),
                         'texto': limpar(df['full_text'])})


def por_moeda(inicio, fim):
    """Uma linha por (moeda, texto). Um texto que cita duas moedas entra nas duas."""
    todos = pd.concat([noticias(), tweets()], ignore_index=True).dropna(subset=['data'])
    todos = todos[todos['data'].between(inicio, fim) & (todos['texto'] != '')]
    todos = todos.drop_duplicates(['fonte', 'texto'])
    minusculo = todos['texto'].str.lower()
    partes = [todos[minusculo.str.contains(padrao)].assign(moeda=moeda) for moeda, padrao in config.PADROES_MOEDA.items()]
    return pd.concat(partes, ignore_index=True)[['moeda', 'fonte', 'data', 'texto']]


def cobertura(longa, inicio, fim):
    """Textos e dias com pelo menos um texto, por moeda e fonte."""
    dias = len(pd.date_range(inicio, fim, freq='D'))
    tab = (longa.groupby(['moeda', 'fonte'], sort=False)
           .agg(textos=('texto', 'size'), dias_com_texto=('data', 'nunique'), primeiro=('data', 'min'), ultimo=('data', 'max'))
           .reset_index())
    tab['cobertura_%'] = tab['dias_com_texto'] / dias * 100
    return tab
