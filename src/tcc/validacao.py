"""Validação manual do CryptoBERT: amostra cega de textos, planilha para rotular e concordância."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CLASSES = ['bearish', 'neutro', 'bullish']
LIMITE = 1 / 3                  # mesmo corte da distribuição das notas na etapa 04


def classe(nota):
    return pd.cut(nota, [-1, -LIMITE, LIMITE, 1], labels=CLASSES, include_lowest=True).astype(str)


def amostra(notas, por_grupo=17, semente=42):
    """Mesmo número de textos em cada (fonte, classe do CryptoBERT), em ordem embaralhada.
    Sem estratificar, quase não haveria textos bearish na amostra."""
    t = notas.drop_duplicates(['fonte', 'texto']).copy()
    t['classe_cryptobert'] = classe(t['nota'])
    t = t.groupby(['fonte', 'classe_cryptobert'], group_keys=False).sample(por_grupo, random_state=semente)
    t['id'] = t['texto'].map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:10])
    return t.sample(frac=1, random_state=semente)[['id', 'fonte', 'moeda', 'data', 'texto', 'nota', 'classe_cryptobert']]


def planilha(a, caminho):
    """Planilha só com id, fonte e texto (sem a nota) e uma lista para escolher a classe."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); ws = wb.active; ws.title = 'rotular'
    ws.append(['id', 'fonte', 'texto', 'minha_classe'])
    for _, r in a.iterrows():
        ws.append([r['id'], r['fonte'], r['texto'], None])
    for col, largura in zip('ABCD', [13, 9, 100, 15]):
        ws.column_dimensions[col].width = largura
    for linha in ws.iter_rows(min_row=2):
        linha[2].alignment = Alignment(wrap_text=True, vertical='top')
    lista = DataValidation(type='list', formula1='"' + ','.join(CLASSES) + '"', allow_blank=True)
    ws.add_data_validation(lista); lista.add(f'D2:D{len(a) + 1}')
    ws.freeze_panes = 'A2'
    caminho.parent.mkdir(parents=True, exist_ok=True)
    wb.save(caminho)


def concordancia(real, previsto):
    """Acurácia, kappa de Cohen e kappa ponderado (as classes têm ordem: bearish < neutro < bullish)."""
    codigo = {c: i for i, c in enumerate(CLASSES)}
    a, b = real.map(codigo), previsto.map(codigo)
    return {'textos': len(a), 'acuracia_%': 100 * np.mean(a.values == b.values),
            'kappa': cohen_kappa_score(a, b), 'kappa_ponderado': cohen_kappa_score(a, b, weights='linear')}


def matriz(real, previsto):
    m = confusion_matrix(real, previsto, labels=CLASSES)
    return pd.DataFrame(m, index=pd.Index(CLASSES, name='minha_classe'), columns=pd.Index(CLASSES, name='cryptobert'))
