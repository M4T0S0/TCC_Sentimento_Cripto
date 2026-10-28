"""LSTM, Transformer e treino."""
import copy
import random

import numpy as np
import torch
from torch import nn

DISPOSITIVO = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTM(nn.Module):
    def __init__(self, n_entradas, hidden, camadas, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_entradas, hidden, camadas, batch_first=True,
                            dropout=dropout if camadas > 1 else 0.0)
        self.saida = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.saida(self.lstm(x)[0][:, -1]).squeeze(-1)


class Transformer(nn.Module):
    """Encoder do Transformer: projeta cada dia em d_modelo, soma a posição aprendida e lê o último dia."""
    def __init__(self, n_entradas, janela, d_modelo, cabecas, camadas, dropout):
        super().__init__()
        self.entrada = nn.Linear(n_entradas, d_modelo)
        self.posicao = nn.Parameter(torch.zeros(1, janela, d_modelo))
        bloco = nn.TransformerEncoderLayer(d_modelo, cabecas, 2 * d_modelo, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(bloco, camadas, enable_nested_tensor=False)
        self.saida = nn.Sequential(nn.LayerNorm(d_modelo), nn.Dropout(dropout), nn.Linear(d_modelo, 1))

    def forward(self, x):
        return self.saida(self.encoder(self.entrada(x) + self.posicao)[:, -1]).squeeze(-1)


def criar(n_entradas, hp):
    if hp.get('arquitetura', 'LSTM') == 'Transformer':
        return Transformer(n_entradas, hp['janela'], hp['d_modelo'], hp['cabecas'], hp['camadas'], hp['dropout'])
    return LSTM(n_entradas, hp['hidden'], hp['camadas'], hp['dropout'])


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def treinar(X_tr, y_tr, X_va, y_va, hp, semente):
    """Adam + MSE. Guarda os pesos da época com menor erro na validação (para após `paciencia` épocas sem melhora)."""
    fixar_semente(semente)
    modelo = criar(X_tr.shape[-1], hp).to(DISPOSITIVO)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=hp['lr'])
    X_tr, y_tr, X_va, y_va = (torch.tensor(a, device=DISPOSITIVO) for a in (X_tr, y_tr, X_va, y_va))
    melhor, melhor_erro, sem_melhora = None, float('inf'), 0
    for _ in range(hp['epocas']):
        modelo.train()
        for lote in torch.randperm(len(X_tr), device=DISPOSITIVO).split(hp['batch']):
            otimizador.zero_grad()
            nn.functional.mse_loss(modelo(X_tr[lote]), y_tr[lote]).backward()
            otimizador.step()
        modelo.eval()
        with torch.no_grad():
            erro = nn.functional.mse_loss(modelo(X_va), y_va).item()
        if erro < melhor_erro:
            melhor, melhor_erro, sem_melhora = copy.deepcopy(modelo.state_dict()), erro, 0
        else:
            sem_melhora += 1
            if sem_melhora >= hp['paciencia']:
                break
    modelo.load_state_dict(melhor)
    return modelo.eval()


def prever(modelo, X):
    with torch.no_grad():
        return modelo(torch.tensor(X, device=DISPOSITIVO)).cpu().numpy()
