"""Parâmetros fixos do estudo."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
GRAVAVEL = Path('/kaggle/working') if Path('/kaggle/working').exists() else RAIZ   # no Kaggle, o dataset é só leitura
PASTA_PRECOS = GRAVAVEL / 'data' / 'precos'
PASTA_RESULTADOS = GRAVAVEL / 'results'

MOEDAS = ['BTC', 'ETH', 'SOL', 'BNB', 'DOGE']
SEMENTES = [42, 7, 123, 2024, 31415]
FRAC_VALIDACAO = 0.15
FRAC_TESTE = 0.15

# LSTM padrão da etapa 01 (escolhida à mão, só para comparar os alvos)
HP_LSTM = dict(janela=14, hidden=64, camadas=1, dropout=0.1, lr=1e-3, batch=32, epocas=100, paciencia=15)

# Transformer padrão da etapa 03 (escolhido à mão, ponto de partida da busca)
HP_TRANSFORMER = dict(arquitetura='Transformer', janela=14, d_modelo=32, cabecas=4, camadas=2, dropout=0.1,
                      lr=5e-4, batch=32, epocas=100, paciencia=15)

# Etapas 02 e 03: busca de hiperparâmetros
TRIALS_OPTUNA = 30
DOBRAS_CV = 5

# Etapa 04: textos e sentimento
ARQ_NOTICIAS = 'cryptonews.csv'                                  # colunas date, title, text
ARQ_TWEETS = 'dataset_52-person-from-2021-02-05_2023-06-12_21-34-17-266_with_sentiment.csv'  # colunas created_at, full_text
PADROES_MOEDA = {'BTC': r'bitcoin|\bbtc\b',
                 'ETH': r'ethereum|\beth\b|\bether\b',
                 'SOL': r'solana|\bsol\b',
                 'BNB': r'\bbnb\b|binance coin|bnb chain',
                 'DOGE': r'dogecoin|\bdoge\b'}
MODELO_SENTIMENTO = 'ElKulako/cryptobert'
MAX_TOKENS = 128

# Etapa 07: walk-forward
INICIO_WALK_FORWARD = '2022-05-01'                               # primeiro dia previsto
MESES_POR_BLOCO = 2                                              # a rede é retreinada a cada bloco
REGIMES = {'Queda 2022': ('2022-05-01', '2022-12-31'),
           'Lateral 2023': ('2023-01-01', '2023-09-30'),
           'Alta fim de 2023': ('2023-10-01', '2023-12-31')}
