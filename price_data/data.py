import yfinance as yf
import pandas as pd
import os

def fetch_data(download=True, start_date=None, interval=None, univ=None, file_name='price_volume_data.pkl'):
    current_folder = os.path.dirname(__file__)
    file_path = os.path.join(current_folder, file_name)

    if download:
        univ = [# --- TECHNOLOGY ---
                'MSFT', 'AAPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'ADBE', 'TXN', 'NVDA', 'QCOM',
                'AMAT', 'ADI', 'MU', 'LRCX', 'KLAC', 'AMD', 'APH', 'GLW', 'HPQ', 'MSI',
                'ADP', 'PAYX', 'FISV', 'FIS', 'CTSH', 'INTU', 'ADSK', 'SNPS', 'CDNS', 'MCHP',
                'STM', 'UMC', 'TSM', 'ASX', 'TEL', 'TER', 'NTAP', 'STX', 'WDC', 'ZBRA',
                'TRMB', 'TYL', 'AKAM', 'VRSN',

                # --- HEALTHCARE ---
                'JNJ', 'PFE', 'MRK', 'LLY', 'UNH', 'ABT', 'BMY', 'AMGN', 'GILD', 'BIIB',
                'SYK', 'MDT', 'BAX', 'BDX', 'CVS', 'CI', 'HUM', 'MCK', 'CAH', 'LH',
                'TMO', 'DHR', 'ISRG', 'EW', 'BSX', 'ZBH', 'STE', 'COO', 'HOLX', 'XRAY',
                'DGX', 'CNC', 'MOH', 'A', 'MTD', 'WAT', 'TECH', 'BIO',
                'RGEN', 'VRTX', 'REGN', 'INCY',

                # --- FINANCIALS ---
                'JPM', 'BAC', 'WFC', 'C', 'AXP', 'GS', 'MS', 'USB', 'BK', 'STT',
                'PGR', 'ALL', 'AIG', 'HIG', 'TRV', 'MMC', 'AON', 'BEN', 'SCHW', 'MCO',
                'SPGI', 'PNC', 'TFC', 'KEY', 'FITB', 'MTB', 'HBAN', 'RF', 'CMA', 'ZION',
                'L', 'CINF', 'WRB', 'AFL', 'PRU', 'MET', 'PFG', 'LNC', 'UNM',
                'RJF', 'SEIC', 'TROW', 'IVZ', 'AMG', 'BLK', 'ICE', 'CME', 'NDAQ', 'JKHY',
                'BRO', 'AJG', 'WTM', 'FAF',

                # --- CONSUMER STAPLES ---
                'KO', 'PEP', 'PG', 'WMT', 'COST', 'CL', 'MO', 'SYY', 'K', 'GIS',
                'HSY', 'CLX', 'MKC', 'TSN', 'CAG', 'TAP', 'EL', 'ADM', 'HRL', 'SJM',
                'KMB', 'CPB', 'SBUX', 'KR', 'DG', 'DLTR', 'TGT',
                'CHD', 'STZ', 'BF-B', 'NWL',

                # --- CONSUMER DISCRETIONARY ---
                'HD', 'LOW', 'MCD', 'NKE', 'F', 'DIS', 'TJX', 'VFC', 'YUM', 'DRI',
                'CMG', 'MAR', 'HLT', 'CCL', 'RCL', 'HAS', 'MAT', 'BBY', 'GPC',
                'AZO', 'ORLY', 'KMX', 'ROST', 'LB', 'M', 'KSS', 'DDS',
                'LEG', 'MHK', 'WHR', 'LEN', 'PHM',

                # --- INDUSTRIALS ---
                'GE', 'BA', 'CAT', 'HON', 'LMT', 'RTX', 'GD', 'MMM', 'UNP', 'FDX',
                'UPS', 'DE', 'EMR', 'ITW', 'ETN', 'PH', 'DOV', 'CMI', 'PCAR', 'NSC',
                'CSX', 'GWW', 'FAST', 'VMI', 'RSG', 'WM', 'CTAS', 'GPN', 'EFX', 'JCI',
                'TXT', 'NOC', 'LHX', 'HII', 'TDG', 'AME', 'ROK', 'SWK', 'SNA', 'MAS',

                # --- ENERGY & MATERIALS ---
                'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'VLO', 'OXY', 'DVN',
                'EOG', 'APA', 'BKR', 'NEM', 'FCX', 'APD', 'ECL', 'SHW', 'PPG',
                'LYB', 'DOW', 'DD', 'IP', 'NUE',

                # --- UTILITIES ---
                'NEE', 'DUK', 'SO', 'AEP', 'ED', 'PEG', 'XEL', 'EIX', 'ETR', 'D',
                'WEC', 'ES', 'AWK', 'SRE', 'FE', 'CMS', 'DTE', 'PPL', 'CNP', 'NI',

                # --- REAL ESTATE ---
                'PLD', 'SPG', 'PSA', 'O', 'VTR', 'BXP', 'AVB', 'EQR', 'ESS',
                'MAA', 'UDR', 'HST', 'VNO', 'SLG']

        # Download new data to local
        print(f'Downloading data to {file_path}...')
        raw_data = yf.download(tickers=univ,
                               start=start_date,
                               interval=interval,
                               auto_adjust=True)
        raw_data[['Close', 'Volume']].to_pickle(file_path)

    price_volume_data = pd.read_pickle(file_path)
    price_data = price_volume_data['Close'].ffill()
    rets = price_data.pct_change()

    return rets