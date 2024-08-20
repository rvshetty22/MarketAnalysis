import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import logging
import os
import time
import random
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import seaborn as sns
import matplotlib.pyplot as plt

# Initiate logfile
def setup_logging(logfile='logfile.log'):
    # Reset logfile settings
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
        encoding='utf-8'
    )

setup_logging('logfile.log')
logging.info('Starting Market Analysis')

# Sector URLs
sector_urls = {
    'Technology': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_technology/",
    'Healthcare': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_healthcare/",
    'Financial Services': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_financial-services/",
    'Consumer Cyclical': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_consumer-cyclical/",
    'Industrials': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_industrials/",
    'Communication Services': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_communication-services/",
    'Consumer Defensive': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_consumer-defensive/",
    'Energy': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_energy/",
    'Real Estate': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_real-estate/",
    'Basic Materials': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_basic-materials/",
    'Utilities': "https://finance.yahoo.com/screener/predefined/sec-ind_sec-largest-equities_utilities/"
}

# Returns top 25 companies based on market cap from yfinance
def get_top_companies(sector_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(sector_url, headers=headers)
    if response.status_code != 200:
        logging.error(f"Failed to retrieve data from {sector_url}. Status code: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    companies = [item.text for item in soup.find_all('a', class_='Fw(600) C($linkColor)')][:25]  # LImit of 25 companies
    
    if not companies:
        logging.warning(f"No companies found for URL: {sector_url}")
    return companies

# Creates a URL based on the stock ticker and metric name
def construct_url(ticker, metric_name):
    stock = yf.Ticker(ticker)
    stock_name = stock.info['shortName'].replace(' ', '-')
    ticker = ticker.replace('-','.')
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{stock_name}/{metric_name}"
    
    # 'https://www.macrotrends.net/stocks/charts/ES/Eversource-Energy-(D/B/A)/pe-ratio'

    return url

# Scrapes macrotrends and gathers relevant data based on url
def scrape_metric(url, metric_name):
    # Configure scraper
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service("C:\\Users\\rishi\\Downloads\\chromedriver-win64\\chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Add in delay for page to load
    driver.get(url)
    time.sleep(random.uniform(10, 15))

    # Parse the HTML soup for correct table
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, 'html.parser')
    with open("soup.html", "w", encoding="utf-8") as file:
        file.write(soup.prettify())
    table = soup.find('table')
    if table is None:
        return pd.DataFrame()

    html_content = str(table)
    html_io = StringIO(html_content)

    # Turn relevant data into a pd DataFrame
    df = pd.read_html(html_io)[0]
    df.columns = df.columns.droplevel(0) 
    df = df[['Date', df.columns[3]]]
    df.set_index('Date', inplace=True)  

    if metric_name == 'roe':
        df[df.columns[0]] = df[df.columns[0]].replace('%', '', regex=True).astype(float)

    return df

# Calls scraper for all metrics
def get_historical_metrics(ticker): 
    # Checks if metrics are already found for select ticker
    if ticker in metrics_cache:
        return metrics_cache[ticker]
    
    # Scrape the required metrics
    metrics = {
        'P/E': scrape_metric(construct_url(ticker, 'pe-ratio'), 'pe-ratio'),
        'P/B': scrape_metric(construct_url(ticker, 'price-book'), 'price-book'),
        'D/E': scrape_metric(construct_url(ticker, 'debt-equity-ratio'), 'debt-equity-ratio'),
        'P/FCF': scrape_metric(construct_url(ticker, 'price-fcf'),'price-fcf'),
        'ROE': scrape_metric(construct_url(ticker, 'roe'),'roe'),
    }

    if any(v.empty for v in metrics.values()):
        return pd.DataFrame()
        
    # Combines all metric data and stores in cache
    all_metrics = pd.concat(metrics.values(), axis=1, join='outer')
    all_metrics.insert(0, 'Ticker', ticker)
    metrics_cache[ticker] = all_metrics
    return all_metrics

# Calculates percent increase in price for a stock quarterly
def fetch_quarterly_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1d')
        quarterly_data = data['Close'].resample('QE').last()
        return quarterly_data
    except Exception as e:
        return pd.Series()

# Percent growth calculation
def calculate_growth(data):
    data = data.pct_change().dropna() * 100
    return data

# Finds closest data when metric dates (macrotrends) do not align with growth data (yfinance)
def get_closest_date(date, all_metrics):
    closest_date = min(all_metrics.index, key=lambda x: abs(x - date))
    return closest_date

# Returns a table with all ticker metrics and growth data
def prepare_data(ticker, start_date, end_date):
    price_data = fetch_quarterly_data(ticker, start_date, end_date)
    growth_data = calculate_growth(price_data)
    if growth_data.index.isna().all():
        logging.warning(f'Skipping data collection for: {ticker}')
        return pd.DataFrame()
    growth_data.index = growth_data.index.date    
    all_metrics = get_historical_metrics(ticker)
    if all_metrics.empty or all_metrics.index.isna().all():
        logging.warning(f'No valid data or dates found for ticker: {ticker}')
        logging.warning(f'Skipping data collection for: {ticker}')
        return pd.DataFrame()
    
    all_metrics.index = pd.to_datetime(all_metrics.index).date

    metrics_list = []
    for date in growth_data.index:
        if date in all_metrics.index:
            metrics_row = all_metrics.loc[date].to_dict()
        else:
            closest_date = get_closest_date(date, all_metrics)
            metrics_row = all_metrics.loc[closest_date].to_dict()
        metrics_row['Date'] = date
        metrics_row['Quarterly_Growth'] = growth_data.loc[date]
        metrics_list.append(metrics_row)
    
    return pd.DataFrame(metrics_list)

# Appends stock data to sector file
def write_ticker_data(ticker_data, filename, include_header):
    columns_order = ['Ticker', 'Date'] + [col for col in ticker_data.columns if col not in ['Ticker', 'Date']]
    ticker_data = ticker_data[columns_order]

    with open(filename, 'a') as file:
        file.write(ticker_data.to_string(index=False, header=include_header, float_format='{:.2f}'.format))
        file.write("\n")

# Reads stock data from sector file
def read_sector_data(filename):
    sector_data = []
    headers = ['Ticker', 'Date', 'PE Ratio', 'Price to Book Ratio', 'Debt to Equity Ratio', 'Price to FCF Ratio', 'Return on Equity', 'Quarterly_Growth']
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(('Ticker')):
                continue
            else:
                values = line.split()
                sector_data.append(values)
    
    # Convert the sector data into a DataFrame
    if sector_data and headers:
        df = pd.DataFrame(sector_data, columns=headers)
        numeric_columns = headers[2:]  # Exclude 'Ticker' and 'Date'
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df['Ticker'] = df['Ticker'].astype(str)

        return df

    # Return an empty DataFrame if no data is found
    return pd.DataFrame()

# Collects all sector data by reading or scraping
def collect_sector_data(start_date, end_date, collect):
    combined_data = {}
    # For each sector checks to see if the file is already there
    # It will gather metrics data if missing or incomplete
    for sector, url in sector_urls.items():
        filename = f"{sector.replace(' ', '_').lower()}_metrics.txt"
        logging.info(f'Collecting data for sector: {sector}')
        sector_data = pd.DataFrame()
        if os.path.exists(filename):
            logging.info(f'Reading data from {filename}')
            sector_data = read_sector_data(filename)
            unique_tickers = sector_data['Ticker'].nunique()
            if unique_tickers == 25:
                logging.info(f'Data found for ALL tickers')
            else:
                logging.info(f'Data found for {unique_tickers} tickers')
        else:
            unique_tickers = 0
        
        if unique_tickers < 25 and collect:
            tickers = get_top_companies(url)
            header_written = os.path.exists(filename) and unique_tickers > 0

            if not sector_data.empty:
                unique_ticker_set = set(sector_data['Ticker'].unique())
            else:
                unique_ticker_set = set()

            for ticker in tickers:
                if ticker not in unique_ticker_set:
                    ticker_data = prepare_data(ticker, start_date, end_date)
                    if ticker_data.empty:
                        continue
                    write_ticker_data(ticker_data, filename, include_header=not header_written)
                    header_written = True
                    sector_data = pd.concat([sector_data, ticker_data], ignore_index=False)
                    unique_ticker_set.add(ticker)
                    logging.info(f'Data collected for: {ticker}')
            logging.info(f'Data collection complete for sector: {sector}')

        combined_data[sector] = sector_data

    return combined_data


start_date = '2013-12-31'
end_date = '2023-12-31'
metrics_cache = {}
collect = False
all_sector_data = collect_sector_data(start_date, end_date, collect)
logging.info('Data gathered for all sectors.')

def train_predict_growth(sector_data, sector_name):
    logging.info(f'Creating Models for sector: {sector_name}')
    # Ensure no missing values
    sector_data = sector_data.dropna()
    sector_data['Date'] = pd.to_datetime(sector_data['Date'], errors='coerce')
    sector_data.loc[:, 'Month'] = sector_data['Date'].dt.month
    sector_data.loc[:, 'Year'] = sector_data['Date'].dt.year
    
    # Prepare features and target
    X = sector_data[['PE Ratio', 'Price to Book Ratio', 'Debt to Equity Ratio', 'Price to FCF Ratio', 'Return on Equity', 'Month', 'Year']]
    y = sector_data['Quarterly_Growth']
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    models = {
        'Linear': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'Polynomial': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_model_type = None
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # Train and evaluate Linear, Lasso, and Polynomial regression models
    for name, model in models.items():
        if model is None:
            continue
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        logging.info(f'  {name} Regression - R²: {r2:.4f}, MSE: {mse:.4f}')
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_type = name

        # Plot the predictions
        sns.lineplot(x=range(len(y_pred)), y=y_pred, label=f'{name} Prediction - R²: {r2:.4f}, MSE: {mse:.4f}')

    # Plot actual values
    sns.scatterplot(x=range(len(y_test)), y=y_test, label='Actual', color='blue')

    # Add title, labels, and legend
    plt.title(f'{sector_name} - Model Comparisons')
    plt.xlabel('Data Point')
    plt.ylabel('Quarterly Growth')
    plt.legend()

    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f'{sector_name}_model_comparisons.png')
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f'Best Model: {best_model_type}')
    
    # Save the best model and its type
    return best_model, best_model_type, scaler




# Collect metrics for all companies and prepare models
sector_models = {}
for sector_name, sector_data in all_sector_data.items():
    model = train_predict_growth(sector_data, sector_name)
    sector_models[sector_name] = model

# User allocation and investment logic
user_allocation = {
    'Technology': 0.50,
    'Financial Services': 0.0,
    'Healthcare': 0.0,
    'Consumer Cyclical': 0.0,
    'Industrials': 0.0,
    'Communication Services': 0.0,
    'Consumer Defensive': 0.0,
    'Energy': 0.0,
    'Real Estate': 0.10,
    'Basic Materials': 0.0,
    'Utilities': 0.40
}

# Budget and allocation
budget = 1000
sector_investment = {sector: allocation * budget for sector, allocation in user_allocation.items()}

logging.info(f"Total Budget: ${budget}")
logging.info(f"Sector Investments: {sector_investment}")

ratio_columns = ['PE Ratio', 'Price to Book Ratio', 'Debt to Equity Ratio', 'Price to FCF Ratio', 'Return on Equity']
final_selection = {}
for sector, investment in sector_investment.items():
    logging.info(f"Processing sector: {sector}")
    model, model_type, scaler = sector_models[sector]
    
    # Retrieve the most recent data for each stock in the sector
    latest_data = all_sector_data[sector].groupby('Ticker').apply(lambda df: df.iloc[-1]).reset_index(drop=True)
    latest_data = latest_data[~latest_data.isin([float('inf'), -float('inf')]).any(axis=1)]
    latest_data['Date'] = pd.to_datetime(latest_data['Date'], errors='coerce')
    
    # Prepare the latest data features
    X_latest = latest_data[ratio_columns]
    ratio_means = X_latest[ratio_columns].mean()
    # Impute NaN values with the average sector values
    X_latest_filled = X_latest.copy()
    X_latest_filled[ratio_columns] = X_latest_filled[ratio_columns].fillna(ratio_means)
    X_latest_filled.loc[:, 'Month'] = latest_data['Date'].dt.month
    X_latest_filled.loc[:, 'Year'] = latest_data['Date'].dt.year
    X_latest_filled.set_index(latest_data['Ticker'], inplace=True)
    
    # Standardize using the same scaler as used during training
    X_latest_scaled = scaler.transform(X_latest_filled)
    
    predicted_growth = np.zeros(len(latest_data))
    if model_type == 'Polynomial':
        predicted_growth = model.predict(X_latest_scaled)
    else:
        predicted_growth = model.predict(X_latest_scaled)     

    latest_data['Predicted_Growth'] = predicted_growth
    sector_projected_growth = latest_data['Predicted_Growth'].mean()
    logging.info(f"Average Projected Growth: {sector_projected_growth:.2f}")
  
    
    # Select top stocks based on predicted growth
    sorted_stocks = latest_data.sort_values(by='Predicted_Growth', ascending=False).head(3)
    top_3_stocks = sorted_stocks.head(3)
    top_3_price = yf.download(top_3_stocks['Ticker'].tolist(), period='1d')['Adj Close'].iloc[-1]
    top_stocks_prices = top_3_price.to_dict()
    logging.info("Top 3 Stocks:")
    for stock in top_3_stocks.itertuples(index=False):
        price = top_stocks_prices[stock.Ticker]
        logging.info(f"  Stock: {stock.Ticker}, Price: ${price:.2f}, Predicted Growth: {stock.Predicted_Growth:.2f}")
    
    # Retrieve top stocks and their prices
    stock_price = yf.download(sorted_stocks['Ticker'].tolist(), period='1d')['Adj Close'].iloc[-1]
    top_stocks_prices = stock_price.to_dict()
    stock_allocations = {}
    for stock in sorted_stocks.itertuples():
        if len(stock_allocations) >= 3:
            break
        price = top_stocks_prices[stock.Ticker]
        num_shares = int(investment // price)
        if num_shares > 0:  # Only allocate if num_shares is greater than 0
            stock_allocations[stock.Ticker] = {
                'shares': num_shares,
                'price': price,
                'total': num_shares * price,
                'predicted_growth': stock.Predicted_Growth
            }
    
    final_selection[sector] = stock_allocations

# Output the final stock selection
os.system('cls')
print(f"Total Budget: ${budget}")
allocation_str = " ".join([f"{sector}: {allocation * 100:.1f}%" for sector, allocation in user_allocation.items()])
print(f"{allocation_str}\n")
for sector, stocks in final_selection.items():
    allocated_money = sector_investment.get(sector, 0)
    print(f"${allocated_money} was alloted to {sector}")
    if stocks:
        print(f"Top stocks:")
        for stock, details in stocks.items():
            print(f"  Stock: {stock}, Shares: {details['shares']}, Price: ${details['price']:.2f}, Total: ${details['total']:.2f}, Projected Quarterly Growth: {details['predicted_growth']:.2f}")
        print()  # Add an extra newline for better readability
    else:
        print(f"No stocks found matching allocation\n")

logging.info("Market Analysis Finished")

