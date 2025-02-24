import asyncio
import os
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import sqlite3
import time
from datetime import datetime
import ta
import requests  

# Tool Functions
def get_stock_data(symbol: str, period: str = '1mo'):
    for attempt in range(3):
        try:
            data = yf.download(symbol, period=period, auto_adjust=False, progress=False)
            if not data.empty:
                return data
            else:
                print(f"No data found for {symbol} on attempt {attempt+1}.")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {symbol}: {e}")
        time.sleep(1)
    return None

def plot_stock_data(data: pd.DataFrame, symbol: str) -> str:
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], marker='o', linestyle='-', color='navy')
    plt.title(f'Stock Price for {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price (USD)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    filename = f'{symbol}_stock_plot.png'
    plt.savefig(filename)
    plt.close()
    return filename

def plot_stock_with_sma(data: pd.DataFrame, symbol: str, window: int) -> str:
    data_with_sma = data.copy()
    data_with_sma['SMA'] = data_with_sma['Close'].rolling(window=window).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(data_with_sma.index, data_with_sma['Close'], marker='o', linestyle='-', label='Close', color='navy')
    plt.plot(data_with_sma.index, data_with_sma['SMA'], linestyle='--', label=f'{window}-day SMA', color='orange')
    plt.title(f'{symbol} Price with {window}-day SMA', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'{symbol}_sma_{window}.png'
    plt.savefig(filename)
    plt.close()
    return filename

def compare_stocks(symbol1: str, symbol2: str, period: str = '1mo') -> str:
    data1 = get_stock_data(symbol1, period)
    data2 = get_stock_data(symbol2, period)
    if data1 is None or data2 is None:
        return "Failed to fetch data for one or both symbols."
    plt.figure(figsize=(10, 5))
    plt.plot(data1.index, data1['Close'], marker='o', linestyle='-', label=symbol1)
    plt.plot(data2.index, data2['Close'], marker='s', linestyle='-', label=symbol2)
    plt.title(f'Comparison: {symbol1} vs {symbol2}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'compare_{symbol1}_{symbol2}.png'
    plt.savefig(filename)
    plt.close()
    return filename

def get_stock_info(symbol: str) -> str:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    if not info:
        return f"No info found for {symbol}."
    price = info.get('regularMarketPrice', 'N/A')
    prev_close = info.get('previousClose', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    summary = info.get('longBusinessSummary', 'No summary available.')
    summary = summary[:250] + "..." if len(summary) > 250 else summary
    return (f"Ticker: {symbol}\nCurrent Price: {price}\nPrevious Close: {prev_close}\n"
            f"Market Cap: {market_cap}\nSummary: {summary}")

def plot_candlestick(symbol: str, period: str = '1mo') -> str:
    data = get_stock_data(symbol, period)
    if data is None or data.empty:
        return f"No data for {symbol}."
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        return f"Data for {symbol} is missing required columns: {missing}"
    data = data.copy()
    data[required_cols] = data[required_cols].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=required_cols, inplace=True)
    if data.empty:
        return f"Not enough valid data to plot for {symbol}."
    filename = f'{symbol}_candlestick.png'
    mpf.plot(data, type='candle', style='charles', title=f'Candlestick Chart for {symbol}',
             volume=True, savefig=filename)
    return filename

def plot_tech_indicators(symbol: str, period: str = '1mo') -> str:
    data = get_stock_data(symbol, period)
    if data is None or data.empty:
        return f"No data for {symbol}."
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
    macd_indicator = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='blue')
    ax1.set_title(f'{symbol} Close Price')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(data.index, data['RSI'], label='RSI', color='green')
    ax2.plot(data.index, data['MACD'], label='MACD', color='magenta')
    ax2.plot(data.index, data['MACD_Signal'], label='MACD Signal', color='orange')
    ax2.set_title(f'{symbol} Technical Indicators')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    filename = f'{symbol}_tech_indicators.png'
    plt.savefig(filename)
    plt.close()
    return filename

def store_stock_data_db(symbol: str, data: pd.DataFrame) -> str:
    db_name = 'stock_data.db'
    conn = sqlite3.connect(db_name)
    data_reset = data.reset_index()
    data_reset['Date'] = data_reset['Date'].astype(str)
    data_reset.to_sql(symbol, conn, if_exists='replace', index=False)
    conn.close()
    return f"Data for {symbol} stored in {db_name}."

def fetch_stock_data_db(symbol: str) -> str:
    db_name = 'stock_data.db'
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query(f"SELECT * FROM '{symbol}'", conn)
        conn.close()
        if df.empty:
            return f"No stored data for {symbol}."
        else:
            return df.head(5).to_string(index=False)
    except Exception as e:
        conn.close()
        return f"Error fetching data for {symbol}: {e}"

# News Function using NewsAPI
def get_stock_news(symbol: str) -> str:
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return "News API key not set. Please set the NEWS_API_KEY environment variable."
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error fetching news: {response.status_code}"
    articles = response.json().get("articles", [])
    if not articles:
        return f"No news found for {symbol}."
    headlines = []
    for article in articles:
        headline = article.get("title", "No Title")
        link = article.get("url", "")
        headlines.append(f"- {headline}\n  Link: {link}")
    return f"Recent news for {symbol}:\n" + "\n".join(headlines)


# Specialized Agent Classes (using LangChain's AssistantAgent)

from autogen_agentchat.agents import AssistantAgent

class PlotAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(name="PlotAgent", model_client=model_client)
    async def run_task(self, message: str) -> str:
        tokens = message.strip().split()
        cmd = tokens[0].lower()
        if cmd == 'plot' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            data = get_stock_data(ticker)
            if data is None or data.empty:
                return f"No data found for {ticker}."
            filename = plot_stock_data(data, ticker)
            return f"Plot for {ticker} saved as '{filename}'."
        elif cmd == 'sma' and len(tokens) >= 3:
            ticker = tokens[1].upper()
            try:
                window = int(tokens[2])
            except ValueError:
                return "Invalid window. Provide an integer."
            data = get_stock_data(ticker)
            if data is None or data.empty:
                return f"No data found for {ticker}."
            filename = plot_stock_with_sma(data, ticker, window)
            return f"SMA plot for {ticker} ({window}-day) saved as '{filename}'."
        elif cmd == 'compare' and len(tokens) >= 3:
            ticker1 = tokens[1].upper()
            ticker2 = tokens[2].upper()
            filename = compare_stocks(ticker1, ticker2)
            return f"Comparison plot for {ticker1} vs {ticker2} saved as '{filename}'."
        else:
            return "PlotAgent: command not recognized."

class InfoAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(name="InfoAgent", model_client=model_client)
    async def run_task(self, message: str) -> str:
        tokens = message.strip().split()
        cmd = tokens[0].lower()
        if cmd == 'info' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            return get_stock_info(ticker)
        elif cmd == 'candle' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            filename = plot_candlestick(ticker)
            return f"Candlestick chart for {ticker} saved as '{filename}'."
        elif cmd == 'tech' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            filename = plot_tech_indicators(ticker)
            return f"Technical indicators plot for {ticker} saved as '{filename}'."
        else:
            return "InfoAgent: command not recognized."

class StoreAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(name="StoreAgent", model_client=model_client)
    async def run_task(self, message: str) -> str:
        tokens = message.strip().split()
        cmd = tokens[0].lower()
        if cmd == 'store' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            data = get_stock_data(ticker)
            if data is None or data.empty:
                return f"No data found for {ticker}."
            return store_stock_data_db(ticker, data)
        elif cmd == 'fetch' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            return fetch_stock_data_db(ticker)
        else:
            return "StoreAgent: command not recognized."

class NewsAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(name="NewsAgent", model_client=model_client)
    async def run_task(self, message: str) -> str:
        tokens = message.strip().split()
        if tokens[0].lower() == 'news' and len(tokens) >= 2:
            ticker = tokens[1].upper()
            return get_stock_news(ticker)
        else:
            return "NewsAgent: command not recognized."


# Multi-Agent Orchestrator

class MultiAgentOrchestrator:
    def __init__(self, model_client):
        self.plot_agent = PlotAgent(model_client)
        self.info_agent = InfoAgent(model_client)
        self.store_agent = StoreAgent(model_client)
        self.news_agent = NewsAgent(model_client)
    
    async def route_command(self, command: str) -> str:
        tokens = command.strip().split()
        if not tokens:
            return "No command provided."
        cmd = tokens[0].lower()
        if cmd in ['plot', 'sma', 'compare']:
            return await self.plot_agent.run_task(command)
        elif cmd in ['info', 'candle', 'tech']:
            return await self.info_agent.run_task(command)
        elif cmd in ['store', 'fetch']:
            return await self.store_agent.run_task(command)
        elif cmd == 'news':
            return await self.news_agent.run_task(command)
        else:
            return "Command not recognized by any agent."


# Main Execution

async def main():
    # Get Gemini API key and model from environment variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Please set the GOOGLE_GEMINI_API_KEY environment variable.")
    gemini_model = os.environ.get("GOOGLE_GEMINI_MODEL_ID") or "default-gemini-model-id"
    
    from google_gemini_llm import GoogleGeminiLLM
    gemini_client = GoogleGeminiLLM(api_key=gemini_api_key, model=gemini_model)
    orchestrator = MultiAgentOrchestrator(gemini_client)
    
    print("Multi-Agent Finance System with Google Gemini is ready.")
    print("Available commands:")
    print("  PlotAgent:   plot <TICKER>, sma <TICKER> <WINDOW>, compare <TICKER1> <TICKER2>")
    print("  InfoAgent:   info <TICKER>, candle <TICKER>, tech <TICKER>")
    print("  StoreAgent:  store <TICKER>, fetch <TICKER>")
    print("  NewsAgent:   news <TICKER>")
    print("Type 'exit' to quit.\n")
    
    while True:
        command = input("Enter command: ")
        if command.lower().strip() == "exit":
            break
        response = await orchestrator.route_command(command)
        print(response)

if __name__ == "__main__":
    asyncio.run(main())
