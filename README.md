# Finance AI Agent Bot

This is an enhanced Finance AI Agent built in Python that can:

- Fetch and plot stock price data.
- Plot a Simple Moving Average (SMA) overlay.
- Compare two stocks on the same chart.
- Retrieve basic stock info.
- Plot candlestick charts using mplfinance.
- Compute and plot technical indicators (RSI & MACD).
- Store stock data in a local SQLite database.
- Retrieve recent news headlines from Yahoo Finance.

## Features

- **plot <TICKER>**: Plots the closing price chart.
- **sma <TICKER> <WINDOW>**: Plots the closing price with a specified SMA window.
- **compare <TICKER1> <TICKER2>**: Compares two stocks on the same chart.
- **info <TICKER>**: Retrieves basic market info.
- **candle <TICKER>**: Plots a candlestick chart.
- **tech <TICKER>**: Plots technical indicators (RSI & MACD).
- **store <TICKER>**: Stores stock data in a local SQLite database.
- **fetch <TICKER>**: Retrieves stored stock data summary.
- **news <TICKER>**: Gets recent news headlines.


