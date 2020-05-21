# Crypto-Explorer
GPT-2 And Twitter Sentiment Analysis Crypto Paper Trade Telegram Bot

This is just cheap code that runs gpt-2 that could be buggy. I just was playing around one night and wanted to do something fun. It doesn't do any active real trades, just paper because I don't know how to code websockets.

It runs a twitter scraper, generates GPT-2 samples on them in a time series, then a sentiment analysis on the output which then has a trade signal.


This code is probably garbage but feel free to play with it. Use at your own risk.]



Install tensorflow

Install the requirements

Download the model

Put your telegram bot key in the crypto.py file

run /start and (probably get to debugging.)

P.S. It might take 14-16G of ram on a CPU. ~6.5G on a GPU.



The commands are

```
/timer - trade every hour
/trade - trade on demand
/help - show help
```

