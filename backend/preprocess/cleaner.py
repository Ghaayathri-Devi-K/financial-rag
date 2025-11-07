import os
import json
import pandas as pd
import re

def clean_text(text: str) -> str:
    """Remove extra spaces, special chars, and normalize text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,$%\-–’'\"() ]", "", text)
    return text.strip()

def clean_sec_filings(src_dir="data/sec_filings", out_dir="data/clean/sec"):
    os.makedirs(out_dir, exist_ok=True)
    for ticker in os.listdir(src_dir):
        ticker_path = os.path.join(src_dir, ticker)
        all_text = ""
        for file in os.listdir(ticker_path):
            if file.endswith(".txt"):
                with open(os.path.join(ticker_path, file), "r", encoding="utf-8", errors="ignore") as f:
                    all_text += f.read() + "\n"
        cleaned = clean_text(all_text)
        with open(os.path.join(out_dir, f"{ticker}_clean.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"✅ Cleaned SEC filings for {ticker}")

def clean_news(src_dir="data/news", out_dir="data/clean/news"):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if not file.endswith(".json"): continue
        with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = []
        for art in data:
            text = f"{art.get('title', '')} {art.get('description', '')} {art.get('content', '')}"
            articles.append(clean_text(text))
        merged = "\n\n".join(articles)
        out_file = os.path.join(out_dir, file.replace(".json", "_clean.txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"📰 Cleaned {len(articles)} articles from {file}")

def clean_market(src_dir="data/market", out_dir="data/clean/market"):
    os.makedirs(out_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if not file.endswith(".csv"): continue
        df = pd.read_csv(os.path.join(src_dir, file))
        text = df.tail(30).to_string(index=False)
        out_file = os.path.join(out_dir, file.replace(".csv", "_clean.txt"))
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"📈 Cleaned market data for {file}")

if __name__ == "__main__":
    clean_sec_filings()
    clean_news()
    clean_market()
    print("🎯 All raw data cleaned → data/clean/")
