import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

def fetch_company_news(ticker, company_name, days=7, save_dir="data/news"):
    """
    Fetch recent English-language news articles about a company or ticker symbol
    using NewsAPI.org and save them as JSON.
    """
    if not API_KEY:
        raise ValueError("⚠️ NEWS_API_KEY missing in your .env file")

    os.makedirs(save_dir, exist_ok=True)

    # Compute date range
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Build the API request URL
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={company_name} OR {ticker}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"from={from_date}&"
        f"pageSize=50&"
        f"apiKey={API_KEY}"
    )

    print(f"📰 Fetching {days}-day news for {company_name} ({ticker}) ...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return

    data = response.json()
    articles = data.get("articles", [])

    if not articles:
        print(f"⚠️ No articles found for {company_name}")
        return

    # Format and save
    out_path = os.path.join(save_dir, f"{ticker}_news.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(articles)} articles saved → {out_path}")

    # Return data for chaining
    return articles


if __name__ == "__main__":
    companies = {
        "AAPL": "Apple Inc",
        "NVDA": "Nvidia Corporation"
    }

    for ticker, name in companies.items():
        fetch_company_news(ticker, name, days=7)
