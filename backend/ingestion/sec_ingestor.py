import os
import requests
from bs4 import BeautifulSoup

def fetch_sec_filings(ticker, form_type="10-K", limit=3):
    """
    Fetches the latest SEC filings (10-K, 10-Q, etc.) for a given ticker
    directly from the EDGAR search endpoint.
    Saves them as text files in data/sec_filings/<ticker>/.
    """

    base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
    headers = {"User-Agent": "kgd.careers@gmail.com"}  # ✅ use your real email here
    params = {
        "action": "getcompany",
        "owner": "exclude",
        "count": limit,
        "CIK": ticker,
        "type": form_type,
        "output": "atom",
    }

    print(f"📥 Fetching {limit} {form_type} filings for {ticker}...")
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch listings for {ticker} (status {response.status_code})")
        return

    soup = BeautifulSoup(response.content, "lxml")
    entries = soup.find_all("entry")

    ticker_dir = os.path.join("data", "sec_filings", ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    for i, entry in enumerate(entries[:limit], 1):
        link = entry.find("link")["href"]
        txt_url = link.replace("-index.htm", ".txt")
        txt_resp = requests.get(txt_url, headers=headers)
        if txt_resp.status_code == 200:
            file_path = os.path.join(ticker_dir, f"{ticker}_{form_type}_{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(txt_resp.text)
            print(f"✅ Saved {form_type} filing #{i} for {ticker}")
        else:
            print(f"⚠️ Could not download text for {ticker} filing #{i}")

    print(f"📂 All {ticker} filings stored at {ticker_dir}\n")


if __name__ == "__main__":
    os.makedirs("data/sec_filings", exist_ok=True)
    for t in ["AAPL", "NVDA"]:
        fetch_sec_filings(t, "10-K", limit=2)
