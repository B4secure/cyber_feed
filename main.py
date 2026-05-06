import os
import re
import json
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import feedparser
import pandas as pd
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG
# ---------------------------

RUN_MODE  = os.getenv("RUN_MODE", "daily").lower()

if RUN_MODE == "monthly":
    PAST_DAYS = int(os.getenv("PAST_DAYS_MONTHLY", "30"))
elif RUN_MODE == "weekly":
    PAST_DAYS = int(os.getenv("PAST_DAYS_WEEKLY", "7"))
else:
    PAST_DAYS = int(os.getenv("PAST_DAYS_DAILY", "1"))

MAX_ITEMS     = int(os.getenv("MAX_ITEMS", "50"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.65"))
MODEL_NAME    = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR = Path("data")
DOCS_DIR = Path("docs")
for d in [DATA_DIR, DOCS_DIR]:
    d.mkdir(exist_ok=True)


# ---------------------------
# SEARCH LIBRARY
# 8 queries covering all 6 report questions.
# Split by angle: named groups, UK/EU incidents, healthcare,
# critical infrastructure, vulnerabilities, law enforcement.
# Ransomware_Response uses compound phrases to avoid
# generic "arrested"/"disrupted" noise.
# ---------------------------

SEARCH_LIBRARY_TEXT = r"""
Ransomware_Activity	("LockBit" OR "ALPHV" OR "RansomHub" OR "Akira" OR "Black Basta" OR "Cl0p" OR "Medusa" OR "Scattered Spider" OR "DragonForce" OR "8Base" OR "ShinyHunters" OR "ransomware attack" OR "ransomware incident" OR "ransom demand" OR "data encrypted" OR "systems encrypted") AND (attack OR victim OR hospital OR government OR company OR "critical infrastructure" OR UK OR Europe OR "United States" OR disclosed OR confirmed)
Ransomware_Groups_New	("BlackSuit" OR "Hunters International" OR "Qilin" OR "INC Ransom" OR "Rhysida" OR "NoEscape" OR "Cactus" OR "Meow" OR "Embargo" OR "Fog" OR "Interlock" OR "SafePay" OR "Lynx" OR "Cicada3301" OR "RansomedVC") AND (attack OR victim OR campaign OR active OR claimed OR leaked OR posted OR operation OR arrested)
Ransomware_Incidents_UK_EU	(ransomware OR "cyber attack" OR "data breach") AND (NHS OR "local council" OR "local authority" OR "public sector" OR university OR school OR manufacturer OR retailer OR "law firm") AND (UK OR Britain OR England OR Scotland OR Wales OR Ireland OR Germany OR France OR Italy OR Spain OR Europe)
Ransomware_Tactics_Threats	("double extortion" OR "triple extortion" OR "RaaS" OR "ransomware-as-a-service" OR "initial access broker" OR "zero day exploit" OR "credential theft" OR "MFA bypass" OR "new ransomware" OR "ransomware variant" OR "AI-powered ransomware" OR "OT ransomware" OR "ICS ransomware" OR "wiper malware" OR "supply chain attack") AND (ransomware OR malware OR "threat actor" OR technique OR emerging OR advisory OR warning OR discovered)
Ransomware_Vulnerabilities	(vulnerability OR "zero day" OR CVE OR exploit OR patch OR "security flaw" OR "critical flaw") AND (ransomware OR malware OR "threat actor" OR "actively exploited" OR "exploited in the wild" OR "ransomware group") AND (CISA OR advisory OR warning OR alert OR disclosed)
Ransomware_Response	(Europol OR FBI OR "National Crime Agency" OR CISA OR "Department of Justice" OR Interpol OR "ransomware arrest" OR "hacker indicted" OR "cybercriminal arrested" OR "ransomware gang arrested" OR "ransomware operator charged" OR "ransomware infrastructure seized" OR "cyber operation dismantled" OR "ransomware advisory" OR "mitigation guidance") AND (ransomware OR cybercrime OR "cyber criminal" OR "threat actor" OR "ransomware gang" OR malware)
Ransomware_Healthcare	(ransomware OR "cyber attack" OR "data breach") AND (hospital OR healthcare OR "medical centre" OR clinic OR pharmacy OR "health service" OR "patient data" OR "electronic health") AND (attacked OR disrupted OR encrypted OR offline OR incident OR breach OR warning)
Ransomware_Critical_Infra	(ransomware OR "cyber attack") AND ("critical infrastructure" OR "energy sector" OR "power grid" OR "water utility" OR "transport network" OR "financial services" OR "telecoms" OR "government systems") AND (attacked OR disrupted OR warning OR incident OR threat OR advisory)
""".strip()


# ---------------------------
# HELPERS
# ---------------------------

def parse_published_dt(published_str: str):
    if not published_str:
        return None
    try:
        dt = dateparser.parse(published_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name, query = line.split("\t", 1)
        else:
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                name, query = parts[0], parts[1]
            else:
                rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
                continue
        rows.append({"search_name": name.strip(), "raw_query": query.strip()})
    return pd.DataFrame(rows)


def is_google_news_compatible(q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return False
    if q.startswith("http://") or q.startswith("https://"):
        return False
    if "to:" in q or q.startswith("@") or " @" in q:
        return False
    return True


def google_news_rss_url(query: str, past_days: int) -> str:
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={HL}&gl={GL}&ceid={CEID}"


# ---------------------------
# RSS COLLECTION
# ---------------------------

def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []
    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q    = r["raw_query"]
        rss  = google_news_rss_url(q, past_days)
        feed = feedparser.parse(rss)
        print(f"  {name}: {len(feed.entries)} raw entries")
        for entry in feed.entries[:max_items]:
            out_rows.append({
                "search_name":  name,
                "search_query": q,
                "title":        entry.get("title", ""),
                "published":    entry.get("published", ""),
                "link":         entry.get("link", ""),
                "past_days":    past_days,
                "source":       "google_rss",
            })
    return pd.DataFrame(out_rows)


# ---------------------------
# SEMANTIC DEDUPE — INTRA-TOPIC ONLY
# Removes near-identical articles within the same category.
# Same article in two different categories is kept in both.
# ---------------------------

def semantic_dedupe_within_topic(df: pd.DataFrame, threshold: float, model_name: str) -> tuple:
    if df.empty:
        return df.copy(), pd.DataFrame()

    df["title"] = df["title"].fillna("").astype(str)
    model      = SentenceTransformer(model_name)
    keep_rows  = []
    audit_rows = []

    for topic in sorted(df["search_name"].unique()):
        topic_df   = df[df["search_name"] == topic].copy().reset_index(drop=True)
        valid_mask = topic_df["title"].str.strip().str.len() > 0
        work_df    = topic_df[valid_mask].reset_index(drop=True)
        empty_df   = topic_df[~valid_mask].reset_index(drop=True)

        if work_df.empty:
            keep_rows.append(topic_df)
            continue

        emb = model.encode(work_df["title"].tolist(), normalize_embeddings=True, show_progress_bar=False)
        sim = cosine_similarity(emb, emb)
        n   = sim.shape[0]

        parent = list(range(n))
        rank   = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            if rank[ra] < rank[rb]:   parent[ra] = rb
            elif rank[ra] > rank[rb]: parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        kept_indices = []
        for members in groups.values():
            keep_i = min(members)
            kept_indices.append(keep_i)
            for drop_i in members:
                if drop_i == keep_i: continue
                audit_rows.append({
                    "search_name":   topic,
                    "kept_title":    work_df.loc[keep_i, "title"],
                    "dropped_title": work_df.loc[drop_i, "title"],
                    "similarity":    float(sim[keep_i, drop_i]),
                })

        kept_df = work_df.loc[sorted(kept_indices)].reset_index(drop=True)
        if not empty_df.empty:
            kept_df = pd.concat([kept_df, empty_df], ignore_index=True)
        keep_rows.append(kept_df)
        print(f"  [{topic}] {len(topic_df)} -> {len(kept_df)} (removed {len(topic_df) - len(kept_df)} dupes)")

    df_clean = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
    return df_clean, pd.DataFrame(audit_rows)


# ---------------------------
# MAIN
# ---------------------------

def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    print(f"\nRansomware Intelligence Pipeline")
    print(f"  RUN_MODE  = {RUN_MODE}")
    print(f"  PAST_DAYS = {PAST_DAYS}\n")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    print("Collecting from Google News RSS...")
    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_days(results, n_days=PAST_DAYS)

    if not results.empty:
        results = results.drop_duplicates(subset=["link", "search_name"]).reset_index(drop=True)

    if "published_dt_utc" in results.columns:
        results = results.drop(columns=["published_dt_utc"], errors="ignore")

    print(f"\nRaw: {len(results)} articles")

    print("\nDeduplicating within topics...")
    df_clean, df_audit = semantic_dedupe_within_topic(results, DUP_THRESHOLD, MODEL_NAME)

    # Main timestamped Excel
    output_file = DATA_DIR / f"ransomware_news_{ts}_past{PAST_DAYS}d.xlsx"
    df_clean.to_excel(output_file, index=False, engine="openpyxl")

    # Dedup audit
    audit_file = DATA_DIR / f"ransomware_dedup_audit_{ts}.xlsx"
    df_audit.to_excel(audit_file, index=False, engine="openpyxl")

    # Rolling latest — always overwritten, used by portal
    latest_file = DATA_DIR / "ransomware_latest.xlsx"
    df_clean.to_excel(latest_file, index=False, engine="openpyxl")

    # Portal feed JSON
    articles = []
    for _, row in df_clean.iterrows():
        articles.append({
            "search_name": str(row.get("search_name", "")),
            "title":       str(row.get("title", "")),
            "published":   str(row.get("published", "")),
            "link":        str(row.get("link", "")),
            "past_days":   int(row.get("past_days", PAST_DAYS)),
            "source":      str(row.get("source", "google_rss")),
        })

    payload = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "lookback_days": PAST_DAYS,
        "run_type":      f"{RUN_MODE.capitalize()} run",
        "feed_type":     "ransomware",
        "articles":      articles,
    }
    feed_path = DOCS_DIR / "ransomware_feed.json"
    with open(feed_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nResults by category:")
    for cat, count in df_clean["search_name"].value_counts().items():
        print(f"  {cat}: {count}")

    print(f"\nDone.")
    print(f"  Main Excel : {output_file}")
    print(f"  Latest     : {latest_file}")
    print(f"  Portal feed: {feed_path}")
    print(f"  Total      : {len(df_clean)} articles")


if __name__ == "__main__":
    main()
