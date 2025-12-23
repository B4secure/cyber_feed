import os
import re
import glob
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil

import feedparser
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG (can be overridden by GitHub Actions env vars)
# ---------------------------
PAST_DAYS = int(os.getenv("PAST_DAYS", "7"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "50"))
DUP_THRESHOLD = float(os.getenv("DUi P_THRESHOLD", "0.60"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# Your search library (unchanged)
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Dragonforce	"Dragonforce ransomware" OR "Dragonforce threat actor" OR "Dragonforce cyber attack"
Qilin	"Qilin ransomware" OR "Qilin threat actor" OR "Qilin cyber attack"
Akira	"Akira ransomware" OR "Akira threat actor" OR "Akira cyber attack"
Warlock	"Warlock ransomware" OR "Warlock threat actor" OR "Warlock cyber attack"
Beast	"Beast ransomware" OR "Beast threat actor" OR "Beast cyber attack"
Everest	"Everest ransomware" OR "Everest threat actor" OR "Everest cyber attack"
LockBit	"LockBit ransomware" OR "LockBit threat actor" OR "LockBit cyber attack"
Lynx	"Lynx ransomware" OR "Lynx threat actor" OR "Lynx cyber attack"
Play	"Play ransomware" OR "Play threat actor" OR "Play cyber attack"
RansomHub	"RansomHub ransomware" OR "RansomHub threat actor" OR "RansomHub cyber attack"
RunSomeWares	"Run Some Wares ransomware" OR "Run Some Wares threat actor" OR "Run Some Wares cyber attack"
TheGentlemen	"The Gentlemen ransomware" OR "The Gentlemen threat actor" OR "The Gentlemen cyber attack"
Incransom	"Incransom ransomware" OR "Incransom threat actor" OR "Incransom cyber attack"
Medusa	"Medusa ransomware" OR "Medusa threat actor" OR "Medusa cyber attack"
Obscura	"Obscura ransomware" OR "Obscura threat actor" OR "Obscura cyber attack"
Cl0p	"Cl0p ransomware" OR "Cl0p threat actor" OR "Cl0p cyber attack"
d4rk4rmy	"d4rk4rmy ransomware" OR "d4rk4rmy threat actor" OR "d4rk4rmy cyber attack"
Devman	"Devman ransomware" OR "Devman threat actor" OR "Devman cyber attack"
ElDorado	"El Dorado ransomware" OR "El Dorado threat actor" OR "El Dorado cyber attack"
Interlock	"Interlock ransomware" OR "Interlock threat actor" OR "Interlock cyber attack"
Killsec	"Killsec ransomware" OR "Killsec threat actor" OR "Killsec cyber attack"
Safepay	"Safepay ransomware" OR "Safepay threat actor" OR "Safepay cyber attack"
Spacebears	"Spacebears ransomware" OR "Spacebears threat actor" OR "Spacebears cyber attack"
Sinbobi	"Sinbobi ransomware" OR "Sinbobi threat actor" OR "Sinbobi cyber attack"
Direwolf	"Direwolf ransomware" OR "Direwolf threat actor" OR "Direwolf cyber attack"
Alphalocker	"Alphalocker ransomware" OR "Alphalocker threat actor" OR "Alphalocker cyber attack"
""".strip()


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


def filter_last_n_days(df, n_days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
            continue
        name, query = line.split("\t", 1)
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


def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []
    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q = r["raw_query"]
        rss = google_news_rss_url(q, past_days)
        feed = feedparser.parse(rss)

        for entry in feed.entries[:max_items]:
            out_rows.append(
                {
                    "search_name": name,
                    "search_query": q,
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "past_days": past_days,
                }
            )
    return pd.DataFrame(out_rows)


def latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def semantic_dedupe_csv(infile: str, out_clean: str, out_audit: str,
                        threshold: float, model_name: str) -> tuple[int, int]:
    df = pd.read_excel(infile)
    df["compare_text"] = df["title"].fillna("").astype(str)

    mask = df["compare_text"].str.len() > 0
    df_work = df[mask].copy().reset_index(drop=True)
    orig_idx = df.index[mask].to_numpy()

    if df_work.empty:
        df.drop(columns=["compare_text"], errors="ignore").to_excel(out_clean, index=False)
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return len(df), len(df)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        df_work["compare_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sim = cosine_similarity(emb, emb)
    n = sim.shape[0]

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    keep_work = set()
    audit_rows = []

    for g in groups.values():
        if len(g) == 1:
            keep_work.add(g[0])
            continue

        g_map = [(int(orig_idx[i]), i) for i in g]
        g_map.sort(key=lambda x: x[0])

        keep_orig, keep_i = g_map[0]
        keep_work.add(keep_i)

        for drop_orig, drop_i in g_map[1:]:
            audit_rows.append(
                {
                    "kept_original_row": keep_orig,
                    "dropped_original_row": int(drop_orig),
                    "similarity": float(sim[keep_i, drop_i]),
                    "kept_title": df.loc[keep_orig, "title"],
                    "dropped_title": df.loc[int(drop_orig), "title"],
                }
            )

    kept_orig_rows = {int(orig_idx[i]) for i in keep_work}
    drop_orig_rows = set(map(int, orig_idx.tolist())) - kept_orig_rows

    keep_mask = np.ones(len(df), dtype=bool)
    for r in drop_orig_rows:
        keep_mask[r] = False

    df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    df_clean.to_excel(out_clean, index=False, engine = "openpyxl")
    audit.to_excel(out_audit, index=False, engine = "openpyxl")
    return len(df), len(df_clean)


def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_days(results, n_days=PAST_DAYS)

    if not results.empty:
        results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)

    raw_results_file = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    
    results = results.apply(lambda s: s.dt.tz_localize(None) if hasattr(s, "dt") and getattr(s.dt, "tz", None) is not None else s)

    results.to_excel(raw_results_file, index=False, engine = "openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine = "openpyxl")

    # Dedupe the raw file we just created
    dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_csv(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )
    # Always keep a stable single file for automation
    latest = DATA_DIR / "latest_ransomware_news.xlsx"
    df_final = pd.read_excel(dedup_file)
    df_final.to_excel(latest, index=False, engine="openpyxl")
    print(f"Updated latest file: {latest}")


    print(f"Saved raw:   {raw_results_file} | rows={len(results)}")
    print(f"Saved audit: {audit_search_file} | searches={len(search_df)}")
    print(f"Dedupe: original={orig} cleaned={cleaned}")
    print(f"Saved dedup: {dedup_file}")
    print(f"Saved dedup audit: {dedup_audit}")


if __name__ == "__main__":
    main()


# %%










