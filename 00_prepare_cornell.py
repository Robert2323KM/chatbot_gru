"""
Descarga el corpus Cornell, extrae pares (Q,A) limpios
y guarda en `data/pairs.tsv` (tab-separated).
"""

import os, zipfile, urllib.request, re, random, csv, json
from pathlib import Path

random.seed(42)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
ZIP_URL  = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
ZIP_PATH = DATA_DIR/"cornell.zip"

if not (DATA_DIR/"cornell movie-dialogs corpus").exists():
    print("▸ descargando corpus …")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z: z.extractall(DATA_DIR)
    ZIP_PATH.unlink()

BASE = DATA_DIR/"cornell movie-dialogs corpus"
lines_f = BASE/"movie_lines.txt"
conv_f  = BASE/"movie_conversations.txt"

# ─── lines a diccionario ─────────────────────────────────────
id2line = {}
with open(lines_f, encoding="latin-1") as f:
    for row in f:
        _id, *_rest, txt = row.strip().split(" +++$+++ ")
        id2line[_id] = txt

# ─── conversaciones → pares Q,A ──────────────────────────────
pairs = []
with open(conv_f, encoding="latin-1") as f:
    for row in f:
        line_ids = eval(row.strip().split(" +++$+++ ")[-1])
        for i in range(len(line_ids)-1):
            q, a = id2line[line_ids[i]], id2line[line_ids[i+1]]
            pairs.append((q, a))

# limpieza ligera
def norm(t:str)->str:
    t = re.sub(r"[^a-zA-Z0-9.!?]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()

pairs = [(norm(q), norm(a)) for q,a in pairs
         if 2<=len(q.split())<=20 and 2<=len(a.split())<=20]

random.shuffle(pairs)
with open(DATA_DIR/"pairs.tsv","w",newline='',encoding="utf-8") as f:
    wr = csv.writer(f, delimiter="\t")
    wr.writerows(pairs)

print(f"Pairs listos → {len(pairs):,} líneas.")