# build_pairs.py ─ genera formatted_movie_lines.txt ───────────
import os, re, zipfile, urllib.request, csv, unicodedata

URL  = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
ZIP  = "cornell.zip"
ROOT = "cornell movie-dialogs corpus"
OUT  = "formatted_movie_lines.txt"   # ← lo que usamos después
MAX_SENT = 20                        # descarta frases larguísimas

def ascii(txt):
    return "".join(c for c in unicodedata.normalize("NFD", txt)
                   if unicodedata.category(c) != "Mn")

def norm(s):
    s = ascii(re.sub(r"[^a-zA-Z0-9?!.]+", " ", s.lower()))
    s = re.sub(r"([?.!])", r" \1 ", s)
    return re.sub(r"\s+", " ", s).strip()

# ─── descarga y des-zip ──────────────────────────────────────
if not os.path.isdir(ROOT):
    print("⏬ descargando corpus…")
    urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP) as z: z.extractall()
    os.remove(ZIP)

# ─── lee líneas y conversaciones ─────────────────────────────
print("🔧 procesando…")
lines = {}
with open(os.path.join(ROOT,"movie_lines.txt"),encoding="latin-1") as f:
    for ln in f:
        parts = ln.strip().split(" +++$+++ ")
        lines[parts[0]] = norm(parts[-1])

pairs = []
with open(os.path.join(ROOT,"movie_conversations.txt"),
          encoding="latin-1") as f:
    for conv in f:
        ids = eval(conv.strip().split(" +++$+++ ")[-1])
        for a,b in zip(ids,ids[1:]):
            q, r = lines[a], lines[b]
            if (2<=len(q.split())<MAX_SENT and
                2<=len(r.split())<MAX_SENT):
                pairs.append((q,r))

# ─── guarda en TSV (pregunta[TAB]respuesta) ──────────────────
with open(OUT,"w",encoding="utf-8",newline="") as f:
    wr = csv.writer(f,delimiter='\t')
    wr.writerows(pairs)

print(f"✅ creado {OUT} con {len(pairs):,} pares")
