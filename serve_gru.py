# serve_gru.py ────────────────────────────────────────────────
import re, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_PATH, TOK_PATH = "chatbot_seq2seq.keras", "tokenizer.json"
MAXLEN    = 22
START, END = "<start>", "<end>"

# ── utilidades ------------------------------------------------
def _norm(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9?!.]+", " ", s.lower())
    s = re.sub(r"([?.!])", r" \1 ", s)
    return re.sub(r"\s+", " ", s).strip()

def _pad(seq):
    return tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAXLEN, padding="post"
    )

# ── carga modelo y tokenizer ----------------------------------
print("‣ cargando modelo y tokenizer…", end="", flush=True)
model = load_model(MODEL_PATH)
with open(TOK_PATH, encoding="utf-8") as f:
    tok = tokenizer_from_json(f.read())

emb_layer = model.get_layer("emb")
enc_gru   = model.get_layer("enc_gru")
dec_gru   = model.get_layer("dec_gru")
dense     = model.get_layer("dense")

enc_model = tf.keras.Model(model.input[0], enc_gru.output[1])
dec_cell  = dec_gru.cell

UNK_ID    = tok.word_index["<unk>"]
START_ID  = tok.word_index[START]
END_ID    = tok.word_index[END]

print(" listo 🟢")

# ── paso único del decoder ------------------------------------
def _step(tok_id, state):
    # token → embedding
    x = tf.constant([[tok_id]], dtype=tf.int32)    # (1,1)
    x = emb_layer(x)                                # (1,1,emb)
    x = tf.squeeze(x, axis=1)                       # (1,emb)
    h, _ = dec_cell(x, states=state)                # (1,units)
    logits = dense(h)[0].numpy()                    # (vocab,)
    logits[UNK_ID] = -1e9                           # nunca <unk>
    return logits, [h]

# ── función de inferencia greedy -----------------------------
def reply(msg: str, max_len: int = MAXLEN) -> str:
    # normaliza y codifica
    seq   = _pad(tok.texts_to_sequences([f"{START} {_norm(msg)} {END}"]))
    h_enc = enc_model.predict(seq, verbose=0)       # (1,units)
    state = [tf.convert_to_tensor(h_enc)]           # [(1,units)]

    tok_id, out_ids = START_ID, []
    for _ in range(max_len):
        logits, state = _step(tok_id, state)
        # greedy: la más probable
        tok_id = int(np.argmax(logits))

        # condiciones de parada
        if tok_id in (END_ID, START_ID):
            break
        if len(out_ids) >= 2 and tok_id == out_ids[-1] == out_ids[-2]:
            break

        out_ids.append(tok_id)

    # reconstruye texto
    return " ".join(tok.index_word[i] for i in out_ids) or "(sin respuesta)"

# ── demo CLI (opcional) ---------------------------------------
if __name__ == "__main__":
    while True:
        q = input("Tú: ").strip()
        if not q: continue
        print("Bot:", reply(q))
