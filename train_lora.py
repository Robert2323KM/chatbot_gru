# train_lora.py

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType

# ─── 1. Parámetros rápidos ────────────────────────────────────
MODEL_NAME    = "distilgpt2"
OUTPUT_DIR    = "lora_distilgpt2_cornell"
EPOCHS        = 3
BATCH_SIZE    = 8
LEARNING_RATE = 5e-4
MAX_LEN       = 64   # máximo tokens por ejemplo

# ─── 2. Carga y formatea el dataset Cornell ─────────────────
dataset = load_dataset(
    "csv",
    data_files="data/pairs.tsv",
    sep="\t",
    column_names=["q", "a"]
)

def concat_examples(ex):
    return {"text": f"<|start|> {ex['q']} <|sep|> {ex['a']} <|end|>"}

ds = dataset["train"].map(
    concat_examples,
    remove_columns=["q", "a"]
)

# ─── 3. Tokenización ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "bos_token": "<|start|>",
    "eos_token": "<|end|>",
    "pad_token": "<|sep|>"
})

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

ds_tok = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"]
)

# ─── 4. Carga el modelo y aplica LoRA ───────────────────────
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)

# ─── 5. Data collator ────────────────────────────────────────
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ─── 6. Configura el Trainer ─────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=100,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=False,   # para conservar `input_ids`
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tok,
    data_collator=collator,
)

# ─── 7. Entrena ──────────────────────────────────────────────
trainer.train()

# ─── 8. Guarda el modelo y el tokenizador ───────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning terminado y guardado en «{OUTPUT_DIR}»")
