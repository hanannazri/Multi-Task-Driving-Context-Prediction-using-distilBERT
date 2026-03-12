"""# **Tokenisation**"""

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased")

# Preprocessing function
def tokenize(batch):
    tokens = tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=32   )
    return {
        **tokens,
        "intent_label": batch["intent_id"],
        "emotion_label": batch["emotion_id"],
        "mode_label": batch["mode_id"]
    }

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["sentence", "intent", "emotion", "driving_mode"]
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["sentence", "intent", "emotion", "driving_mode"]
)
