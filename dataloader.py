#Set PyTorch format
cols = [
    "input_ids",
    "attention_mask",
    "intent_label",
    "emotion_label",
    "mode_label"
]

train_dataset.set_format(type="torch", columns=cols)
val_dataset.set_format(type="torch", columns=cols)

#Check tensor
sample = train_dataset[0]

print(sample["intent_label"], sample["intent_label"].dtype)
print(sample["emotion_label"])
print(sample["mode_label"])
