device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertMultiTask(
    num_intents,
    num_emotions,
    num_modes).to(device)

optimizer = AdamW( model.parameters(), lr=2e-5, weight_decay=0.01)

"""**Loss function (combined loss)**"""

#Compute mode weights
mode_classes = np.unique(df["mode_id"])

mode_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=mode_classes,
    y=df["mode_id"]
)

mode_class_weights = torch.tensor(mode_class_weights, dtype=torch.float).to(device)

intent_loss_fn = nn.CrossEntropyLoss()
emotion_loss_fn = nn.CrossEntropyLoss()
mode_loss_fn = nn.CrossEntropyLoss(weight=mode_class_weights)

"""Training"""

model.train()
num_epochs = 8

num_training_steps = len(train_loader) * num_epochs

#Learning scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps)

#Training loop
for epoch in range(num_epochs):

    total_epoch_loss = 0
    intent_epoch_loss = 0
    emotion_epoch_loss = 0
    mode_epoch_loss = 0  # reset every epoch

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent_label"].to(device)
        emotion_labels = batch["emotion_label"].to(device)
        mode_labels = batch["mode_label"].to(device)

        intent_logits, emotion_logits, mode_logits = model(
            input_ids, attention_mask
        )

        loss_intent = intent_loss_fn(intent_logits, intent_labels)
        loss_emotion = emotion_loss_fn(emotion_logits, emotion_labels)
        loss_mode = mode_loss_fn(mode_logits, mode_labels)

        loss = 1.2*loss_intent + 1.5*loss_mode + 0.8*loss_emotion

        loss.backward()

        # Gradient clipping AFTER backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        #Total average loss
        total_epoch_loss += loss.item()
        intent_epoch_loss += loss_intent.item()
        emotion_epoch_loss += loss_emotion.item()
        mode_epoch_loss += loss_mode.item()

    print(f"\nEpoch {epoch+1}")
    print(f"Total Loss   : {total_epoch_loss / len(train_loader):.4f}")
    print(f"Intent Loss  : {intent_epoch_loss / len(train_loader):.4f}")
    print(f"Emotion Loss : {emotion_epoch_loss / len(train_loader):.4f}")
    print(f"Mode Loss    : {mode_epoch_loss / len(train_loader):.4f}")
