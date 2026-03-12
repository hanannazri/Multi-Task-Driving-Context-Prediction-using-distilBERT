def predict(sentence):

    model.eval()

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    ).to(device)

    with torch.no_grad():
        intent_logits, emotion_logits, mode_logits = model(**inputs)

    # Convert logits → probabilities
    intent_probs = F.softmax(intent_logits, dim=1)
    emotion_probs = F.softmax(emotion_logits, dim=1)
    mode_probs = F.softmax(mode_logits, dim=1)

    # Get predictions
    intent_id = intent_probs.argmax(dim=1).item()
    emotion_id = emotion_probs.argmax(dim=1).item()
    mode_id = mode_probs.argmax(dim=1).item()

    # Confidence scores
    intent_conf = intent_probs.max().item()
    emotion_conf = emotion_probs.max().item()
    mode_conf = mode_probs.max().item()

    # Convert back to labels
    intent = intent_encoder.inverse_transform([intent_id])[0]
    emotion = emotion_encoder.inverse_transform([emotion_id])[0]
    mode = mode_encoder.inverse_transform([mode_id])[0]

    return {
        "intent": (intent, intent_conf),
        "emotion": (emotion, emotion_conf),
        "mode": (mode, mode_conf)
    }

predict("Take the fastest route, I am late")

predict("go fast")

predict("I want to go to hospital")

predict("it is raining")
