save_path = "saved_model"

os.makedirs(save_path, exist_ok=True)

# Save model weights
torch.save(model.state_dict(), f"{save_path}/model.pt")

# Save tokenizer
tokenizer.save_pretrained(f"{save_path}/tokenizer")

# Save encoders
with open(f"{save_path}/intent_encoder.pkl", "wb") as f:
    pickle.dump(intent_encoder, f)

with open(f"{save_path}/emotion_encoder.pkl", "wb") as f:
    pickle.dump(emotion_encoder, f)

with open(f"{save_path}/mode_encoder.pkl", "wb") as f:
    pickle.dump(mode_encoder, f)

print("Model saved successfully.")
