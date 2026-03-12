model.eval()

true_modes = []
pred_modes = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mode_labels = batch["mode_label"].to(device)

        intent_logits, emotion_logits, mode_logits = model(
            input_ids, attention_mask
        )

        preds = mode_logits.argmax(dim=1)

        true_modes.extend(mode_labels.cpu().numpy())
        pred_modes.extend(preds.cpu().numpy())

cm = confusion_matrix(true_modes, pred_modes)

import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=mode_encoder.classes_
)

disp.plot(cmap="Blues")
plt.title("Mode Confusion Matrix")
plt.show()
