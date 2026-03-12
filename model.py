# make no. of classes dynamic
num_intents = len(intent_encoder.classes_)
num_emotions = len(emotion_encoder.classes_)
num_modes = len(mode_encoder.classes_)

# Defining a custom PyTorch model that inherits from nn.Module
class DistilBertMultiTask(nn.Module):
    def __init__(self, num_intents, num_emotions, num_modes):
        super().__init__()

        #Load pretrained DistilBERT (shared encoder)
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")

        # Each token is represented by a 768-dimensional vector
        hidden_size = self.bert.config.hidden_size

        #Three separate classification heads
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)
        self.mode_classifier = nn.Linear(hidden_size, num_modes)

        # To prevent overfitting
        self.dropout = nn.Dropout(0.3)


    # Forward pass
    def forward(self, input_ids, attention_mask):

        # Pass input through DistilBERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        # gives output "outputs.last_hidden_state"

        #Pooling: convert tokens → sentence vector
        #Mean pooling is stable and commonly used
        pooled = outputs.last_hidden_state[:,0]
        pooled = self.dropout(pooled)

        # Three predictions (parallel heads)
        intent_logits = self.intent_classifier(pooled)
        emotion_logits = self.emotion_classifier(pooled)
        mode_logits = self.mode_classifier(pooled)

        return intent_logits, emotion_logits, mode_logits
