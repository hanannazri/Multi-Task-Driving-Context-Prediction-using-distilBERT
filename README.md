# Multi-Task-Driving-Context-Prediction-using-distilBERT

A Multi-Task NLP System built with DistilBERT that understands driver commands and automatically determines:

- Driver Intent
- Driver Emotion
- Optimal Driving Mode

This project enables AI-powered personalized driving experiences by dynamically adapting vehicle behavior based on natural language commands and emotional context.

The model is designed for Agentic AI-based Advanced Driver Assistance Systems (ADAS) where the system intelligently decides the appropriate driving strategy instead of blindly executing commands.

# Project Overview

Modern vehicles increasingly rely on voice interfaces and intelligent assistants. However, most systems only perform single-task intent detection.

This project introduces a Multi-Task Transformer Architecture that simultaneously predicts:

| Task                       | Description                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------- |
| **Intent Classification**  | Understand the driver’s objective (e.g., urgency, fuel management, traffic avoidance) |
| **Emotion Detection**      | Detect emotional context (relaxed, stressed, etc.)                                    |
| **Driving Mode Selection** | Determine the optimal driving behavior                                                |
Instead of training three separate models, the system uses one shared DistilBERT encoder with three specialized classification heads.

This approach improves efficiency, contextual understanding, inference speed and shared linguistic representation.

🏗 System Architecture
Driver Speech / Command
          │
          ▼
   Text Processing
          │
          ▼
     DistilBERT Encoder
(shared contextual representation)
          │
 ┌────────┼─────────┐
 ▼        ▼         ▼
Intent   Emotion   Mode
 Head     Head      Head
Outputs
Sentence → 3 Predictions

"I am getting late to the office"

Intent   → Office Rush
Emotion  → Stressed
Mode     → Priority Mode
📊 Dataset

The dataset contains natural driving commands annotated with:

Intent

Emotion

Driving Mode

Example dataset format:

Sentence	Intent	Emotion	Driving Mode
Take the fastest route	urgency	stressed	priority mode
Petrol is running low	fuel management	relaxed	eco mode
Let’s enjoy the scenery	scenic drive	relaxed	comfort mode
Dataset Preprocessing

The dataset undergoes:

text normalization

label cleaning

emotion merging

duplicate removal

class balancing

Example cleaning operations:

happy → calm

calm → relaxed

remove angry samples

Word length distribution and class distributions are visualized during EDA.

🔬 Exploratory Data Analysis

The training pipeline includes:

Word count distribution

Intent frequency visualization

Emotion distribution

Driving mode distribution

These analyses help understand dataset imbalance and guide model design.

⚙️ Model Architecture

The model is implemented using PyTorch and HuggingFace Transformers.

Base Model
distilbert-base-uncased

DistilBERT provides:

lightweight architecture

faster inference

lower memory usage

strong contextual embeddings

Multi-Task Model

A custom PyTorch model with three classification heads.

DistilBERT
     │
     ▼
 Sentence Embedding
     │
     ▼
 Dropout
     │
 ┌───┼───────────────┐
 ▼   ▼               ▼
Intent Head    Emotion Head    Mode Head
Linear Layer   Linear Layer    Linear Layer

Each head independently predicts its label.

This design enables shared learning across tasks.

🧪 Training Strategy
Train / Validation Split
80% Training
10% Validation
10% Test
Hyperparameters
Parameter	Value
Model	DistilBERT
Max Sequence Length	32
Batch Size	16
Epochs	8
Learning Rate	2e-5
Optimizer	AdamW
Dropout	0.3
Multi-Task Loss Function

Each task has its own loss.

Total Loss =
1.2 × Intent Loss
+ 1.5 × Mode Loss
+ 0.8 × Emotion Loss

Mode prediction is weighted higher because it directly affects vehicle behavior.

Class Imbalance Handling

Driving mode imbalance is handled using:

compute_class_weight()

This prevents the model from favoring dominant modes.

📈 Evaluation

Evaluation includes confusion matrix analysis for driving modes.

Example evaluation pipeline:

True Modes vs Predicted Modes
→ Confusion Matrix
→ Visualization

This helps identify:

misclassification patterns

overlapping intents

weak class boundaries

🔮 Example Predictions
Input:
"I am getting late to the office"

Output:
Intent   : Office Rush
Emotion  : Stressed
Mode     : Priority Mode
Input:
"Petrol is running low"

Output:
Intent   : Fuel Management
Emotion  : Relaxed
Mode     : Eco Mode
Input:
"Relax, let me enjoy the view"

Output:
Intent   : Scenic Drive
Emotion  : Relaxed
Mode     : Comfort Mode

Prediction is implemented through the predict() function. 

distilbert_training (4)

💾 Model Saving

The trained model and preprocessing components are stored for deployment.

Saved artifacts:

saved_model/
│
├── model.pt
├── tokenizer/
├── intent_encoder.pkl
├── emotion_encoder.pkl
└── mode_encoder.pkl

This enables direct loading for inference or edge deployment.

Model saving implementation is provided in the training script. 

distilbert_training (4)

🛠 Installation
Clone Repository
git clone https://github.com/yourusername/drivelikeme-ai.git
cd drivelikeme-ai
Install Dependencies
pip install torch
pip install transformers
pip install datasets
pip install scikit-learn
pip install pandas
pip install seaborn
pip install matplotlib
pip install nltk
▶️ Training the Model

Run the training script:

python distilbert_training.py

The pipeline will automatically:

Load dataset

Preprocess text

Train DistilBERT

Evaluate predictions

Save the trained model

🚀 Deployment (Edge Devices)

The saved model can be deployed on:

Raspberry Pi

Edge AI modules

In-vehicle infotainment systems

Typical deployment pipeline:

Microphone → Speech to Text
               │
               ▼
           NLP Model
               │
               ▼
        Driving Mode Controller
🔮 Future Improvements

Planned improvements include:

🎤 Real-time voice command integration

🧠 Emotion recognition from speech

🚘 Integration with ADAS control systems

⚡ Model quantization for edge deployment

📡 Agentic AI decision layer

📚 Technologies Used

Python

PyTorch

HuggingFace Transformers

DistilBERT

Scikit-Learn

NLTK

Pandas

Matplotlib / Seaborn

🎓 Research Contribution

This project demonstrates:

Multi-Task NLP for automotive systems

Emotion-aware driving assistance

Transformer-based intent recognition

Personalized vehicle behavior

👨‍💻 Author

Hanan

AI Research & Machine Learning Enthusiast
