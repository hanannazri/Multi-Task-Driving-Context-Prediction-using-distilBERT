"""# **Label Encoding**"""

intent_encoder = LabelEncoder()
emotion_encoder = LabelEncoder()
mode_encoder = LabelEncoder()

df["intent_id"]  = intent_encoder.fit_transform(df["intent"])
df["emotion_id"] = emotion_encoder.fit_transform(df["emotion"])
df["mode_id"]    = mode_encoder.fit_transform(df["driving_mode"])

df['mode_id'].value_counts()

"""**Train - Test Split**"""

# first split train and temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# split temp into validation and test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

#Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)
