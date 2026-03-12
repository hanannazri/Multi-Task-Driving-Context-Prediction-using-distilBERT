"""Extracting File"""

df = pd.read_csv(r"/content/sample_data/drivelikeme_dataset.csv")
df.head()

"""# **Exploratary Data Analysis**"""

df.shape

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

df['count'] = df['sentence'].apply(lambda x: len(x.split()))
df.head()

df['intent'].unique()

df.loc[df["intent"] == "office rush", "driving_mode"] = "priority mode"

df["intent"].value_counts()

# Combine calm and neutral into calm
df['emotion'] = df['emotion'].replace({'happy': 'calm'})

# Rename Calm to relaxed
df["emotion"] = df["emotion"].replace("calm", "relaxed")

#Remove anger rows
df = df[df['emotion'] != 'angry']

plt.figure(figsize= (8, 8))
sns.histplot(df['count'], bins=50)
plt.xlim(0, 100)
plt.ylabel('The no of sentences ', fontsize = 16)
plt.xlabel('word count ', fontsize = 16)
plt.title("Words Distribution", fontsize = 18)
plt.show()

intent_count=df['intent'].value_counts()
emotion_count=df['emotion'].value_counts()
mode_count=df['driving_mode'].value_counts()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

sns.barplot(x=intent_count.index, y=intent_count.values, ax=ax)

for p, label in zip(ax.patches, intent_count.index):
    ax.annotate(
        f'{int(p.get_height())}',
        xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='center',
        size=13,
        color='black',
        alpha=0.8
    )

plt.xlabel('Intents', size=15)
plt.ylabel('Count', size=15)
plt.xticks(size=12, rotation=45)
plt.title("Intent Distribution", size=18)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

sns.barplot(x=emotion_count.index, y=emotion_count.values, ax=ax)

for p, label in zip(ax.patches, intent_count.index):
    ax.annotate(
        f'{int(p.get_height())}',
        xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='center',
        size=13,
        color='black',
        alpha=0.8
    )

plt.xlabel('emotion', size=15)
plt.ylabel('Count', size=15)
plt.xticks(size=12, rotation=45)
plt.title("Emotion Distribution", size=18)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

sns.barplot(x=mode_count.index, y=mode_count.values, ax=ax)

for p, label in zip(ax.patches, intent_count.index):
    ax.annotate(
        f'{int(p.get_height())}',
        xy=(p.get_x() + p.get_width() / 2.0, p.get_height()),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='center',
        size=13,
        color='black',
        alpha=0.8
    )

plt.xlabel('Mode', size=15)
plt.ylabel('Count', size=15)
plt.xticks(size=12, rotation=45)
plt.title("Mode Distribution", size=18)
plt.tight_layout()
plt.show()
