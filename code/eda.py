import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg') 

# Loads the dataset
column_names = ['id', 'entity', 'sentiment', 'content']
df = pd.read_csv('data/twitter_training.csv', names=column_names)

# Handles nulls
print(f"Missing values before: {df['content'].isnull().sum()}")
df = df.dropna(subset=['content'])
print(f"Missing values after: {df['content'].isnull().sum()}")

# Get sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment', order=['Positive', 'Negative', 'Neutral', 'Irrelevant'])
plt.title('Sentiment Class Distribution')
plt.savefig('sentiment_distribution.png') 
print("Saved: sentiment_distribution.png")

# Analysis of top 10  most mentioned
top_10_entities = df['entity'].value_counts().head(10).index
df_top_10 = df[df['entity'].isin(top_10_entities)]

plt.figure(figsize=(12, 6))
sns.countplot(data=df_top_10, y='entity', hue='sentiment')
plt.title('Top 10 Entities by Sentiment')
plt.tight_layout()
plt.savefig('entity_analysis.png') 
print("Saved: entity_analysis.png")