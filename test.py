import gdown
import zipfile
import nltk
import pandas as pd
import numpy as np
import json
from numpy.lib.function_base import median
from numpy import mean
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import nltk
# from nltk.corpus import stopwords
# from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


gdown.download('https://drive.google.com/uc?id=1SucKmwpwhoptI7eO3XQM4AjibWaJc9oc', quiet=False)
with zipfile.ZipFile('BERT_trial_1_model_new16Nov-20231116T234003Z-001.zip', 'r') as zip_ref:
    zip_ref.extractall()

file_path = 'news_class_dataset.json'
df = pd.read_json(file_path, lines=True)
# Converting to lower case
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Dropping entries with NULL values
df = df.dropna()

links = df["link"]
dates = df["date"]

# Dropping irrelevant attributes
df.drop('link', axis=1)
df.drop('date', axis=1)

df["category"] = df["category"].replace(
              {"healthy living": "wellness",
              "queer voices": "groups voices",
              "worldpost": "world news",
              "science": "science & tech",
              "tech": "science & tech",
              "money": "business & finances",
              "arts": "arts & culture",
              "college": "education",
              "latino voices": "groups voices",
              "business": "business & finances",
              "parents": "parenting",
              "black voices": "groups voices",
              "the worldpost": "world news",
              "style": "style & beauty",
              "green": "environment",
              "taste": "food & drink",
              "culture & arts": "arts & culture",
              "good news": "miscellaneous",
              "weird news": "miscellaneous",
              "fifty": "miscellaneous"}
            )

# Feature transformation for training

df['combined_text'] = df['headline'] + " " + df['short_description']

reduced_df = df[['combined_text', 'category']]
seed = 42

train_df, temp_df = train_test_split(reduced_df, test_size=0.3, stratify=reduced_df['category'], random_state=seed)
validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=seed)

test_df.to_csv('test_dataset.csv', index=False)


# Create datasets using tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['combined_text'].values, pd.get_dummies(train_df['category'].values)))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_df['combined_text'].values, pd.get_dummies(validation_df['category'].values)))
test_dataset = tf.data.Dataset.from_tensor_slices((test_df['combined_text'].values, pd.get_dummies(test_df['category'].values)))

dataset_classes = pd.get_dummies(train_df['category'].values).columns.tolist()
print("\nDATASET CLASSES: ", dataset_classes)
print("\nActual test classes: \n", test_df['category'])

loaded_model = tf.saved_model.load('BERT_trial_1_model_new16Nov')

pred_classes = []
last_index = 0

for i in range(int(len(test_df)/1500)):
  pred = loaded_model(tf.constant((test_df['combined_text'].values.tolist())[i*1500:(i+1)*1500]))
  pred_classes_per_batch = [dataset_classes[i] for i in np.argmax(pred, axis=1)]
  pred_classes = pred_classes + pred_classes_per_batch
  last_index = i

pred = loaded_model(tf.constant((test_df['combined_text'].values.tolist())[(last_index+1)*1500:]))
pred_classes_per_batch = [dataset_classes[i] for i in np.argmax(pred, axis=1)]
pred_classes = pred_classes + pred_classes_per_batch
print("\n\nPredicted classes: \n", pred_classes)

print("Model performance based on the test dataset")

test_actual_classes = test_df["category"]
print("\n\nActual classes: \n", test_actual_classes)

test_df['prediction'] = pred_classes
test_df.to_csv('test_dataset_withpred.csv', index=False)


# Calculate metrics
accuracy = accuracy_score(test_actual_classes, pred_classes)
f1 = f1_score(test_actual_classes, pred_classes, average='weighted')
precision = precision_score(test_actual_classes, pred_classes, average='weighted')
recall = recall_score(test_actual_classes, pred_classes, average='weighted')

# Print metrics
print("Accuracy: {:.5f}".format(accuracy))
print("F1-score: {:.5f}".format(f1))
print("Precision: {:.5f}".format(precision))
print("Recall: {:.5f}".format(recall))


# for pred in pred_classes:
#   print(news_classes[pred])

# Confusion matrix
cm = confusion_matrix(test_actual_classes, pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=pd.get_dummies(train_df['category'].values).columns.tolist(), yticklabels=pd.get_dummies(train_df['category'].values).columns.tolist())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()



