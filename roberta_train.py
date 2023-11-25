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
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import tensorflow.keras.backend as K
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def build_classifier_model():
  input_word_ids = tf.keras.Input(shape=(256,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.Input(shape=(256,), dtype=tf.int32, name='input_mask')
  roberta_tf = TFRobertaModel.from_pretrained("roberta-base")
  embeddings = roberta_tf(input_word_ids, attention_mask = input_mask)[0]
  net = tf.keras.layers.GlobalMaxPool1D()(embeddings)
  # net = tf.keras.layers.Dense(128, activation='relu')(net)
  net = tf.keras.layers.Dropout(0.45)(net)
  op = tf.keras.layers.Dense(26, activation='softmax')(net)
  return tf.keras.Model(inputs=[input_word_ids, input_mask], outputs=op)  

nltk.download('stopwords')
nltk.download("punkt")

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

#print("Authors before cleaning: \n", df["authors"])
authors_list = []

for value in df["authors"]:
  authors = []
  if value[-4:] == ", ap":
    value = value[:-4]
  # Split the string based on ", " or " and "
  authors = re.split(r', | and ', value)
  authors = [x for x in authors if x]
  authors_list.append(authors)

#print("\nAuthors after cleaning: \n", authors_list[:200])
df["authors"] = authors_list

# Output class distribution
news_classes = list(df["category"].unique())
news_classes_count = {}
headline_lengths_per_class = {}
desc_lengths_per_class = {}
avg_headline_lengths_per_class = {}
avg_desc_lengths_per_class = {}
median_headline_lengths_per_class = {}
median_desc_lengths_per_class = {}

for news_class in news_classes:
  headline_lengths_per_class[news_class] = []
  desc_lengths_per_class[news_class] = []
  df_class = df[df["category"] == news_class]
  news_classes_count[news_class] = df_class.shape[0]
  for val in df_class["headline"]:
    headline_lengths_per_class[news_class].append(len(val))
  avg_headline_lengths_per_class[news_class] = mean(headline_lengths_per_class[news_class])
  median_headline_lengths_per_class[news_class] = median(headline_lengths_per_class[news_class])
  for val in df_class["short_description"]:
    desc_lengths_per_class[news_class].append(len(val))
  avg_desc_lengths_per_class[news_class] = mean(desc_lengths_per_class[news_class])
  median_desc_lengths_per_class[news_class] = median(desc_lengths_per_class[news_class])

news_classes_count = dict(sorted(news_classes_count.items(), key=lambda item: item[1], reverse=True))
avg_headline_lengths_per_class = dict(sorted(avg_headline_lengths_per_class.items(), key=lambda item: item[1], reverse=True))
median_headline_lengths_per_class = dict(sorted(median_headline_lengths_per_class.items(), key=lambda item: item[1], reverse=True))
avg_desc_lengths_per_class = dict(sorted(avg_desc_lengths_per_class.items(), key=lambda item: item[1], reverse=True))
median_desc_lengths_per_class = dict(sorted(median_desc_lengths_per_class.items(), key=lambda item: item[1], reverse=True))

authors_ctgry_map = {}
authors_count_map = {}
for i in range(len(df)):
  # print(df["authors"].iloc[i])
  for author in df["authors"].iloc[i]:
    if author in authors_ctgry_map:
        authors_count_map[author] += 1
        if df["category"].iloc[i] in authors_ctgry_map[author]:
          (authors_ctgry_map[author])[df["category"].iloc[i]] +=1
        else:
          (authors_ctgry_map[author])[df["category"].iloc[i]] = 1
    else:
      authors_ctgry_map[author] = {}
      authors_count_map[author] = 1

authors_count_map = dict(sorted(authors_count_map.items(), key=lambda item: item[1], reverse=True))

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

tf.get_logger().setLevel('ERROR')

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 8
seed = 42

train_df, temp_df = train_test_split(reduced_df, test_size=0.3, stratify=reduced_df['category'], random_state=seed)
validation_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=seed)

# Print class distribution for verification
print("Training class distribution:\n", train_df['category'].value_counts(normalize=True))
print("Validation class distribution:\n", validation_df['category'].value_counts(normalize=True))
print("Test class distribution:\n", test_df['category'].value_counts(normalize=True))

# Create datasets using tf.data.Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((train_df['combined_text'].values, pd.get_dummies(train_df['category'].values)))
# validation_dataset = tf.data.Dataset.from_tensor_slices((validation_df['combined_text'].values, pd.get_dummies(validation_df['category'].values)))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_df['combined_text'].values, pd.get_dummies(test_df['category'].values)))

print("\nDATASET CLASSES: ", pd.get_dummies(train_df['category'].values).columns.tolist())
print("\nActual test classes: ", test_df['category'])

# Shuffle and batch datasets
# train_dataset = train_dataset.shuffle(buffer_size=len(train_df), seed=seed, reshuffle_each_iteration=False).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
X_train_encoded = roberta_tokenizer(text=train_df["combined_text"].tolist(), padding='max_length', truncation=True, max_length=256,return_token_type_ids=True,return_tensors='tf')
X_validation_encoded = roberta_tokenizer(text=validation_df["combined_text"].tolist(), padding='max_length', truncation=True, max_length=256,return_token_type_ids=True,return_tensors='tf')

train_dataset = tf.data.Dataset.from_tensor_slices(({'input_word_ids': X_train_encoded['input_ids'],'input_mask': X_train_encoded['attention_mask']}, pd.get_dummies(train_df['category'].values))).shuffle(buffer_size=len(X_train_encoded)).batch(batch_size).prefetch(1)
validation_dataset = tf.data.Dataset.from_tensor_slices(({'input_word_ids': X_validation_encoded['input_ids'],'input_mask': X_validation_encoded['attention_mask']}, pd.get_dummies(validation_df['category'].values))).shuffle(buffer_size=len(X_validation_encoded)).batch(batch_size).prefetch(1)

classifier_model = build_classifier_model()
# bert_raw_result = classifier_model(tf.constant(["Maria Sharapova beats Victoria Azarenka"]))
# print(tf.sigmoid(bert_raw_result))
loss = tf.keras.losses.CategoricalCrossentropy()
metrics = tf.metrics.CategoricalAccuracy()
epochs = 4
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 2e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

hist = classifier_model.fit(x=train_dataset, validation_data=validation_dataset, epochs=epochs)

print("History: \n", hist)
loss, accuracy = classifier_model.evaluate(validation_dataset)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
classifier_model.save("roberta_trial_1_model_new21Nov", include_optimizer=False)
