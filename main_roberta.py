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
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from transformers import TFRobertaModel, RobertaTokenizer


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def build_classifier_model():
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    text_tokenized = tokenizer(text_input, truncation=True, padding=True, return_tensors="tf")['input_ids']
    encoder = TFRobertaModel.from_pretrained("roberta-base", trainable=True, name='RoBERTa_encoder')
    outputs = encoder(text_tokenized)
    net = tf.keras.layers.GlobalAveragePooling1D()(outputs.last_hidden_state)
    net = tf.keras.layers.Dropout(0.45)(net)
    net = tf.keras.layers.Dense(26, activation='softmax', name='classifier')(net)
    model = tf.keras.Model(text_input, net)

    return model

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

#i = 0
#for key in news_classes_count.keys():
#  if i==3:
#    break
#  plt.figure(figsize=(10, 10))
#  plt.title('Headline Length boxplot for the Class ' + key)
#  plt.ylabel('Length')
#  plt.boxplot(headline_lengths_per_class[key])
#  plt.figure(figsize=(10, 10))
#  plt.title('Description Length boxplot for the Class ' + key)
#  plt.ylabel('Length')
#  plt.boxplot(desc_lengths_per_class[key])
#  plt.show()
#  i += 1


#print("News class count: ", news_classes_count)
#print("Average headline lengths per class: ", avg_headline_lengths_per_class)
#print("Average short desc lengths per class: ", avg_desc_lengths_per_class)
#print("Median headline lengths per class: ", median_headline_lengths_per_class)
#print("Median short desc lengths per class: ", median_desc_lengths_per_class)

# Displaying the Wordcloud, word frequencies and
#for i in range(3):
#  news_class = list(news_classes_count.keys())[i]
#  print(f'The Wordcloud for class "{news_class}" is shown below:\n')
#  combined_class_headlines = " ".join(df[df["category"] == news_class]["headline"])
  # print(combined_class_headlines[:1000])
#  words = re.findall(r'\w+', combined_class_headlines)
#  filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
#  word_freq = Counter(filtered_words)
#  print(f'Word frequencies (minus stopwords) for class "{news_class}": {word_freq}')

#  for j in [2,3]:
#    n_grams = ngrams(words, j)
#    n_gram_counter = Counter(n_grams)
#    top_ten_n_grams = n_gram_counter.most_common(10)
#    print(f'Most common {j}-grams for class {news_class}: {top_ten_n_grams}')

#  wordcloud = WordCloud(width=600, height=300, background_color='white').generate(combined_class_headlines)
#  plt.figure(figsize=(10, 5))
#  plt.imshow(wordcloud, interpolation='bilinear')
#  plt.axis('off')
#  plt.show()

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

#print("Author - news category relationship: ", authors_ctgry_map)

authors_count_map = dict(sorted(authors_count_map.items(), key=lambda item: item[1], reverse=True))
#print("No. of articles each author has written: ", authors_count_map)

# Top 3 writers
#print("Articles written by lee moran: ", authors_ctgry_map["lee moran"])
#print("Articles written by ron dicker: ", authors_ctgry_map["ron dicker"])
#print("Articles written by ed mazza: ", authors_ctgry_map["ed mazza"])

#plt.figure(figsize=(10, 10))
#plt.pie(x=df.category.value_counts(), labels=df.category.value_counts().index, autopct='%1.1f%%', textprops={'fontsize' : 8,
#                                                                                                'alpha' : .7});
#plt.title('Percentage Class Distribution', alpha=.7)
#plt.tight_layout()

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

# news_classes = list(reduced_df["category"].unique())
# news_class_index = {}

# for i in range(len(news_classes)):
#   news_class_index[news_classes[i]] = i

# label_encoder = LabelEncoder()
# reduced_df["category"] = label_encoder.fit_transform(reduced_df["category"])

#print("Dataframe: \n", reduced_df)

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
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['combined_text'].values, pd.get_dummies(train_df['category'].values)))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_df['combined_text'].values, pd.get_dummies(validation_df['category'].values)))
test_dataset = tf.data.Dataset.from_tensor_slices((test_df['combined_text'].values, pd.get_dummies(test_df['category'].values)))

print("\nDATASET CLASSES: ", pd.get_dummies(train_df['category'].values).columns.tolist())
print("\nActual test classes: ", test_df['category'])

# Shuffle and batch datasets
train_dataset = train_dataset.shuffle(buffer_size=len(train_df), seed=seed, reshuffle_each_iteration=False).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/4"
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_encoder_model = hub.KerasLayer(tfhub_handle_encoder)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(["Maria Sharapova beats Victoria Azarenka"]))
print(tf.sigmoid(bert_raw_result))
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
classifier_model.save("BERT_trial_1_model_new16Nov", include_optimizer=False)


loaded_model = tf.saved_model.load('BERT_trial_1_model_new16Nov')
#loaded_model.summary()

pred = loaded_model(test_df['combined_text'].values)
pred_classes = np.argmax(pred, axis=1)
print("\n\nPredicted classes: \n", pred_classes)

print("Model performance based on the test dataset")

test_actual_classes = test_df["category"]
print("\n\nActual classes: \n", test_actual_classes)



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

