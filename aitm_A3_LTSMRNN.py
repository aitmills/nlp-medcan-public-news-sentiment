# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:06:59 2023

@author: Adam
"""
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard  # Import TensorBoard

# Assuming the dataset is in a CSV file
print("LOAD DATA")
data = pd.read_csv('combined_df_textblob_sentiment.csv')
data.dropna(subset=['composite_text'], inplace=True)
# Convert sentiment labels to binary classes
#data["sentiment_numeric"] = np.where(data["sentiment"] == "positive", 1, 0)

## Preprocessing
print("PREPROCESSING")
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove unwanted characters and lowercase the text
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    # Tokenize and remove stop words
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in set(stopwords.words("english"))]
    return " ".join(words)

def convertto_numeric(sentimentlabel):
    if sentimentlabel == 'positive':
        output = 1
    else:
        output = 0
    return output

data["sentiment_numeric"] = data["composite_label"].apply(convertto_numeric)
data["clean_composite"] = data["composite_text"].apply(preprocess_text)

# EDA PLOTS
print("EDA PLOTS")
data['composite_length'] = data['clean_composite'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(data['composite_length'], kde=True)
plt.title('Distribution of Composite Text Lengths')
plt.xlabel('Article Length (Words)')
plt.ylabel('Frequency')
plt.show()

word_freq = Counter(" ".join(data['clean_composite']).split())
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency']).sort_values(by='frequency', ascending=False)
plt.figure(figsize=(10, 6))
top_n = 30  # Number of top words to display
sns.barplot(x='word', y='frequency', data=word_freq_df.head(top_n))
plt.title(f'Top {top_n} Word Frequencies')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# FEATURE EXTRACTION
# Set the maximum sequence length and vocabulary size
print("FEATURE EXTRACTION")
max_len = 250
vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data["clean_composite"])

# Convert the text to sequences and pad them to have the same length
sequences = tokenizer.texts_to_sequences(data["clean_composite"])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data["sentiment_numeric"], test_size=0.2, random_state=42)

# DEFINE LTSM MODEL
print("DEFINE LTSM MODEL")
embedding_dim = 128
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# TRAIN MODEL
print("TRAIN MODEL")
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Create a TensorBoard callback object and specify the log directory
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stop, tensorboard_callback]
)

# PLOT ACCURACY VS EPOCHS
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# EVALUATE
# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)
print("OUTPUT EVAL PLOTS")

# Plot ROC curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = {0:0.2f})'
             ''.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(y_test, y_pred_prob.ravel())

# TUNE MODEL
print("Start model tuning with keras-tuner")
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Define a function that builds and compiles the model with hyperparameters
def build_model(hp: HyperParameters):
    model = Sequential([
        Embedding(vocab_size,
                  hp.Int("embedding_dim", min_value=64, max_value=256, step=32),
                  input_length=max_len),
        Bidirectional(LSTM(hp.Int("lstm_units_1", min_value=32, max_value=128, step=32),
                           return_sequences=True)),
        Bidirectional(LSTM(hp.Int("lstm_units_2", min_value=16, max_value=64, step=16))),
        Dense(hp.Int("dense_units", min_value=32, max_value=128, step=32),
              activation="relu"),
        Dropout(hp.Float("dropout", min_value=0.3, max_value=0.7, step=0.1)),
        Dense(1, activation="sigmoid")  # Change the output layer to have 3 units and softmax activation
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # Change the loss function back to categorical_crossentropy
    return model

# Initialize the tuner and perform random search:
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=3,
    directory="random_search",
    project_name="medcan_composite_sentiment_analysis"
)

tuner.search_space_summary()
tuner.search(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[tensorboard_callback])

# Get the best hyperparameters:
best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
print("BEST HYPER PARAMETERS")
print(best_hyperparams)
print("---")
print("Train the model with the best hyperparameters:")
best_model = tuner.hypermodel.build(best_hyperparams)
best_model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)
print("Evaluate the best model on the test set")
best_model.evaluate(X_test, y_test)
print("---")
# Plots
print("Output plots")
#trial_history = tuner.oracle.get_trial_data()

trial_data = []

for trial in tuner.oracle.trials.values():
    trial_params = trial.hyperparameters.values.copy()
    val_accuracy_metric = trial.metrics.metrics.get('val_accuracy')
    if val_accuracy_metric:
        trial_params['validation_accuracy'] = np.max([mo.value for mo in val_accuracy_metric.get_history()])
    else:
        trial_params['validation_accuracy'] = None
    trial_data.append(trial_params)

trial_df = pd.DataFrame(trial_data)

# Sort trial_df by validation_accuracy in descending order
trial_df = trial_df.sort_values(by='validation_accuracy', ascending=False)

# Display the sorted trial_df
print("Hyperparameters tested and their validation accuracies:")
print(trial_df)
trial_df.to_csv('trial_df.csv')

sns.pairplot(trial_df, diag_kind='kde', markers='o', corner=True, plot_kws=dict(s=50, edgecolor="b", linewidth=1))
plt.suptitle("Hyperparameters vs Validation Accuracy", y=1.02)
plt.tight_layout()
plt.show()

# TUNING ROC PLOTS
plt.figure()
lw = 2

for idx, trial in enumerate(tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))):
    # Build and fit the model with the trial's hyperparameters
    model = tuner.hypermodel.build(trial.hyperparameters)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Make predictions on the test set
    y_pred_prob = model.predict(X_test).ravel()
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             lw=lw, label='Trial {}: ROC curve (area = {:.2f})'.format(idx+1, roc_auc))


# Plot the ROC curve for a random classifier
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for All Keras Tuner Trials')
plt.legend(loc="lower right", bbox_to_anchor=(1.05, 0))

plt.show()
