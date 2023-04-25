# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:23:41 2023

@author: Adam
"""
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize

combined_df = pd.read_csv('combined_df_textblob_sentiment.csv')

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(combined_df['composite_text'])
y = combined_df['composite_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2023)
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=np.unique(y))

# Apply SMOTE to the training data
smote = SMOTE(random_state=2023)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=2023),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['newton-cg', 'lbfgs', 'liblinear']
        }
    },
    'Support Vector Machines': {
        'model': SVC(random_state=2023, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=2023),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=2023),
        'params': {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [-1, 10, 20]
        }
    }
}

def evaluate_models(models, X_train, X_test, y_train, y_test, smote=False):
    if smote:
        smote_obj = SMOTE(random_state=2023)
        X_train, y_train = smote_obj.fit_resample(X_train, y_train)

    results = {}
    skf = StratifiedKFold(n_splits=5)

    for name, model_info in models.items():
        print(f"Training {name}...")
        start_time = time()
        clf = GridSearchCV(model_info['model'], model_info['params'], cv=skf, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Get predicted probabilities
        y_pred_prob = clf.predict_proba(X_test)
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        else:
            # For models like SVM with probability=False, use decision_function instead
            y_score = clf.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())  # Scale to [0, 1]
        end_time = time()

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        results[name] = {
            'Accuracy': accuracy,
            'F1-score': f1,
            'Confusion Matrix': cm,
            'Classification Report': cr,
            'Best Parameters': clf.best_params_,
            'Predicted Probabilities': y_pred_prob,
            'y_score': y_score,
            'Time': end_time - start_time
        }

        print(f"{name} training completed in {end_time - start_time:.2f} seconds.\n")

    return results

# Evaluate models without SMOTE
evaluation_results_no_smote = evaluate_models(models, X_train, X_test, y_train, y_test, smote=False)

# Evaluate models with SMOTE
evaluation_results_smote = evaluate_models(models, X_train, X_test, y_train, y_test, smote=True)

for name, result in evaluation_results_no_smote.items():
    print(f"{name}:")
    print(f"  Accuracy: {result['Accuracy']:.2f}")
    print(f"  F1-score: {result['F1-score']:.2f}")
    print(f"  Time: {result['Time']:.2f} seconds")
    print("\nBest Parameters:\n", result['Best Parameters'])
    print("\nClassification Report:\n", result['Classification Report'])
    print()

for name, result in evaluation_results_smote.items():
    print(f"{name}:")
    print(f"  Accuracy: {result['Accuracy']:.2f}")
    print(f"  F1-score: {result['F1-score']:.2f}")
    print(f"  Time: {result['Time']:.2f} seconds")
    print("\nBest Parameters:\n", result['Best Parameters'])
    print("\nClassification Report:\n", result['Classification Report'])
    print()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ordered_bar_subplot(ax, data_no_smote, data_smote, title, ylabel, ylim=None):
    data_no_smote = sorted(data_no_smote, key=lambda x: x[0])
    data_smote = sorted(data_smote, key=lambda x: x[0])
    names_no_smote, values_no_smote = zip(*data_no_smote)
    names_smote, values_smote = zip(*data_smote)
    
    bar_width = 0.35
    x = np.arange(len(names_no_smote))
    
    ax.bar(x - bar_width / 2, values_no_smote, bar_width, label='No SMOTE')
    ax.bar(x + bar_width / 2, values_smote, bar_width, label='With SMOTE')
    
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)  # Set y-axis limits if ylim is provided
    ax.set_xticks(x)
    ax.set_xticklabels(names_no_smote, rotation=45)
    ax.legend()


# Prepare data
accuracy_data_no_smote = [(name, result['Accuracy']) for name, result in evaluation_results_no_smote.items()]
accuracy_data_smote = [(name, result['Accuracy']) for name, result in evaluation_results_smote.items()]
f1_data_no_smote = [(name, result['F1-score']) for name, result in evaluation_results_no_smote.items()]
f1_data_smote = [(name, result['F1-score']) for name, result in evaluation_results_smote.items()]
time_data_no_smote = [(name, result['Time']) for name, result in evaluation_results_no_smote.items()]
time_data_smote = [(name, result['Time']) for name, result in evaluation_results_smote.items()]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_ordered_bar_subplot(axes[0], accuracy_data_no_smote, accuracy_data_smote, 'Accuracy', 'Accuracy Score', ylim=(0, 1))
plot_ordered_bar_subplot(axes[1], f1_data_no_smote, f1_data_smote, 'F1-score', 'F1 Score', ylim=(0, 1))
plot_ordered_bar_subplot(axes[2], time_data_no_smote, time_data_smote, 'Model Comparison: Training Time', 'Training Time (seconds)')

plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()

def plot_roc_curves(evaluation_results_no_smote, evaluation_results_smote):
    plt.figure(figsize=(16, 6))
    
    # First subplot for No SMOTE results
    plt.subplot(1, 2, 1)
    for name in evaluation_results_no_smote.keys():
        result_no_smote = evaluation_results_no_smote[name]
        y_score_no_smote = result_no_smote['y_score'][:, 1]
        fpr_no_smote, tpr_no_smote, _ = roc_curve(y_test_binarized.ravel(), y_score_no_smote.ravel())
        roc_auc_no_smote = auc(fpr_no_smote, tpr_no_smote)
        plt.plot(fpr_no_smote, tpr_no_smote, label=f"{name} - AUC: {roc_auc_no_smote:.2f}")
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (No SMOTE)')
    plt.legend(loc="lower right")
    
    # Second subplot for SMOTE results
    plt.subplot(1, 2, 2)
    for name in evaluation_results_smote.keys():
        result_smote = evaluation_results_smote[name]
        y_score_smote = result_smote['y_score'][:, 1]
        fpr_smote, tpr_smote, _ = roc_curve(y_test_binarized.ravel(), y_score_smote.ravel())
        roc_auc_smote = auc(fpr_smote, tpr_smote)
        plt.plot(fpr_smote, tpr_smote, label=f"{name} - AUC: {roc_auc_smote:.2f}")
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (SMOTE)')
    plt.legend(loc="lower right")
    
    plt.show()

plot_roc_curves(evaluation_results_no_smote, evaluation_results_smote)



# Number of models
n_models = len(models)
# Create confusion matrix subplots
fig, axes = plt.subplots(nrows=2, ncols=n_models, figsize=(20, 12), sharey=True)

for i, (name, result_no_smote) in enumerate(evaluation_results_no_smote.items()):
    for _, (name_smote, result_smote) in enumerate(evaluation_results_smote.items()):
        if name == name_smote:
            cm_no_smote = result_no_smote['Confusion Matrix']
            cm_smote = result_smote['Confusion Matrix']

    
    cm_normalized_no_smote = cm_no_smote.astype('float') / cm_no_smote.sum(axis=1)[:, np.newaxis]
    cm_normalized_smote = cm_smote.astype('float') / cm_smote.sum(axis=1)[:, np.newaxis]

    # No SMOTE confusion matrices
    im_no_smote = axes[0, i].imshow(cm_normalized_no_smote, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, i].set_title(f'{name}\n(No SMOTE): Confusion Matrix')
    axes[0, i].set_xlabel('Predicted')
    if i == 0:
        axes[0, i].set_ylabel('Actual')

    # With SMOTE confusion matrices
    im_smote = axes[1, i].imshow(cm_normalized_smote, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, i].set_title(f'{name}\n(With SMOTE): Confusion Matrix')
    axes[1, i].set_xlabel('Predicted')
    if i == 0:
        axes[1, i].set_ylabel('Actual')

    # Add annotations
    for j in range(cm_no_smote.shape[0]):
        for k in range(cm_no_smote.shape[1]):
            axes[0, i].text(k, j, format(cm_normalized_no_smote[j, k], '.2f'), horizontalalignment="center", color="white" if cm_normalized_no_smote[j, k] > 0.5 else "black")
            axes[1, i].text(k, j, format(cm_normalized_smote[j, k], '.2f'), horizontalalignment="center", color="white" if cm_normalized_smote[j, k] > 0.5 else "black")

# Add a colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
fig.colorbar(im_no_smote, cax=cbar_ax, label='Normalized Value')

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

