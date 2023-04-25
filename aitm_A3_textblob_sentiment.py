# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:10:22 2023

@author: Adam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime

combined_df = pd.read_csv('combined_df.csv')

combined_df['date_std'] = pd.to_datetime(combined_df['date_std'])
# Assuming your date_std column is already in datetime format
combined_df = combined_df[combined_df['date_std'] >= datetime(2013, 1, 1)]

def sentiment_score(text):
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    else:
        return 0
def sentiment_label(score, threshold=0):
    if score > threshold:
        return 'positive'
    else:
        return 'negitive'
# Sentiment Scoring
combined_df['headline_sentiment'] = combined_df['headline'].apply(sentiment_score)
combined_df['snippet_sentiment'] = combined_df['snippet_std'].apply(sentiment_score)
combined_df['body_sentiment'] = combined_df['body'].apply(sentiment_score)
combined_df['composite_sentiment'] = (combined_df['headline_sentiment'] + combined_df['snippet_sentiment'] + combined_df['body_sentiment']) / 3
# Sentiment labelling
combined_df['headline_label'] = combined_df['headline_sentiment'].apply(sentiment_label)
combined_df['snippet_label'] = combined_df['snippet_sentiment'].apply(sentiment_label)
combined_df['body_label'] = combined_df['body_sentiment'].apply(sentiment_label)
combined_df['composite_label'] = combined_df['composite_sentiment'].apply(sentiment_label)

combined_df['composite_text'] = combined_df['headline'].astype(str) + ' ' + combined_df['snippet_std'].astype(str) + ' ' + combined_df['body'].astype(str)

# OUTPUT CSV
combined_df.to_csv('combined_df_textblob_sentiment.csv',index = False)


# PLOTS
cumulative_sentiments = combined_df.groupby('date_std')[['headline_sentiment', 'snippet_sentiment', 'body_sentiment', 'composite_sentiment']].sum().cumsum()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_sentiments.index, cumulative_sentiments['headline_sentiment'], label='Headline Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['snippet_sentiment'], label='Snippet Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['body_sentiment'], label='Body Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['composite_sentiment'], label='Composite Sentiment')

plt.title('Cumulative Sentiment Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Sentiment Score')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_sentiments.index, cumulative_sentiments['headline_sentiment'], label='Headline Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['snippet_sentiment'], label='Snippet Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['body_sentiment'], label='Body Sentiment')
plt.plot(cumulative_sentiments.index, cumulative_sentiments['composite_sentiment'], label='Composite Sentiment')

# Fill between headline and snippet sentiment
plt.fill_between(cumulative_sentiments.index,
                 cumulative_sentiments['headline_sentiment'],
                 cumulative_sentiments['snippet_sentiment'],
                 where=(cumulative_sentiments['snippet_sentiment'] > cumulative_sentiments['headline_sentiment']),
                 color='lightblue', interpolate=True)

# First annotation
arrow_date_1 = datetime(2016, 11, 1)
arrow_y_1 = cumulative_sentiments.loc[arrow_date_1, 'composite_sentiment']
plt.annotate('Federal law change takes effect', xy=(arrow_date_1, arrow_y_1), xytext=(arrow_date_1 - pd.DateOffset(months=36), arrow_y_1 + 20),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, color='black')

# Second annotation
arrow_date_2 = datetime(2016, 2, 24)
arrow_y_2 = cumulative_sentiments.loc[arrow_date_2, 'composite_sentiment']
plt.annotate('Federal law amendment passed', xy=(arrow_date_2, arrow_y_2), xytext=(arrow_date_2 - pd.DateOffset(months=40), arrow_y_2 + 15),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, color='black')

# Third annotation
arrow_date_3 = datetime(2019, 9, 25)
arrow_y_3 = cumulative_sentiments.loc[arrow_date_3, 'composite_sentiment']
plt.annotate('ACT law amendment passed', xy=(arrow_date_3, arrow_y_3), xytext=(arrow_date_3 - pd.DateOffset(months=55), arrow_y_3 + 15),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, color='black')


plt.title('Cumulative Sentiment Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Sentiment Score')
plt.legend()
plt.show()

# Group the data by year and sentiment label, and count the occurrences of each label
sentiment_counts_by_year = combined_df.groupby([combined_df['date_std'].dt.year, 'composite_label']).size().unstack()

# Create a bar plot with separate bars for each sentiment label and custom colors
ax = sentiment_counts_by_year.plot(kind='bar', figsize=(15, 7), color=['red', 'green'], alpha=0.7)

# Set x-axis and y-axis labels
ax.set_xlabel('Year')
ax.set_ylabel('Count of Sentiment Labels')

# Add a title
ax.set_title('Counts of Positive and Negative Sentiments Per Year')

# Add a legend
ax.legend(['Negative', 'Positive'])

# Display the plot
plt.show()

