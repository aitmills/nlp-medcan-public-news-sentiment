# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:26:50 2023

@author: aitmi
"""

import pandas as pd
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

df_raw_abcnewsau = pd.read_csv('df_abc.csv')
df_raw_sbsnewsau = pd.read_csv('df_sbs.csv')
df_raw_theagenewsau = pd.read_csv('df_theage.csv')
df_raw_skynewsau = pd.read_csv('df_skynews.csv')
df_raw_newscomau = pd.read_csv('df_news.csv')
df_raw_smhau = pd.read_csv('df_smh.csv')
df_raw_brisbanetimesau = pd.read_csv('df_smh.csv')
df_raw_mamamiasau = pd.read_csv('df_mamamia.csv')
df_raw_7newsau = pd.read_csv('df_7news.csv')
df_raw_9newsau = pd.read_csv('df_9news.csv')
df_raw_thewestau = pd.read_csv('df_thewest.csv')
df_raw_watodayau = pd.read_csv('df_watoday.csv')
df_raw_perthnowau = pd.read_csv('df_perthnow.csv')
df_raw_hospitalhealthau = pd.read_csv('df_hospitalhealth.csv')
df_raw_monashau = pd.read_csv('df_monash.csv')
df_raw_griffithnewsau = pd.read_csv('df_giffithnews.csv')
df_raw_psychologytoday = pd.read_csv('df_psychologytoday.csv')
df_raw_racgp = pd.read_csv('df_racgp.csv')
df_raw_sydneyuniau = pd.read_csv('df_sydney.csv')
df_raw_theconversation = pd.read_csv('df_theconversation.csv')

#--- Youtube transcript data
df_raw_youtubetranscriptsau = pd.read_csv('medicinal_cannabis_videos_by_year.csv')
df_youtubetranscriptsau = df_raw_youtubetranscriptsau[['title','date_std','transcript','title_word_count','transcript_word_count']]
df_youtubetranscriptsau = df_youtubetranscriptsau.rename(columns={"title": "headline", "transcript": "body", "title_word_count": "headline_word_count", "transcript_word_count":"body_word_count"})


def remove_useless_strings(text, useless_strings):
    # If the input text is not a string, return it unchanged
    if not isinstance(text, str):
        return text
    # Remove line breaks
    text = text.replace('\n', '').replace('\r', '')
    
    for string in useless_strings:
        text = text.replace(string, '')

    return text.strip()

useless_strings = [
    'We’re sorry, this feature is currently unavailable. We’re working to restore it. Please try again later. ',
    ' To join the conversation, please',
    "log in. Don't have an account?"
    "Register Join the conversation, you are commenting as  Logout One of the first Covid-19 jabs offered to Australians has been quietly discontinued, the federal government has confirmed. The delicious easter treat has been recalled amid fears they could’ve been contaminated with metal shavings. A search is currently underway in a state’s high country, where an Irish hiker has reportedly been missing for days. Our Apps",
    "By MAMAMIA TEAM",
    "[Applause] ",
    "[Music] "]

# Create standardised date column
# Function to extract and standardize date
def standardize_date(snippet):
    date_formats = ['%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y']
    
    # Attempt to extract and standardize date from the 'snippet' column
    snippet_date_match = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b", snippet, re.IGNORECASE)
    if snippet_date_match:
        snippet_date = snippet_date_match.group()
        for fmt in date_formats:
            try:
                return datetime.strptime(snippet_date, fmt).strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    # If no date found, check for relative dates like "3 days ago"
    relative_date_match = re.search(r'(\d+)\s+days? ago', snippet, re.IGNORECASE)
    if relative_date_match:
        days_ago = int(relative_date_match.group(1))
        date_obj = datetime.now() - timedelta(days=days_ago)
        return date_obj.strftime('%Y-%m-%d')

    # Return None if no date found
    return None

# Function to extract words and remove non-word characters
def clean_snippet(snippet):
    # Remove the date from the snippet
    date_pattern = r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b"
    snippet_no_date = re.sub(date_pattern, '', snippet, flags=re.IGNORECASE)
    
    # Extract words and remove non-word characters
    words = re.findall(r'\b\w+\b', snippet_no_date)
    return ' '.join(words)

# Function to count words in a given text
def word_count(text):
    if isinstance(text, str):
        return len(text.split())
    else:
        return 0  # or any other default value you want

# List of dataframes
dataframe_names = [
    'df_raw_abcnewsau',
    'df_raw_sbsnewsau',
    'df_raw_theagenewsau',
    'df_raw_skynewsau',
    'df_raw_newscomau',
    'df_raw_smhau',
    'df_raw_brisbanetimesau',
    'df_raw_mamamiasau',
    'df_raw_7newsau',
    'df_raw_9newsau',
    'df_raw_thewestau',
    'df_raw_watodayau',
    'df_raw_perthnowau',
    'df_raw_hospitalhealthau',
    'df_raw_monashau',
    'df_raw_griffithnewsau',
    'df_raw_psychologytoday',
    'df_raw_racgp',
    'df_raw_sydneyuniau',
    'df_raw_theconversation']

# Create 'date_std', 'snippet_std', and 'snippet_std_word_count' columns in each dataframe
for df_name in dataframe_names:
    exec(f"{df_name}['date_std'] = {df_name}['snippet'].apply(standardize_date)")
    exec(f"{df_name}['snippet_std'] = {df_name}['snippet'].apply(clean_snippet)")
    exec(f"{df_name}['snippet_std_word_count'] = {df_name}['snippet_std'].apply(word_count)")
    
# Where there are relative dates they are not being removed from the snippet_std

# dataframes = [df_raw_abcnewsau, df_raw_sbsnewsau, df_raw_theagenewsau,
#                   df_raw_skynewsau, df_raw_newscomau, df_raw_smhau,
#                   df_raw_brisbanetimesau, df_raw_mamamiasau, df_raw_7newsau,
#                   df_raw_9newsau, df_raw_thewestau, df_raw_watodayau, df_raw_perthnowau, df_youtubetranscriptsau]
dataframes = [
    (df_raw_abcnewsau, "abcnewsau"),
    (df_raw_sbsnewsau, "sbsnewsau"),
    (df_raw_theagenewsau, "theagenewsau"),
    (df_raw_skynewsau, "skynewsau"),
    (df_raw_newscomau, "newscomau"),
    (df_raw_smhau, "smhau"),
    (df_raw_brisbanetimesau, "brisbanetimesau"),
    (df_raw_mamamiasau, "mamamiasau"),
    (df_raw_7newsau, "7newsau"),
    (df_raw_9newsau, "9newsau"),
    (df_raw_thewestau, "thewestau"),
    (df_raw_watodayau, "watodayau"),
    (df_raw_perthnowau, "perthnowau"),
    (df_youtubetranscriptsau, "youtubetranscriptsau"),
    (df_raw_hospitalhealthau, "hospitalhealthau"),
    (df_raw_monashau, "monashau"),
    (df_raw_griffithnewsau, "griffithnewsau"),
    (df_raw_psychologytoday, "psychologytoday"),
    (df_raw_racgp, "racgp"),
    (df_raw_sydneyuniau, "sydneyuniau"),
    (df_raw_theconversation, "theconversation"),
]

for df, name in dataframes:
    df['source'] = name

# Combine the data frames into one
combined_df = pd.concat([df for df, _ in dataframes])

combined_df['body'] = combined_df['body'].apply(lambda x: remove_useless_strings(x, useless_strings))
combined_df['snippet_std'] = combined_df['snippet_std'].apply(lambda x: remove_useless_strings(x, useless_strings))
combined_df['headline'] = combined_df['headline'].apply(lambda x: remove_useless_strings(x, useless_strings))


# Recalculate the 'body_word_count' column
combined_df['body_word_count'] = combined_df['body'].apply(word_count)
# Remove dupilicates
combined_df = combined_df.drop_duplicates()
# Remove 

# Filter out rows with less than 100 words in the 'body_word_count' column
combined_body_over100 = combined_df[combined_df['body_word_count'] >= 100]
combined_snippets = combined_df[combined_df['snippet_std'].notnull()]
combined_headlines = combined_df[combined_df['headline'].notnull()]

plot_df = combined_df[combined_df['source'] != 'youtubetranscriptsau']
plt.figure(figsize=(16, 6))
sns.boxplot(x='source', y='body_word_count', data=plot_df, showfliers=False)
plt.title('Boxplots of Body Word Count for Different Data Frames (Excluding youtubetranscriptsau)')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()



plt.figure(figsize=(16, 6))
sns.boxplot(x='source', y='body_word_count', data=combined_body_over100)
plt.title('Boxplots of Body Word Count for Different Data Frames')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

plt.figure(figsize=(16, 6))
sns.boxplot(x='source', y='snippet_std_word_count', data=combined_snippets)
plt.title('Boxplots of Snippet Word Count for Different Data Frames')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

plt.figure(figsize=(16, 6))
sns.boxplot(x='source', y='headline_word_count', data=combined_headlines)
plt.title('Boxplots of Headline Word Count for Different Data Frames')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

# Drop rows with 'source' values 'sbsnewsau' and '9newsau'
combined_df = combined_df[~combined_df['source'].isin(['sbsnewsau', '9newsau'])]

# Exclude 'youtubetranscriptsau' and create a new data frame for plotting
plot_df = combined_df[combined_df['source'] != 'youtubetranscriptsau']

# Create the boxplot without outliers
plt.figure(figsize=(16, 6))
sns.boxplot(x='source', y='body_word_count', data=plot_df, showfliers=False)
plt.title('Boxplots of Body Word Count for Different Data Frames (Excluding youtubetranscriptsau, sbsnewsau, 9newsau, and Outliers)')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

# Save to CSV file
combined_body_over100.to_csv('combined_body_over100.csv',index = False)
combined_snippets.to_csv('combined_snippets.csv',index = False)
combined_headlines.to_csv('combined_headlines.csv',index = False)
combined_df.to_csv('combined_df.csv',index = False)
