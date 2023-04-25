# nlp-medcan-public-news-sentiment
> Analysing Public News Sentiment on Medicinal Cannabis in Australia using Natural Language Processing

## File Summary
- aitm_a3_call_webscraper.py
Iterates through a list of Australian news websites, calling a web scraping function main() from the imported module aitm_A3_GCSE_scrapewebdata for each website domain. The function scrapes news articles related to medicinal cannabis and returns a pandas DataFrame containing the scraped information.

The script creates a separate DataFrame variable for each domain and saves the DataFrame to a CSV file named with the domain's prefix. It includes a 10-second sleep between each domain to reduce the load on the websites being scraped. Finally, the script calculates and prints the total elapsed time taken to scrape the data from all websites in the list.

- aitm_A3_GCSE_scrapewebdata.py
Web scraper that searches for Australian news articles related to medicinal cannabis and scrapes the content of the articles. It uses the Google Custom Search API to find relevant articles and BeautifulSoup to parse the HTML content of the web pages. The script also has functionality to extract text from PDF files using the PyPDF2 library. Finally, it stores the scraped data, including URL, headline, date, body, and snippet of each article, in a pandas DataFrame along with the word count for headlines and bodies. The main function accepts a domain parameter, which is set to 'theconversation.com' by default, to specify the website to search for articles.

- aitm_A3_youtube_totalvideos_yearly.py
Uses the YouTube Data API to search for videos related to the term 'medicinal cannabis australia' published within a specific year range (2013 to the current year). It collects the video IDs for each year and then retrieves the transcripts for those videos using the YouTube Transcript API. The main function iterates through each year in the specified range and prints the total number of transcripts retrieved for each year.

- aitm_A3_youtube_transcript_scrape.py
Uses the YouTube Data API to search for high-definition videos related to the term 'medicinal cannabis australia' published within a specific year range (2013 to 2023) and prioritizes results relevant to viewers in Australia. For each video found, it retrieves the video title, publication date, and the English transcript (if available) using the YouTube Transcript API. The script then creates a DataFrame with the collected data, adds columns for title and transcript word counts, converts the publication date column to datetime objects, and standardizes the date format. The DataFrame is saved as a CSV file.

The script also plots the number of video transcripts retrieved per year using matplotlib and seaborn libraries, displaying the plot as a bar chart with custom colors, edge colors, and axis labels. It removes the top and right spines of the plot for better aesthetics, and shows the plot with a specified y-axis limit.

- aitm_A3_load_clean.py
Reads in multiple CSV files containing news data from different Australian news sources and YouTube transcripts. The script then processes the data by cleaning the text and standardizing the date format.

After the cleaning and processing, the script calculates word counts for various fields such as headlines, body, and snippets. The data from each source is then combined into a single DataFrame, and the body word count is used to filter the combined data to include only entries with more than 100 words in the body. The script also generates boxplots of word counts for different sources.

Lastly, the script removes some sources from the combined DataFrame, generates another boxplot without outliers, and saves the resulting DataFrames to CSV files.

- aitm_A3_textblob_sentiment.py
	1. Imports necessary libraries such as pandas, numpy, matplotlib, textblob, and datetime.
	2. Reads a CSV file named 'combined_df.csv' into a DataFrame named combined_df.
	3. Converts the 'date_std' column to datetime format and filters the DataFrame to include only rows with 'date_std' values greater than or equal to January 1, 2013.
	4. Defines two functions, sentiment_score and sentiment_label, to calculate sentiment scores and labels for the given text.
	5. Calculates sentiment scores for the 'headline', 'snippet_std', and 'body' columns of the DataFrame, and computes the average of the three scores as the composite sentiment.
	6. Assigns sentiment labels to the sentiment scores using the previously defined sentiment_label function.
	7. Concatenates the 'headline', 'snippet_std', and 'body' columns into a new column named 'composite_text'.
	8. Exports the modified DataFrame to a CSV file named 'combined_df_textblob_sentiment.csv'.
	9. Plots the cumulative sentiment scores over time for 'headline_sentiment', 'snippet_sentiment', 'body_sentiment', and 'composite_sentiment', and displays the plots with and without annotations.
	10. Creates a bar plot showing the counts of positive and negative sentiment labels per year.
	The script processes a dataset containing text data, such as headlines, snippets, and body text, and performs sentiment analysis using the TextBlob library. It then visualizes the sentiment analysis results in multiple ways, including cumulative sentiment scores over time and counts of sentiment labels per year.

- aitm_A3_TradML.py
This code is designed to build and evaluate several machine learning models for text classification using various algorithms, such as Logistic Regression, Support Vector Machines, Naive Bayes, Random Forest, and LightGBM. The models are trained on a dataset with text and labels, and their performance is evaluated using accuracy, F1-score, and other metrics. The code also incorporates SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset before training and compares the model performance with and without SMOTE. Finally, the results are visualized with bar plots, ROC curves, and confusion matrices.
	1. Import required libraries and modules.
	2. Read the dataset into a DataFrame.
	3. Vectorize the text data using the TF-IDF vectorizer.
	4. Split the dataset into training and testing sets.
	5. Define the models and their hyperparameters for grid search.
	6. Create a function to train and evaluate models with or without SMOTE.
	7. Evaluate models without SMOTE and with SMOTE.
	8. Print evaluation results for each model.
	9. Visualize accuracy, F1-score, and training time for all models.
	10. Plot ROC curves for each model.
	11. Visualize confusion matrices for all models.

- aitm_A3_LTSMRNN.py
This script is for sentiment analysis using an LSTM neural network. It starts by importing necessary libraries and reading the dataset. After preprocessing the text, the script performs some exploratory data analysis and visualizes the data. Then, it extracts features and splits the dataset into training and testing sets.

The script defines an LSTM model and trains it using the training data. The model's performance is then evaluated by plotting accuracy versus epochs and a receiver operating characteristic (ROC) curve. The script also uses Keras Tuner to find the best hyperparameters for the model. Finally, it outputs plots comparing different trials' hyperparameters and validation accuracies, as well as ROC curves for each trial.

Here's a summary of each section:

	1. Load data: Read the CSV file and preprocess it.
	2. Preprocessing: Clean the text and convert sentiment labels to binary classes.
	3. EDA plots: Plot the distribution of composite text lengths and word frequencies.
	4. Feature extraction: Tokenize the text and pad the sequences.
	5. Define LSTM model: Create an LSTM neural network architecture.
	6. Train model: Train the model using the training data.
	7. Evaluate: Plot accuracy vs epochs and the ROC curve.
	8. Tune model: Use Keras Tuner to find the best hyperparameters for the model.
Output plots: Display plots of hyperparameters vs validation accuracy and ROC curves for all Keras Tuner trials.

