import os
import googleapiclient.discovery
import pandas as pd

from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt

# Set up YouTube API client
api_service_name = "youtube"
api_version = "v3"
api_key = "AIzaSyAHjM4--WZZCqINCUgwbvSoM-K1iwPLpEA"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=api_key)

# Function to search for videos
def search_videos(query, start_date, end_date, max_results=100):
    request = youtube.search().list(
        part="id,snippet",
        type="video",
        q=query,
        videoDefinition="high",
        publishedAfter=start_date,
        publishedBefore=end_date,
        maxResults=max_results,
        regionCode="AU",  # Prioritize search results more relevant to viewers in Australia
        fields="items(id(videoId),snippet(publishedAt,channelId,title,channelTitle))"
    )
    response = request.execute()
    return response['items']

# Function to get video transcripts using youtube_transcript_api
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Check if there's an English transcript, and get it
        if 'en' in [transcript.language_code for transcript in transcript_list]:
            transcript = transcript_list.find_transcript(['en'])
            transcript_text = ' '.join([entry['text'] for entry in transcript.fetch()])
            return transcript_text
        else:
            print(f"No English transcript found for video {video_id}")
            return None
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return None

# Perform search and scrape video titles and transcripts for each year
query = "medicinal cannabis australia"
start_year = 2013
end_year = 2023

video_data = []
transcripts_per_year = {}

for year in range(start_year, end_year + 1):
    start_date = f"{year}-01-01T00:00:00Z"
    end_date = f"{year}-12-31T23:59:59Z"
    print(f"Searching videos for year {year}...")
    videos = search_videos(query, start_date, end_date)
    transcripts_per_year[year] = 0
    
    for video in videos:
        video_id = video['id']['videoId']
        title = video['snippet']['title']
        published_at = video['snippet']['publishedAt']
        transcript = get_transcript(video_id)
        
        if transcript is not None:
            print(f"Successfully retrieved transcript for video '{title}' published on {published_at}")
            transcripts_per_year[year] += 1
        
        video_data.append({'video_id': video_id, 'title': title, 'published_at': published_at, 'transcript': transcript})

df = pd.DataFrame(video_data)
# Add word count columns for title and transcript
df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
df['transcript_word_count'] = df['transcript'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
# Convert the published_at column to datetime objects
df['published_at'] = pd.to_datetime(df['published_at'])
# Create a new column with a standardized date format (e.g., YYYY-MM-DD)
df['date_std'] = df['published_at'].dt.strftime('%Y-%m-%d')
# Save DataFrame to a CSV file
df.to_csv("medicinal_cannabis_videos_by_year.csv", index=False)

# Plot the number of video transcripts retrieved per year
years = list(transcripts_per_year.keys())
transcript_counts = list(transcripts_per_year.values())

plt.bar(years, transcript_counts)
plt.xlabel('Year')
plt.ylabel('Number of Transcripts')
plt.title('Video Transcripts Successfully Retrieved by Year')
plt.xticks(years)
plt.show()

# Print the DataFrame
print(df)

import numpy as np
import seaborn as sns

# Set a seaborn style for better aesthetics
sns.set_style("whitegrid")

# Plot the number of video transcripts retrieved per year
years = list(transcripts_per_year.keys())
transcript_counts = list(transcripts_per_year.values())

# Create a color map to make bars visually appealing
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(years)))

# Plot the bars with custom colors, edge color, and width
plt.bar(years, transcript_counts, color=colors, edgecolor='black', width=0.8)

# Customize x-axis and y-axis labels
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Transcripts', fontsize=12)

# Customize the title
plt.title('Video Transcripts Successfully Retrieved by Year', fontsize=14, fontweight='bold')

# Customize the x-axis ticks and rotate them if necessary
plt.xticks(years, rotation=45)

# Remove the top and right spines for better aesthetics
sns.despine()

plt.ylim(0, 50)

# Show the plot
plt.show()