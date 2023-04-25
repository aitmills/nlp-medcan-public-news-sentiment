# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:06:32 2023

@author: Adam
"""

import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi

API_KEY = 'AIzaSyAHjM4--WZZCqINCUgwbvSoM-K1iwPLpEA'
SEARCH_TERM = 'medicinal cannabis australia'
BASE_URL = 'https://www.googleapis.com/youtube/v3/search'

def get_video_ids_for_year(year):
    start_date = datetime(year, 1, 1).strftime('%Y-%m-%dT00:00:00Z')
    end_date = datetime(year, 12, 31).strftime('%Y-%m-%dT23:59:59Z')

    params = {
        'key': API_KEY,
        'part': 'id',
        'type': 'video',
        'q': SEARCH_TERM,
        'publishedAfter': start_date,
        'publishedBefore': end_date,
        'maxResults': 30000,
    }

    video_ids = []
    nextPageToken = None

    while True:
        if nextPageToken:
            params['pageToken'] = nextPageToken

        response = requests.get(BASE_URL, params=params)
        data = response.json()
        video_ids.extend([item['id']['videoId'] for item in data['items']])

        nextPageToken = data.get('nextPageToken', None)
        if not nextPageToken:
            break

    return video_ids

def get_transcripts_for_year(year):
    video_ids = get_video_ids_for_year(year)
    transcripts = {}

    for video_id in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcripts[video_id] = transcript
        except Exception as e:
            print(f"Error retrieving transcript for video ID {video_id}: {e}")

    return transcripts

def main():
    start_year = 2013
    end_year = datetime.now().year

    for year in range(start_year, end_year + 1):
        transcripts = get_transcripts_for_year(year)
        print(f'Total transcripts retrieved for {year}: {len(transcripts)}')

if __name__ == '__main__':
    main()
