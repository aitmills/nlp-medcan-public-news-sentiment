# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:49:14 2023

@author: aitmi
"""
import pandas as pd
from aitm_A3_GCSE_scrapewebdata import main
import time
start_time = time.time()

# List of popular australian news websites
aus_news_sites = ['theconversation.com']

#COMPLETED:
    #'9news.com.au', 'crikey.com.au','7news.com.au','abc.net.au', 'sbs.com.au', 'theaustralian.com.au','theage.com.au', 'skynews.com.au', 'news.com.au','junkee.com'
# PROBLEMATIC - headline key error
    # 'theguardian.com/au'

for domain in aus_news_sites:
    print("Starting..." + domain)
    df = main(domain)

    # Create a DataFrame variable for each domain 
    var_name = 'df_' + domain.split('.')[0]
    exec(f"{var_name} = df")
    print("dataframe: " + var_name)
    # Save the DataFrame with the suffix of the list item
    suffix = domain.split('.')[0]
    output_filename = f'df_{suffix}.csv'
    print("Output csv file" + str(output_filename))
    df.to_csv(output_filename, index=False)
    print("Have a 10 second rest because scraping is super hard work!")
    time.sleep(10)
    
elapsed_time = time.time() - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes = remainder // 60
print(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}")