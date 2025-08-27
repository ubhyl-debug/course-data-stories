import pandas as pd
from requests.exceptions import HTTPError
from tenacity import RetryError  
import os
from datetime import datetime, timedelta
from ddbapi import zp_pages



def article_extractor(search_dict):
    try:
        result = zp_pages(
            language=search_dict['language'], 
            publication_date=f"[{search_dict['date_begin']}T12:00:00Z TO {search_dict['date_end']}T12:00:00Z]"
        )
        # Just in case it returns something weird instead of raising
        if not isinstance(result, pd.DataFrame):
            print(f"Warning: zp_pages returned {type(result)}, expected DataFrame.")
            return pd.DataFrame()
        return result

    except (HTTPError, RetryError) as e:
        print(f"Handled error for {search_dict['date_begin']} to {search_dict['date_end']}: {e}")
    except Exception as e:
        print(f"Unexpected error of type {type(e)} for {search_dict['date_begin']} to {search_dict['date_end']}: {e}")
    
    return pd.DataFrame()




year_start = 1920
year_end = 1945
start_month = 1 
output_dir = "data_deutsches_zeitungsportal_misc"
os.makedirs(output_dir, exist_ok=True)

for year in range(year_start, year_end + 1):
    # Start from April if it's the first year; otherwise from January
    month_range = range(start_month, 13) if year == year_start else range(1, 13)

    for month in month_range:
        date_begin = f"{year}-{month:02d}-01"
        if month == 12:
            date_end = f"{year + 1}-01-01"
        else:
            date_end = f"{year}-{month + 1:02d}-01"

        search_dict = {
            'language': 'ger',
            'date_begin': date_begin,
            'date_end': date_end
        }

        print(f"Extracting articles from {date_begin} to {date_end}...")
        df_challenge = article_extractor(search_dict)

        if not df_challenge.empty:
            file_name = f"newspapers_{search_dict['language']}_{year}_{month:02d}.pkl"
            df_challenge.to_pickle(os.path.join(output_dir, file_name))
            print(f"Saved {file_name}")
        else:
            print(f"No articles found for {year}-{month:02d}")
