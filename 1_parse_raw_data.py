
# TODO:
# - call duration


# Standard imports
import json
import os
import re
import csv
import datetime
from functools import partial

# Third-party imports
import pandas as pd




file_paths = [
    "messages/message_1.json",
    "messages/message_2.json",
    "messages/message_3.json",
    "messages/message_4.json",
    ]



final_column_order = [
    'sender_name',       # who sent the message
    'timestamp_ms',      # original timestamp
    'datetime',          # human-readable timestamp
    'date',              # human-readable date
    'time',              # human-readable time
    'day_of_week',       # human-readable day of week
    'is_unsent',         # flag for unsent messages
    
    'content',           # main message text
    'content_len',       # length of main message text
    'reactions',         # emoji reactions
    'reactions_len',     # number of reactions
    'media',             # summary string of media types
    
    # Media counts
    'photos',
    'videos',
    'gifs',
    'audio_files',
    'files',
    'sticker',
    'share',
    
    # Call duration
    'call_duration',
    
    # Any additional processed/derived columns could follow
]



# fix bad facebook encoding to utf-8
# https://stackoverflow.com/questions/50008296/facebook-json-badly-encoded
fix_mojibake_escapes = partial(
    re.compile(rb'\\u00([\da-f]{2})').sub,
    lambda m: bytes.fromhex(m[1].decode()),
)

def load_messages_file(file_path) -> dict:

    with open(os.path.join(file_path), 'rb') as binary_data:
        repaired = fix_mojibake_escapes(binary_data.read())

    data = json.loads(repaired)
    messages = data["messages"]

    return messages

# Define list of media columns
#media_column_list = ["photos", "videos", "gifs", "audio_files", "share", "files", "sticker"]

# Define mapping from column names to display names
media_column_mapping = {
    'photos': 'Photos',
    'videos': 'Videos',
    'gifs': 'GIFs',
    'audio_files': 'Audio',
    'share': 'FB posts',
    'files': 'Files',
    'sticker': 'Stickers'
}

def summarize_all_media(row):
    parts = []
    for col, display in media_column_mapping.items():
        items = row.get(col)
        if isinstance(items, list) and len(items) > 0:
            parts.append(f"{display} ({len(items)})")
    return ", ".join(parts)

def get_list_count(items):
    if isinstance(items, list):
        return len(items)
    return 0


def reactions_to_string(reactions):
    if isinstance(reactions, list):
        return ''.join(
            re.sub(r'\ufe0f|\u200d|\u2640|\u2642|[\U0001F3FB-\U0001F3FF]', '', r.get('reaction', '')) # Remove decorators like race and gender
            for r in reactions
        )
    return pd.NA


# List of dataframes with the data
file_dataframes = []

for file_path in file_paths:

    def analyse_message(message):
        response = []
        
        try:
            sender = message["sender_name"]
            text = message["content"]
            timestamp = message["timestamp_ms"]
        except:
            return False
        
        date = datetime.datetime.fromtimestamp(timestamp/1000)
        response.append(date)
        response.append(timestamp)
        response.append(sender)
        response.append(text)
        
        return response
    
    # Load messages from file as a dictionary
    message_file = load_messages_file(file_path)
    
    # Convert it to a dataframe and append
    file_dataframes.append(pd.DataFrame(message_file))


# Merge al dataframes
messages_df = pd.concat(file_dataframes, ignore_index=True)

# Add length of message text
messages_df["content_len"] = messages_df.apply(lambda row: len(row["content"]) if isinstance(row["content"], str) else pd.NA, axis=1)

# Create summarization column for media
messages_df["media"] = messages_df.apply(summarize_all_media, axis=1)

# Convert is_unsent to boolean
messages_df['is_unsent'] = messages_df['is_unsent'].astype('boolean').fillna(False)

# Parse timestamp
messages_df["datetime"] = pd.to_datetime(messages_df["timestamp_ms"], unit='ms')
messages_df["date"] = messages_df["datetime"].dt.date
messages_df["time"] = messages_df["datetime"].dt.round('1s').dt.time
messages_df["day_of_week"] = messages_df["datetime"].dt.day_name()

# Convert reactions dictionary to a simple string
messages_df['reactions_len'] = messages_df['reactions'].apply(get_list_count)
messages_df['reactions'] = messages_df['reactions'].apply(reactions_to_string)

# Convert the media columns to pure count
for col_name in media_column_mapping.keys():
    messages_df[col_name] = messages_df[col_name].apply(get_list_count)

# Drop unused columns
messages_df = messages_df.drop(columns=["is_geoblocked_for_viewer", "is_unsent_image_by_messenger_kid_parent"])

# Reorder columns
messages_df = messages_df[final_column_order]

# Rename columns
messages_df = messages_df.rename(columns=media_column_mapping)

# Sort by timestamp chronologically
messages_df = messages_df.sort_values(by='timestamp_ms', ascending=True).reset_index(drop=True)

# Save to CSV
messages_df.to_csv("messages.csv", index=False, sep=";", decimal=".", quoting=csv.QUOTE_ALL)

print(messages_df)