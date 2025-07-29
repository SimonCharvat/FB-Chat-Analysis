

# Standard imports
import json
import os
import re
import csv
import datetime
from functools import partial

# Third-party imports
import pandas as pd

# Local imports
import visualize
import utils


# Load messages from CSV
messages_df = utils.load_messages_csv("messages.csv")


def extract_group_chat_renames(df: pd.DataFrame):
    """
    Extract rows where 'content' contains 'dal skupině název ',
    extract the group chat name into 'group_chat_name',
    and return only relevant columns.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame with a 'content' column.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with columns:
                      ['sender_name', 'timestamp_ms', 'datetime', 'group_chat_name']
    """
    filtered_df = df[df['content'].str.contains('dal skupině název ', na=False)].copy()
    filtered_df['group_chat_name'] = filtered_df['content'].str.extract(r'dal skupině název (.+?)(?:\.\s*|$)')
    return filtered_df[['sender_name', 'datetime', 'date', 'time', 'group_chat_name']]



# Extract group chat renames
chat_names_df = extract_group_chat_renames(messages_df)

# Create vertical timeline of group chat renames
visualize.draw_group_rename_timeline(chat_names_df)

# Plot user media counts
visualize.plot_and_save_media_counts(messages_df)
visualize.save_media_summary_table(messages_df)
visualize.save_media_user_heatmap(messages_df)

# Analyze message lengths
visualize.plot_avg_message_length_per_user(messages_df)
visualize.plot_message_length_boxplot_per_user(messages_df, ymax=400)
visualize.plot_message_length_histogram_per_user(messages_df, xmax=250)
visualize.plot_message_length_max_table_per_user(messages_df)
visualize.save_longest_messages_to_txt_per_user(messages_df)

# Plot heatmap activity (monthly, hourly)
visualize.plot_day_of_week_heatmap(messages_df)
visualize.plot_hour_of_day_heatmaps(messages_df)

# Plot messages per month
visualize.plot_messages_per_month(messages_df)

# Plot reactions
visualize.plot_reactions_messages_ratio(messages_df)

# Plot call timeline
visualize.draw_call_timeline(messages_df)

# Plot unset messages
visualize.plot_unsent_messages_per_user(messages_df)

# Plot messages by user
visualize.plot_messages_per_sender(messages_df)