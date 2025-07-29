

# Standard imports
import json
import os
import re
import csv
import datetime
from functools import partial

# Third-party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

media_column_list = ['Photos', 'Videos', 'GIFs', 'Audio', 'FB posts', 'Files', 'Stickers']

def draw_group_rename_timeline(df: pd.DataFrame, output_path="outputs/group_rename_timeline.png"):
    df_sorted = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, len(df_sorted) * 0.8))
    ax.set_ylim(-1, len(df_sorted))
    ax.set_xlim(0, 1)
    ax.axis("off")

    ax.plot([0.5, 0.5], [0, len(df_sorted) - 1], color="gray", linewidth=2)

    for i, row in df_sorted.iterrows():
        y: int = i
        dt_str = row["datetime"].strftime("%Y-%m-%d %H:%M")
        group_name = row["group_chat_name"]
        sender = row["sender_name"]

        # Dot marker
        ax.plot(0.5, y, "o", color="tab:blue", markersize=10)

        # Left: datetime
        ax.text(0.48, y, dt_str, va="center", ha="right", fontsize=9)

        # Right: group name in bold
        ax.text(0.52, y + 0.1, group_name, va="bottom", ha="left", fontsize=10, fontweight="bold")

        # Right: sender in italic, below group name
        ax.text(0.52, y - 0.1, sender, va="top", ha="left", fontsize=9, fontstyle="italic", color="gray")

    plt.title("Group Chat Rename Timeline", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_and_save_media_counts(messages_df):
    """
    Generate two bar charts from messages_df which already has renamed media columns:
    1) Stacked media counts by user and media type
    2) Total media counts by user (all types summed, with labels)
    
    Saves plots separately in 'outputs/' folder.
    """
    media_cols = [col for col in media_column_list if col in messages_df.columns]
    if not media_cols:
        raise ValueError("No media columns found in DataFrame")

    df_counts = messages_df[['sender_name'] + media_cols].copy()
    agg = df_counts.groupby('sender_name').sum()
    total_data = agg.sum(axis=1).sort_values(ascending=False)
    agg = agg.loc[total_data.index]

    os.makedirs("outputs", exist_ok=True)

    # Plot 1: stacked media counts by type
    plt.figure(figsize=(11, 8))
    agg.plot(kind='bar', stacked=True, colormap='tab20')
    plt.title("Media Counts by User and Type")
    plt.xlabel("Sender Name")
    plt.ylabel("Count of Media Files")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Media Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/media_by_type.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: total media counts with labels
    plt.figure(figsize=(15, 8))
    ax = total_data.plot(kind='bar', color='tab:blue')
    plt.title("Total Media Counts by User")
    plt.xlabel("Sender Name")
    plt.ylabel("Count of Media Files")
    plt.xticks(rotation=45, ha='right')

    # Add value labels on top
    for i, value in enumerate(total_data):
        ax.text(i, value + 0.5, str(int(value)), ha='center', va='bottom', fontsize=13)

    plt.tight_layout()
    plt.savefig("outputs/media_total.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_media_summary_table(messages_df):
    """
    Creates and saves a summary table image showing total counts per media type.
    """

    media_cols = [col for col in media_column_list if col in messages_df.columns]
    if not media_cols:
        raise ValueError("No media columns found in DataFrame")

    # Sum over all messages
    summary = messages_df[media_cols].sum().astype(int).reset_index()
    summary.columns = ['Media Type', 'Total Count']
    summary = summary.sort_values('Total Count', ascending=False)

    # Plot table
    fig, ax = plt.subplots(figsize=(6, len(summary) * 0.5 + 1))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Total Media Sent by Type", fontsize=14, weight='bold', pad=10)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/media_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_media_user_heatmap(messages_df):
    """
    Creates and saves a heatmap image showing media usage by user and type.
    The color intensity is normalized per media type (column-wise).
    """
    media_column_list = ['Photos', 'Videos', 'GIFs', 'Audio', 'FB posts', 'Files', 'Stickers']
    media_cols = [col for col in media_column_list if col in messages_df.columns]

    if not media_cols:
        raise ValueError("No media columns found in DataFrame")

    # Aggregate counts by sender_name
    df_counts = messages_df[['sender_name'] + media_cols].copy()
    agg = df_counts.groupby('sender_name').sum()

    # Normalize each column independently for relative color scale
    normalized = agg.copy()
    for col in media_cols:
        col_max = agg[col].max()
        if col_max > 0:
            normalized[col] = agg[col] / col_max
        else:
            normalized[col] = 0  # Avoid division by zero

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(len(media_cols) * 1.2 + 2, len(agg) * 0.5 + 1.5))
    sns.heatmap(
        normalized[media_cols],
        annot=agg[media_cols],
        fmt="d",
        cmap="YlGnBu",
        cbar_kws={'label': 'Relative Intensity (per type)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title("Media Sent by User and Type (Heatmap)", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Media Type")
    ax.set_ylabel("Sender Name")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/media_user_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_avg_message_length_per_user(messages_df):
    """
    Plots and saves a vertical bar chart of average message length (content_len) per user.
    Ignores messages with content_len <= 0 or NaN.
    """
    # Filter out invalid content_len values
    valid_df = messages_df[(messages_df['content_len'].notna()) & (messages_df['content_len'] > 0)]

    # Group by sender and calculate average length
    avg_lengths = valid_df.groupby('sender_name')['content_len'].mean().sort_values(ascending=False)

    if avg_lengths.empty:
        print("No valid content_len data to plot.")
        return

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Plot
    plt.figure(figsize=(11, 7))
    ax = avg_lengths.plot(kind='bar', color='tab:purple')

    # Add labels
    plt.title("Average Message Length per User", fontsize=14, weight='bold')
    plt.xlabel("Sender Name")
    plt.ylabel("Average Length (characters)")
    plt.xticks(rotation=45, ha='right')

    # Add numeric labels on top of bars
    for i, value in enumerate(avg_lengths):
        ax.text(i, value + 1, f"{value:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/message_length_average.png", dpi=300, bbox_inches='tight')
    plt.close()




def plot_message_length_boxplot_per_user(messages_df: pd.DataFrame, ymax=None):
    """
    Creates and saves a boxplot of message length (content_len) per user.
    Ignores rows where content_len is NaN or 0.
    """
    # Filter valid messages
    valid_df = messages_df[(messages_df['content_len'].notna()) & (messages_df['content_len'] > 0)]

    if valid_df.empty:
        print("No valid content_len data to plot.")
        return

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Plot boxplot (with hue same as x to satisfy seaborn 0.14+ warning)
    plt.figure(figsize=(12, 12))
    ax = sns.boxplot(
        data=valid_df,
        x='sender_name',
        y='content_len',
        hue='sender_name',         # Assign x also to hue
        palette='Set2',
        legend=False               # Avoid duplicate legend
    )

    # Titles and labels
    plt.title("Distribution of Message Length per User", fontsize=14, weight='bold')
    plt.xlabel("Sender Name")
    plt.ylabel("Message Length (characters)")

    if ymax:
        plt.ylim(0, ymax)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


    # Save figure
    plt.savefig("outputs/message_length_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_message_length_histogram_per_user(messages_df, xmax:int=None):
    """
    Plots and saves overlayed KDE (smoothed histogram) curves of message lengths per user.
    Filters out rows with NaN or 0 content_len.
    """
    # Filter valid messages
    valid_df = messages_df[(messages_df['content_len'].notna()) & (messages_df['content_len'] > 0)]

    if valid_df.empty:
        print("No valid content_len data to plot.")
        return

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(12, 7))

    # Use seaborn's kdeplot to overlay curves per user
    for sender in valid_df['sender_name'].unique():
        subset = valid_df[valid_df['sender_name'] == sender]
        if len(subset) > 1:  # KDE requires at least 2 points
            sns.kdeplot(
                data=subset,
                x='content_len',
                label=sender,
                fill=False,
                common_norm=False,
                linewidth=2,
            )

    if xmax:
        plt.xlim(0, xmax)

    plt.title("Smoothed Distribution of Message Length per User", fontsize=14, weight='bold')
    plt.xlabel("Message Length (characters)")
    plt.ylabel("Density")
    plt.legend(title="Sender", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/message_length_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_message_length_max_table_per_user(messages_df):
    """
    Generates and saves a table as an image showing the maximum message length
    per user and the corresponding datetime.
    """
    # Filter valid messages
    valid_df = messages_df[(messages_df['content_len'].notna()) & (messages_df['content_len'] > 0)].copy()

    if valid_df.empty:
        print("No valid content_len data to summarize.")
        return

    # For each user, find the row with the max message length
    max_rows = valid_df.loc[valid_df.groupby('sender_name')['content_len'].idxmax()]

    # Create summary DataFrame
    summary_df = max_rows[['sender_name', 'content_len', 'datetime']].copy()
    summary_df.columns = ['Sender Name', 'Max Length', 'Date']
    summary_df = summary_df.sort_values('Max Length', ascending=False).reset_index(drop=True)

    # Format datetime
    summary_df['Date'] = summary_df['Date'].dt.strftime('%Y-%m-%d %H:%M')

    # Plot table
    fig, ax = plt.subplots(figsize=(8, len(summary_df) * 0.5 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Longest Message per User", fontsize=14, weight='bold', pad=10)
    plt.tight_layout()

    # Save
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/message_length_max_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_longest_messages_to_txt_per_user(messages_df):
    """
    Extracts the longest message per user and saves it to a plain text file.
    Includes sender name, length, datetime, and message content.
    """
    # Filter valid content
    valid_df = messages_df[
        (messages_df['content_len'].notna()) &
        (messages_df['content_len'] > 0) &
        (messages_df['content'].notna())
    ].copy()

    if valid_df.empty:
        print("No valid messages to export.")
        return

    # Find the longest message per user
    max_rows = valid_df.loc[valid_df.groupby('sender_name')['content_len'].idxmax()]

    # Sort by length descending
    max_rows = max_rows.sort_values('content_len', ascending=False)

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Write to text file
    with open("outputs/message_length_max_full_text.txt", "w", encoding="utf-8") as f:
        for _, row in max_rows.iterrows():
            f.write(f"Sender: {row['sender_name']}\n")
            f.write(f"Length: {row['content_len']} characters\n")
            f.write(f"Date:   {row['datetime'].strftime('%Y-%m-%d %H:%M')}\n")
            f.write("Message:\n")
            f.write(row['content'].strip() + "\n")
            f.write("-" * 50 + "\n\n")



def plot_day_of_week_heatmap(messages_df: pd.DataFrame):
    """
    Generates and saves a heatmap of message counts by user and day of the week.
    Color intensity is normalized per user (row-wise).
    """
    if 'sender_name' not in messages_df.columns or 'day_of_week' not in messages_df.columns:
        raise ValueError("messages_df must contain 'sender_name' and 'day_of_week' columns.")

    # Ensure weekday order
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    messages_df['day_of_week'] = pd.Categorical(messages_df['day_of_week'], categories=ordered_days, ordered=True)

    # Count messages
    heatmap_data = messages_df.groupby(['sender_name', 'day_of_week'], observed=False).size().unstack(fill_value=0)

    # Reorder columns
    heatmap_data = heatmap_data[ordered_days]

    # Normalize row-wise (per user)
    normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0).fillna(0)

    # Plot
    plt.figure(figsize=(11, 7))
    sns.heatmap(
        normalized,
        annot=heatmap_data,  # Show absolute values
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Relative Intensity (per user)'}
    )

    plt.title("Messages per Day (Relative to User Activity)", fontsize=14, weight='bold', pad=12)
    plt.xlabel("Day of the Week")
    plt.ylabel("Sender Name")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/day_of_week_heatmap_per_user.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_hour_of_day_heatmaps(messages_df: pd.DataFrame):
    """
    Generates and saves two heatmaps of message counts by user and hour of the day:
    1) Colors normalized per user (row-wise)
    2) Colors normalized per hour (column-wise)

    Assumes 'sender_name' and 'time' (datetime.time) columns exist.
    """
    if 'sender_name' not in messages_df.columns or 'time' not in messages_df.columns:
        raise ValueError("messages_df must contain 'sender_name' and 'time' columns.")

    # Extract hour from time column
    df = messages_df.copy()
    df['hour'] = df['time'].apply(lambda t: t.hour if pd.notnull(t) else None)
    df = df[df['hour'].notna()]

    # # Counts each day/hour combination - counts each chatting session only once
    # df = df.drop_duplicates(['sender_name', 'hour', 'date'])

    # Count messages by user and hour
    heatmap_data = df.groupby(['sender_name', 'hour'], observed=False).size().unstack(fill_value=0)

    # Ensure all 24 hours present
    all_hours = list(range(24))
    heatmap_data = heatmap_data.reindex(columns=all_hours, fill_value=0)

    # === Heatmap 1: normalize per user (row-wise) ===
    normalized_per_user = heatmap_data.div(heatmap_data.max(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        normalized_per_user,
        annot=heatmap_data,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Relative Intensity (per user)'}
    )
    plt.title("Messages per Hour (Normalized Per User)", fontsize=16, weight='bold', pad=12)
    plt.xlabel("Hour of Day")
    plt.ylabel("Sender Name")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/hour_of_day_heatmap_per_user.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === Heatmap 2: normalize per hour (column-wise) ===
    normalized_per_hour = heatmap_data.div(heatmap_data.max(axis=0), axis=1).fillna(0)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        normalized_per_hour,
        annot=heatmap_data,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Relative Intensity (per hour)'}
    )
    plt.title("Messages per Hour (Normalized Per Hour)", fontsize=16, weight='bold', pad=12)
    plt.xlabel("Hour of Day")
    plt.ylabel("Sender Name")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("outputs/hour_of_day_heatmap_per_hour.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_messages_per_month(messages_df):
    """
    Generates and saves two plots:
    1) Total messages per month (all users combined)
    2) Stacked messages per month by user

    X-axis shows months (01, 02, ...), years displayed below grouped months.
    """
    # Ensure datetime is datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(messages_df['datetime']):
        messages_df['datetime'] = pd.to_datetime(messages_df['datetime'], errors='coerce')

    df = messages_df.dropna(subset=['datetime']).copy()
    df['year_month'] = df['datetime'].dt.to_period('M').dt.to_timestamp()

    # Aggregate totals
    total_per_month = df.groupby('year_month').size()
    stacked = df.groupby(['year_month', 'sender_name']).size().unstack(fill_value=0)
    stacked = stacked[stacked.sum().sort_values(ascending=False).index]

    # Helper for x-axis labels (month only)
    months_str = total_per_month.index.strftime('%m')

    # Helper for years, repeated per month
    years = total_per_month.index.year

    # Find where year changes to place ticks for year labels
    year_change_idx = np.where(years[:-1] != years[1:])[0] + 1
    year_ticks = np.concatenate(([0], year_change_idx, [len(years)]))

    year_labels_pos = []
    year_labels = []
    for i in range(len(year_ticks)-1):
        start = year_ticks[i]
        end = year_ticks[i+1]
        # position year label centered under months in that year
        pos = (start + end - 1) / 2
        year_labels_pos.append(pos)
        year_labels.append(str(years[start]))

    ### Plot 1: total messages per month ###

    fig, ax = plt.subplots(figsize=(14, 7))
    total_per_month.plot(kind='bar', color='tab:blue', ax=ax)

    ax.set_title("Total Messages Per Month", fontsize=16, weight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Messages")

    # Main x-axis: show month numbers
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=0)

    # Create second x-axis below for year labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(year_labels_pos)
    ax2.set_xticklabels(year_labels, fontsize=12, weight='bold')
    ax2.tick_params(axis='x', length=0)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 35))
    ax2.set_xlabel('Year')

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/messages_per_month_total.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Plot 2: stacked messages per month by user ###

    fig, ax = plt.subplots(figsize=(14, 7))
    stacked.plot(kind='bar', stacked=True, colormap='tab20', legend='best', ax=ax)

    ax.set_title("Messages Per Month by User (Stacked)", fontsize=16, weight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Messages")

    # Main x-axis: month numbers
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=0)

    # Second x-axis for year labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(year_labels_pos)
    ax2.set_xticklabels(year_labels, fontsize=12, weight='bold')
    ax2.tick_params(axis='x', length=0)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 35))
    ax2.set_xlabel('Year')

    # Legend outside
    ax.legend(title="Sender Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/messages_per_month_stacked.png", dpi=300, bbox_inches='tight')
    plt.close()




def plot_messages_to_reactions_ratio(messages_df: pd.DataFrame):
    """
    Calculates and plots the ratio: number of messages / total reactions count per user.
    
    Assumes:
    - 'sender_name' column exists
    - 'reactions' column contains list of reactions or NaN/None
    
    Saves plot to outputs/messages_to_reactions_ratio.png
    """
    
    # Aggregate total messages and total reactions per user
    agg = messages_df.groupby('sender_name').agg(
        total_messages=('sender_name', 'size'),
        total_reactions=('reactions_len', 'sum')
    )

    # Avoid division by zero: if total_reactions = 0, set ratio to NaN or inf (decide)
    agg['ratio'] = agg['total_messages'] / agg['total_reactions'].replace(0, float('nan'))

    # Drop users with NaN ratios (no reactions at all)
    agg = agg.dropna(subset=['ratio']).sort_values('ratio', ascending=False)

    if agg.empty:
        print("No data with reactions to plot.")
        return

    # Plot
    plt.figure(figsize=(12, 7))
    ax = agg['ratio'].plot(kind='bar', color='tab:orange')

    plt.title("Number of Messages / Total Reactions per User", fontsize=14, weight='bold')
    plt.xlabel("User")
    plt.ylabel("Messages to Reactions Ratio")
    plt.xticks(rotation=45, ha='right')

    # Add ratio labels on bars
    for i, val in enumerate(agg['ratio']):
        ax.text(i, val + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/messages_to_reactions_ratio.png", dpi=300, bbox_inches='tight')
    plt.close()



def draw_call_timeline(df: pd.DataFrame, output_path="outputs/call_timeline.png"):
    # Filter rows with valid call_duration (non-null and > 0)
    df_calls = df[df['call_duration'].notna()].copy()
    df_sorted = df_calls.sort_values("datetime", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(df_sorted) * 0.8)))
    ax.set_ylim(-1, len(df_sorted))
    ax.set_xlim(0, 1)
    ax.axis("off")

    # Vertical timeline line
    ax.plot([0.5, 0.5], [0, len(df_sorted) - 1], color="gray", linewidth=2)

    for i, row in df_sorted.iterrows():
        y = i
        dt_str = row["datetime"].strftime("%Y-%m-%d %H:%M")
        duration_sec = row["call_duration"]
        sender = row["sender_name"]

        # Format duration nicely: H:M:S or M:S
        hrs, rem = divmod(duration_sec, 3600)
        mins, secs = divmod(rem, 60)
        if hrs > 0:
            duration_str = f"{int(hrs)}:{int(mins):02d}:{int(secs):02d}"
        else:
            duration_str = f"{int(mins)}:{int(secs):02d}"

        # Dot marker on timeline
        ax.plot(0.5, y, "o", color="tab:green", markersize=10)

        # Left side: datetime
        ax.text(0.48, y, dt_str, va="center", ha="right", fontsize=9)

        # Right side: call duration in bold
        ax.text(0.52, y + 0.1, duration_str, va="bottom", ha="left", fontsize=10, fontweight="bold")

        # Right side: sender in italic, below duration
        ax.text(0.52, y - 0.1, sender, va="top", ha="left", fontsize=9, fontstyle="italic", color="gray")

    plt.title("Call Timeline", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()




def plot_unsent_messages_per_user(messages_df, output_path="outputs/unsent_messages_per_user.png"):
    """
    Plots and saves a vertical bar chart of the count of unsent messages per user.
    
    Assumes 'sender_name' and 'is_unsent' columns exist.
    """
    if 'sender_name' not in messages_df.columns or 'is_unsent' not in messages_df.columns:
        raise ValueError("messages_df must contain 'sender_name' and 'is_unsent' columns.")

    # Filter unsent messages
    unsent_counts = messages_df[messages_df['is_unsent']].groupby('sender_name').size()

    if unsent_counts.empty:
        print("No unsent messages found.")
        return

    # Sort by count descending
    unsent_counts = unsent_counts.sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    ax = unsent_counts.plot(kind='bar', color='tab:red')

    plt.title("Number of Unsent Messages per User", fontsize=14, weight='bold')
    plt.xlabel("User")
    plt.ylabel("Unsent Messages Count")
    plt.xticks(rotation=45, ha='right')

    # Add counts on top of bars
    for i, count in enumerate(unsent_counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_messages_per_sender(messages_df, output_path="outputs/messages_per_sender.png"):
    # Count messages per sender
    sender_counts = messages_df['sender_name'].value_counts()

    # Total number of messages
    total_messages = sender_counts.sum()

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sender_counts.index, sender_counts.values, color='skyblue')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}', ha='center', va='bottom', fontsize=9)

    # Title and labels
    plt.title(f'Number of Messages per Sender\nTotal Messages: {total_messages}', fontsize=14)
    plt.xlabel('Sender', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Layout adjustment
    plt.tight_layout()

    # Save plot to file
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to: {output_path}")
