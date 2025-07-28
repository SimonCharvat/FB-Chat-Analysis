

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import sys
import os
import re
from datetime import timedelta

# Add path to allow import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils



def assign_colors(df: pd.DataFrame):
    palette = [
        "#b4ddff",  # blue
        "#afe6b1",  # green
        "#ffd54f",  # yellow
        "#f5beff",  # purple
        "#c0fff9",  # teal
        "#ffcaba",  # orange
        "#ffc6c6",  # red
    ]
    unique_senders = df["sender_name"].dropna().unique()
    color_map = {}
    for i, sender in enumerate(sorted(unique_senders)):
        color_map[sender] = palette[i % len(palette)]
    return color_map


app = Flask(__name__)


# Load dataframe here
df = utils.load_messages_csv("messages.csv")

# Filter out unsent messages
df = df[df["is_unsent"] != True]

# Fill content with media where content is NA
df['content'] = df['content'].fillna(df['media'])

# Fill remaining NAs with empty string
df['content'] = df['content'].fillna("")
df['reactions'] = df['reactions'].fillna("")

# Make sure df is sorted by datetime ascending
df = df.sort_values("datetime").reset_index(drop=True)

color_map = assign_colors(df)

PAGE_SIZE = 30


@app.route("/")
def index():
    # Pagination
    page = request.args.get("page", default=1, type=int)
    q = request.args.get("q", default="", type=str).strip()
    jump_to = request.args.get("jump_to", default="", type=str)

    jump_index = None

    # Handle jump before filtering
    if jump_to:
        try:
            jump_dt = pd.to_datetime(jump_to)
            # Always jump within full dataset (to get context)
            idx = df[df["datetime"] >= jump_dt].index.min()
            if pd.notna(idx):
                page = (idx // PAGE_SIZE) + 1
                jump_index = idx
        except Exception:
            pass

    filtered_df = df

    # If query is provided, filter by sender_name or content substring (case-insensitive)
    if q:
        mask = filtered_df["sender_name"].str.contains(q, case=False, na=False) | \
            filtered_df["content"].str.contains(q, case=False, na=False)
        filtered_df = filtered_df[mask]


    total = len(filtered_df)
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = filtered_df.iloc[start:end]

    # After you get page_df:
    times = page_df["datetime"].reset_index(drop=True)
    gaps = [False]  # No gap before first message on page

    for i in range(1, len(times)):
        diff = times[i] - times[i-1]
        if diff > timedelta(minutes=30):
            gaps.append(True)
        else:
            gaps.append(False)

    # Pass gaps list to template
    return render_template(
        "index.html",
        page=page,
        q=q,
        jump_to=jump_to,
        page_df=page_df.reset_index(),  # Reset index for easy matching
        start=start,
        end=min(end, total),
        total=total,
        jump_index=jump_index,
        gaps=gaps,
        color_map=color_map
    )

if __name__ == "__main__":
    app.run(debug=True)
