

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


def load_messages_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=";",
        decimal=".",
        quoting=csv.QUOTE_ALL
    )

    # Convert columns to proper types
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Convert 'date' column to datetime.date objects
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Convert 'time' column to datetime.time objects with seconds precision
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time

    # Explicitly cast string columns to object/string type
    for col in ["sender_name", "content", "reactions", "media"]:
        df[col] = df[col].astype(str)

    # Cast bool column
    df["is_unsent"] = df["is_unsent"].astype(bool)

    # Cast integer columns
    int_cols = ["timestamp_ms", "content_len", 'Photos', 'Videos', 'GIFs', 'Audio', 'FB posts', 'Files', 'Stickers']
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype("int64")

    # Cast float column
    df["call_duration"] = pd.to_numeric(df["call_duration"], errors='coerce').astype("float64")

    # Convert 'nan' strings to real NaN
    df['content'] = df['content'].replace("nan", np.nan)
    df['media'] = df['media'].replace("nan", np.nan)
    df['reactions'] = df['reactions'].replace("nan", np.nan)

    df["content"] = df["content"].str.strip()

    return df