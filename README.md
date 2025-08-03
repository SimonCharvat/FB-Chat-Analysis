# 📊 FB Chat Analysis

A Python project that analyzes Facebook chat data to generate insightful statistics and visualizations. Includes a simple Flask web app for browsing and searching chat messages. Ideal for group chats.

---

## 🚀 Features

- Parses Facebook chat data downloaded in JSON format
- Merges multiple JSON message files into one CSV
- Generates static plots with chat statistics
- Interactive Flask app to browse and search messages

---

## 📁 Project Structure

```
FB-Chat-Analysis/
├── 1_parse_raw_data.py       # Loads and processes raw JSON chat data
├── 2_static_analysis.py      # Generates plots and statistics
├── messages.csv              # Preprocessed chat data (generated)
├── requirements.txt          # Required libraries
├── utils.py                  # Utility functions
├── visualize.py              # Plotting helpers
├── flaskr/                   # Flask web app
│   ├── __init__.py           # Starts the Flask app
│   └── templates/
│       └── index.html        # Web UI
├── messages/                 # Folder for downloaded JSON files
│   ├── message_1.json
│   ├── ...
└── .gitignore
```

---

## 📥 Getting the Data

1. Go to [Facebook Download Your Information](https://accountscenter.facebook.com/info_and_permissions/dyi)
2. Choose **Messages** as the data type
3. Select **JSON** format
4. Download and extract the archive
5. Copy all message JSON files into the `/messages/` directory

---

## ⚙️ How to Run

### 1. Preprocess the Chat Data

Run the script to combine and clean all message files:

```bash
python 1_parse_raw_data.py
```

This will generate `messages.csv`.

---

### 2. Generate Static Plots

```bash
python 2_static_analysis.py
```

This script uses the CSV file to generate visual insights (e.g. message frequency per sender, chat activity over time, etc.).

---

### 3. Run the Flask App

To browse and search messages via a local web interface:

```bash
cd flaskr
python -m flask run
```

Then visit `http://127.0.0.1:5000/` in your browser.

---

## 🧩 Requirements

- Python 3.10+
- Flask
- Pandas
- Numpy
- Matplotlib
- Seaborn

Install requirements via:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Only private messages (not comments or posts) are supported.
- Make sure the `messages/` folder contains all relevant JSON files before running the scripts.
