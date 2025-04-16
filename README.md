# Book Recommender System

A content-based book recommendation system that suggests books based on user preferences using natural language processing and machine learning.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the books dataset from Kaggle:
   - Go to https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
   - Download the `books.csv` file
   - Place it in the same directory as the application

## Running the Application

To run the application, use the following command:
```bash
streamlit run app.py
```

## How to Use

1. Open the application in your web browser
2. Enter your book preferences or describe what kind of book you're looking for
3. Click "Get Recommendations" to see suggested books
4. The system will show you 5 books that match your preferences

## Features

- Content-based recommendation using TF-IDF and cosine similarity
- Natural language processing for text analysis
- User-friendly interface using Streamlit
- Shows book title, authors, and description for each recommendation 
