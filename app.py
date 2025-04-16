import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import csv

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        # Check if file exists
        if not os.path.exists('books.csv'):
            st.error("books.csv file not found. Please make sure the file is in the same directory as the application.")
            return None
        
        # First, read the CSV file to get the correct number of columns
        with open('books.csv', 'r', encoding='latin1') as f:
            reader = csv.reader(f)
            header = next(reader)
            num_columns = len(header)
        
        # Load the books dataset with error handling
        books_df = pd.read_csv('books.csv', 
                              encoding='latin1',
                              on_bad_lines='skip',  # Skip problematic lines
                              quoting=csv.QUOTE_MINIMAL,
                              escapechar='\\')
        
        # Debug information
        st.write("Dataset loaded successfully!")
        st.write(f"Number of rows: {len(books_df)}")
        st.write("Columns in dataset:", books_df.columns.tolist())
        
        # Check if required columns exist
        required_columns = ['title', 'authors', 'description']
        missing_columns = [col for col in required_columns if col not in books_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in the dataset: {', '.join(missing_columns)}")
            return None
        
        # Preprocess the data
        books_df['combined_features'] = books_df['title'].fillna('') + ' ' + \
                                      books_df['authors'].fillna('') + ' ' + \
                                      books_df['description'].fillna('')
        
        # Remove rows with empty combined_features
        books_df = books_df[books_df['combined_features'].str.strip() != '']
        
        st.write(f"Number of valid rows after preprocessing: {len(books_df)}")
        return books_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def get_recommendations(user_input, books_df, vectorizer, tfidf_matrix):
    try:
        # Preprocess user input
        processed_input = preprocess_text(user_input)
        
        # Transform user input
        input_vector = vectorizer.transform([processed_input])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
        
        # Get top 5 recommendations
        top_indices = similarity_scores.argsort()[-5:][::-1]
        
        recommendations = []
        for idx in top_indices:
            book = books_df.iloc[idx]
            recommendations.append({
                'title': book['title'],
                'authors': book['authors'],
                'description': book['description'],
                'similarity_score': similarity_scores[idx]
            })
        
        return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

def main():
    st.title("Book Recommender System")
    st.write("Enter your preferences or describe what kind of book you're looking for:")
    
    # Load data
    books_df = load_data()
    
    if books_df is None:
        st.stop()
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'].apply(preprocess_text))
        
        # User input
        user_input = st.text_area("Describe your book preferences:", height=100)
        
        if st.button("Get Recommendations"):
            if user_input:
                recommendations = get_recommendations(user_input, books_df, vectorizer, tfidf_matrix)
                
                if recommendations:
                    st.subheader("Recommended Books:")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"### {i}. {rec['title']}")
                        st.write(f"**Authors:** {rec['authors']}")
                        st.write(f"**Description:** {rec['description']}")
                        st.write("---")
                else:
                    st.warning("No recommendations found. Try being more specific in your preferences.")
            else:
                st.warning("Please enter your book preferences to get recommendations.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 