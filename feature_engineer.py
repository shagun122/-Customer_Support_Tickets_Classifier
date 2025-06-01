import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
try:
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Could not initialize SentimentIntensityAnalyzer: {e}. Sentiment scores will be default.")
    analyzer = None

def get_ticket_length(text: str) -> int:
    """
    Calculates the length of the text (number of words after basic split).
    """
    if not isinstance(text, str):
        return 0
    return len(text.split())

def get_sentiment_scores(text: str) -> dict:
    """
    Calculates sentiment scores (pos, neg, neu, compound) for the text using VADER.
    Returns a dictionary of scores.
    """
    if analyzer is None or not isinstance(text, str) or not text.strip():
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    
    # VADER expects raw text, not heavily preprocessed text for best results
    # So, this function should ideally be called on text that is not yet stripped of all punctuation
    # or stopwords that VADER might use.
    # For simplicity in this pipeline, we might call it on the original or lightly cleaned text.
    vs = analyzer.polarity_scores(text)
    return vs

def get_sentiment_compound_score(text: str) -> float:
    """
    Extracts only the compound sentiment score.
    """
    if not isinstance(text, str): # Ensure text is a string
        return 0.0 
    return get_sentiment_scores(text).get('compound', 0.0)


def create_text_features_tfidf(corpus: pd.Series, vectorizer: TfidfVectorizer = None, fit_vectorizer: bool = False):
    """
    Creates TF-IDF features from a corpus of text.
    If fit_vectorizer is True, it fits a new TfidfVectorizer.
    Otherwise, it uses the provided (pre-fitted) vectorizer.
    Returns the TF-IDF matrix and the vectorizer.
    """
    if vectorizer is None or fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2)) # Example parameters
        tfidf_matrix = vectorizer.fit_transform(corpus)
    else:
        tfidf_matrix = vectorizer.transform(corpus)
    return tfidf_matrix, vectorizer

def combine_features(tfidf_matrix, additional_features_df: pd.DataFrame) -> np.ndarray:
    """
    Combines TF-IDF matrix (sparse) with other dense features.
    Converts TF-IDF to dense array for simplicity here, though for very large
    datasets, keeping it sparse and using sparse-compatible models is better.
    """
    # Convert sparse TF-IDF matrix to dense array
    dense_tfidf_array = tfidf_matrix.toarray()
    
    # Concatenate with other features
    combined = np.hstack((dense_tfidf_array, additional_features_df.values))
    return combined


if __name__ == '__main__':
    # Example Usage
    from text_processor import preprocess_text_pipeline # Assuming text_processor.py is in the same directory

    sample_data = {
        'ticket_id': [1, 2, 3],
        'ticket_text': [
            "My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404.",
            "The AlphaWidget is not working as expected. It's showing a strange message and I am very unhappy.",
            "Everything is great with BetaService! Just a quick question about billing."
        ],
        'processed_text': [
            preprocess_text_pipeline("My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404."),
            preprocess_text_pipeline("The AlphaWidget is not working as expected. It's showing a strange message and I am very unhappy."),
            preprocess_text_pipeline("Everything is great with BetaService! Just a quick question about billing.")
        ]
    }
    df = pd.DataFrame(sample_data)

    # 1. TF-IDF
    # Fit on the 'processed_text'
    print("Fitting TF-IDF...")
    tfidf_matrix, fitted_vectorizer = create_text_features_tfidf(df['processed_text'], fit_vectorizer=True)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    # print(f"Feature names (sample): {fitted_vectorizer.get_feature_names_out()[:20]}") # View some feature names

    # 2. Ticket Length
    # Calculate on 'processed_text' as it's cleaner
    df['ticket_length'] = df['processed_text'].apply(get_ticket_length)
    print("\nTicket Lengths:")
    print(df[['ticket_text', 'ticket_length']])

    # 3. Sentiment Score
    # Calculate on original 'ticket_text' as VADER performs better on less processed text
    df['sentiment_compound'] = df['ticket_text'].apply(get_sentiment_compound_score)
    print("\nSentiment Scores (Compound):")
    print(df[['ticket_text', 'sentiment_compound']])
    
    # Example of getting all sentiment scores
    # df['sentiment_all'] = df['ticket_text'].apply(get_sentiment_scores)
    # print(df[['ticket_text', 'sentiment_all']])


    # 4. Combine features
    additional_features = df[['ticket_length', 'sentiment_compound']]
    
    print("\nCombining features...")
    all_features_combined = combine_features(tfidf_matrix, additional_features)
    print(f"Shape of combined features: {all_features_combined.shape}")
    # print("\nSample of combined features (first row):")
    # print(all_features_combined[0, :10]) # Print first 10 features of the first sample
    # print(all_features_combined[0, -2:]) # Print last 2 features (length, sentiment) of the first sample
