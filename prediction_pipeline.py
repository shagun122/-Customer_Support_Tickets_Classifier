import joblib
import os
import pandas as pd
import numpy as np

# Import from our custom modules
from text_processor import preprocess_text_pipeline
from feature_engineer import (
    get_ticket_length,
    get_sentiment_compound_score,
    combine_features # TF-IDF transformation will be handled by the loaded vectorizer
)
from entity_extractor import extract_all_entities
from data_utils import load_data, clean_data # To load product list

# Define paths (should match model_trainer.py and actual saved locations)
MODEL_DIR = 'saved_models'
ISSUE_TYPE_MODEL_PATH = os.path.join(MODEL_DIR, 'issue_type_classifier.joblib')
URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, 'urgency_level_classifier.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
DATA_FILE_PATH = 'ai_dev_assignment_tickets_complex_1000.xls' # For product list

# --- Load models and vectorizer once when the module is loaded ---
try:
    issue_type_model = joblib.load(ISSUE_TYPE_MODEL_PATH)
    urgency_level_model = joblib.load(URGENCY_MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("Models and TF-IDF vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models/vectorizer: {e}. Ensure 'model_trainer.py' has been run successfully.")
    # Set to None so the app can potentially still run with a warning or limited functionality
    issue_type_model = None
    urgency_level_model = None
    tfidf_vectorizer = None
except Exception as e:
    print(f"An unexpected error occurred loading models/vectorizer: {e}")
    issue_type_model = None
    urgency_level_model = None
    tfidf_vectorizer = None
    
# --- Load product list for entity extraction ---
PRODUCT_LIST_FOR_EXTRACTION = []
try:
    print("Loading product list for entity extraction...")
    df_full = load_data(DATA_FILE_PATH)
    if not df_full.empty:
        # Intentionally not using clean_data's product fill for the source of truth product list
        # We want actual product names, not 'Unknown'
        unique_products = df_full['product'].dropna().unique()
        PRODUCT_LIST_FOR_EXTRACTION = [str(p) for p in unique_products if str(p).lower() != 'unknown']
        print(f"Product list loaded. Found {len(PRODUCT_LIST_FOR_EXTRACTION)} unique products.")
        if not PRODUCT_LIST_FOR_EXTRACTION:
             print("Warning: Product list for extraction is empty after loading. Entity extraction for products might be ineffective.")
    else:
        print("Warning: Could not load data to create product list for entity extraction. Using default or empty list.")
except Exception as e:
    print(f"Error loading product list: {e}. Entity extraction for products might be affected.")


def analyze_ticket(raw_ticket_text: str) -> dict:
    """
    Analyzes raw ticket text to predict issue type, urgency, and extract entities.
    """
    if not all([issue_type_model, urgency_level_model, tfidf_vectorizer]):
        return {
            "error": "Models or vectorizer not loaded. Please run model_trainer.py first.",
            "predicted_issue_type": "N/A",
            "predicted_urgency_level": "N/A",
            "extracted_entities": {}
        }

    if not isinstance(raw_ticket_text, str) or not raw_ticket_text.strip():
        return {
            "predicted_issue_type": "N/A (No input text)",
            "predicted_urgency_level": "N/A (No input text)",
            "extracted_entities": {"products": [], "dates": [], "complaint_keywords": []}
        }

    # 1. Preprocess Text
    processed_text = preprocess_text_pipeline(raw_ticket_text)

    # 2. Feature Engineering
    # TF-IDF
    # Input to transform must be an iterable (e.g., a list containing the single processed text)
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    
    # Additional features
    ticket_len = get_ticket_length(processed_text)
    # Use original 'raw_ticket_text' for VADER sentiment
    sentiment = get_sentiment_compound_score(raw_ticket_text) 
    
    # Create a DataFrame for additional features to match the structure used in combine_features
    # The order of columns must match how it was during training
    additional_features_df = pd.DataFrame([[ticket_len, sentiment]], columns=['ticket_length', 'sentiment_compound'])
    
    # Combine features
    # The combine_features function expects a TF-IDF matrix and a DataFrame
    current_features = combine_features(tfidf_features, additional_features_df)

    # 3. Predictions
    predicted_issue_type = issue_type_model.predict(current_features)[0]
    predicted_urgency_level = urgency_level_model.predict(current_features)[0]
    
    # 4. Entity Extraction
    # Use the dynamically loaded product list
    extracted_entities = extract_all_entities(raw_ticket_text, product_list=PRODUCT_LIST_FOR_EXTRACTION)
    
    return {
        "predicted_issue_type": str(predicted_issue_type), # Ensure string output
        "predicted_urgency_level": str(predicted_urgency_level), # Ensure string output
        "extracted_entities": extracted_entities
    }

if __name__ == '__main__':
    # This part will only work if models have been trained and saved by model_trainer.py
    if not all([issue_type_model, urgency_level_model, tfidf_vectorizer]):
        print("\\nModels/vectorizer not loaded. Run model_trainer.py first to test this script.")
    else:
        print(f"Using product list for extraction: {PRODUCT_LIST_FOR_EXTRACTION[:10]}...") # Show a sample

        sample_ticket_1 = "My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404 on 2023-03-15. This is a disaster."
        analysis_1 = analyze_ticket(sample_ticket_1)
        print(f"\\nAnalysis for ticket 1 ('{sample_ticket_1[:50]}...'):")
        print(analysis_1)

        sample_ticket_2 = "The AlphaWidget is not working as expected since 1st April 2024. It's showing a strange message and I am very unhappy. Also, my GammaConnector seems faulty."
        analysis_2 = analyze_ticket(sample_ticket_2)
        print(f"\\nAnalysis for ticket 2 ('{sample_ticket_2[:50]}...'):")
        print(analysis_2)

        sample_ticket_3 = "Everything is great with BetaService! Just a quick question about billing for May 2023."
        analysis_3 = analyze_ticket(sample_ticket_3)
        print(f"\\nAnalysis for ticket 3 ('{sample_ticket_3[:50]}...'):")
        print(analysis_3)
        
        sample_ticket_4 = "This is a test ticket with no specific product mentioned, but it seems urgent and is definitely a complaint."
        analysis_4 = analyze_ticket(sample_ticket_4)
        print(f"\\nAnalysis for ticket 4 ('{sample_ticket_4[:50]}...'):")
        print(analysis_4)

        analysis_empty = analyze_ticket("")
        print(f"\\nAnalysis for empty ticket:")
        print(analysis_empty)
