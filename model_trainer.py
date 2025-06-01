import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Import from our custom modules
from data_utils import load_data, clean_data
from text_processor import preprocess_text_pipeline
from feature_engineer import (
    create_text_features_tfidf, 
    get_ticket_length, 
    get_sentiment_compound_score,
    combine_features
)

# Define paths for saving models and vectorizer
MODEL_DIR = 'saved_models'
ISSUE_TYPE_MODEL_PATH = os.path.join(MODEL_DIR, 'issue_type_classifier.joblib')
URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, 'urgency_level_classifier.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
DATA_FILE_PATH = 'ai_dev_assignment_tickets_complex_1000.xls'

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name="Classifier"):
    """
    Trains a Logistic Regression model and evaluates it.
    """
    print(f"\\nTraining {model_name}...")
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000) # Added max_iter
    model.fit(X_train, y_train)
    
    print(f"Evaluating {model_name}...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"Training Accuracy for {model_name}: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Test Accuracy for {model_name}: {accuracy_score(y_test, y_pred_test):.4f}")
    
    print(f"Classification Report for {model_name} (Test Set):")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    
    return model

def main():
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load and Clean Data
    print("Loading and cleaning data...")
    df = load_data(DATA_FILE_PATH)
    if df.empty:
        print("Failed to load data. Exiting.")
        return
    df = clean_data(df)
    if df.empty:
        print("Data is empty after cleaning. Exiting.")
        return

    # 2. Preprocess Text
    print("\\nPreprocessing text data...")
    df['processed_text'] = df['ticket_text'].apply(preprocess_text_pipeline)

    # 3. Feature Engineering
    print("\\nEngineering features...")
    # TF-IDF (Fit on the entire 'processed_text' for now, then split. Or fit on train set only)
    # For simplicity here, fit on all, then split. In a strict setup, fit only on train.
    # However, for saving the vectorizer, fitting on all available text data before splitting can be common.
    corpus = df['processed_text']
    tfidf_matrix, tfidf_vectorizer = create_text_features_tfidf(corpus, fit_vectorizer=True)
    
    # Additional features
    df['ticket_length'] = df['processed_text'].apply(get_ticket_length)
    # Use original 'ticket_text' for VADER sentiment as it performs better on less processed text
    df['sentiment_compound'] = df['ticket_text'].apply(get_sentiment_compound_score)
    
    additional_features_df = df[['ticket_length', 'sentiment_compound']]
    
    # Combine features
    all_features = combine_features(tfidf_matrix, additional_features_df)
    
    # Define targets
    y_issue_type = df['issue_type']
    y_urgency_level = df['urgency_level']

    # --- Train Issue Type Classifier ---
    print("\\n--- Preparing for Issue Type Classifier ---")
    X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(
        all_features, y_issue_type, test_size=0.2, random_state=42, stratify=y_issue_type
    )
    issue_type_model = train_and_evaluate_model(
        X_train_issue, y_train_issue, X_test_issue, y_test_issue, "Issue Type Classifier"
    )
    joblib.dump(issue_type_model, ISSUE_TYPE_MODEL_PATH)
    print(f"Issue Type Classifier saved to {ISSUE_TYPE_MODEL_PATH}")

    # --- Train Urgency Level Classifier ---
    print("\\n--- Preparing for Urgency Level Classifier ---")
    # Ensure labels are strings for stratification if they are not already
    y_urgency_level_str = y_urgency_level.astype(str)
    X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(
        all_features, y_urgency_level, test_size=0.2, random_state=42, stratify=y_urgency_level_str
    )
    urgency_level_model = train_and_evaluate_model(
        X_train_urgency, y_train_urgency, X_test_urgency, y_test_urgency, "Urgency Level Classifier"
    )
    joblib.dump(urgency_level_model, URGENCY_MODEL_PATH)
    print(f"Urgency Level Classifier saved to {URGENCY_MODEL_PATH}")

    # Save the TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    print(f"TF-IDF Vectorizer saved to {VECTORIZER_PATH}")

    print("\\nModel training and saving complete.")

if __name__ == '__main__':
    main()
