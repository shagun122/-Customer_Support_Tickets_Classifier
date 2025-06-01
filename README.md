# Customer Support Ticket Analysis Pipeline

## Project Overview

This project develops a machine learning pipeline to classify customer support tickets by their issue type and urgency level, and to extract key entities (product names, dates, complaint keywords) from the ticket text. The goal is to automate parts of the customer support workflow, enabling faster routing and understanding of customer issues.

The pipeline uses traditional NLP techniques and classical machine learning models. An interactive web interface is provided using Gradio.

## Core Components

The project is structured into several Python scripts:

-   `data_utils.py`: Handles loading and initial cleaning of the dataset.
-   `text_processor.py`: Contains functions for text preprocessing (normalization, tokenization, stopword removal, lemmatization).
-   `feature_engineer.py`: Implements feature creation, including TF-IDF, ticket length, and sentiment scores.
-   `model_trainer.py`: Script to train two separate classifiers (for issue type and urgency level) and save the trained models and TF-IDF vectorizer.
-   `entity_extractor.py`: Provides functions to extract product names, dates, and complaint keywords from text.
-   `prediction_pipeline.py`: Contains the core `analyze_ticket` function that orchestrates the full analysis pipeline for a given raw ticket text, loading pre-trained models.
-   `app.py`: The Gradio web application for interactive ticket analysis.

## Setup Instructions

### 1. Prerequisites

-   Python 3.8 or higher.
-   `pip` for installing Python packages.

### 2. Clone the Repository (if applicable)

```bash
# If this were a git repository:
# git clone <repository_url>
# cd <repository_directory>
```
For now, ensure all provided `.py` files and the `ai_dev_assignment_tickets_complex_1000.xls` dataset are in the same directory.

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\\Scripts\\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

A `requirements.txt` file is provided (or will be generated). Install the necessary packages using:

```bash
pip install -r requirements.txt
```

### 5. Download NLTK Resources

The scripts (`text_processor.py`, `feature_engineer.py`) will attempt to download necessary NLTK resources (stopwords, punkt, wordnet, vader_lexicon) on their first run if they are not found. You can also pre-download them by running Python and executing:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Added based on runtime error
nltk.download('wordnet')
nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger') # Optional, for more advanced lemmatization
```

## How to Run

### 1. Train the Models

Before you can use the prediction pipeline or the Gradio app, you need to train the models and save the TF-IDF vectorizer. Run the `model_trainer.py` script:

```bash
python model_trainer.py
```

This will:
-   Load and preprocess the data from `ai_dev_assignment_tickets_complex_1000.xls`.
-   Engineer features.
-   Train a Logistic Regression model for `issue_type` classification.
-   Train a Logistic Regression model for `urgency_level` classification.
-   Save the trained models and the TF-IDF vectorizer into a `saved_models/` directory (it will be created if it doesn't exist).
-   Print evaluation metrics (accuracy, classification report) for both models to the console.

### 2. Run the Gradio Web Application

Once the models are trained and saved, you can launch the interactive Gradio app:

```bash
python app.py
```

This will start a local web server, and you'll see a URL in your console (usually `http://127.0.0.1:7860` or similar). Open this URL in your web browser to use the application.

You can input raw ticket text into the provided text area and see the predicted issue type, urgency level, and extracted entities.

### 3. (Optional) Test the Prediction Pipeline Directly

You can also test the core prediction logic by running `prediction_pipeline.py` directly if models are already trained:
```bash
python prediction_pipeline.py
```
This will run predictions on a few sample texts defined within the script.

## Design Choices

-   **Modularity**: The code is split into multiple Python files for better organization and reusability.
-   **Text Preprocessing**: Standard NLP techniques are used: lowercasing, removal of special characters, tokenization, stopword removal, and lemmatization (NLTK WordNetLemmatizer).
-   **Feature Engineering**:
    -   **TF-IDF**: Chosen for its effectiveness in converting text into meaningful numerical features for classification. `ngram_range=(1,2)` is used to capture bi-grams along with unigrams. `max_features=5000` helps manage dimensionality.
    -   **Ticket Length**: A simple but potentially useful feature; longer tickets might correlate with more complex issues.
    -   **Sentiment Score**: VADER (Valence Aware Dictionary and sEntiment Reasoner) is used for sentiment analysis, as it's well-suited for text with emotional content, like customer feedback. The compound score is used as a single sentiment feature. Sentiment is calculated on the original ticket text for better VADER performance.
-   **Machine Learning Models**:
    -   **Logistic Regression**: Chosen as a baseline classical ML model. It's relatively simple, interpretable, and often performs well on text classification tasks. `class_weight='balanced'` is used to handle potential class imbalances in the target variables.
    -   Two separate models are trained for `issue_type` and `urgency_level` as per the assignment's multi-task learning requirement (though implemented as separate tasks here).
-   **Entity Extraction**:
    -   **Product Names**: Extracted by matching against a list derived from the `product` column in the dataset. This ensures relevance to known products.
    -   **Dates**: The `datefinder` library is used for robust date extraction.
    -   **Complaint Keywords**: A predefined list of keywords is used to identify potential complaints.
-   **Model & Vectorizer Persistence**: `joblib` is used to save and load the trained scikit-learn models and TF-IDF vectorizer.
-   **Gradio Interface**: Provides an easy-to-use web UI for interacting with the prediction pipeline.

## Model Evaluation (Expected)

When `model_trainer.py` is run, it will output:
-   **Training Accuracy**: Accuracy of the models on the data they were trained on.
-   **Test Accuracy**: Accuracy of the models on a held-out test set (20% of the data). This is a more realistic measure of performance on unseen data.
-   **Classification Report (Test Set)**: For each class, this report includes:
    -   **Precision**: The ability of the classifier not to label as positive a sample that is negative.
    -   **Recall (Sensitivity)**: The ability of the classifier to find all the positive samples.
    -   **F1-score**: A weighted average of precision and recall.
    -   **Support**: The number of actual occurrences of the class in the specified dataset.

The specific values will depend on the dataset and the inherent difficulty of the classification tasks.

## Limitations

-   **Dataset Size**: The provided dataset has 1000 samples. Performance might improve with a larger and more diverse dataset.
-   **Fixed Product List**: Product name extraction relies on a list derived from the `product` column of the training data. It won't identify new or misspelled product names not present in this list.
-   **Simple Entity Extraction Rules**: Date and complaint keyword extraction are rule-based (datefinder library and a fixed keyword list). They might not capture all nuances or less common expressions.
-   **Sentiment Analysis Context**: VADER is generally good, but sentiment can be highly contextual and might be misinterpreted in complex sentences.
-   **Model Simplicity**: Logistic Regression is used. More complex models (e.g., SVM with different kernels, Random Forest, Gradient Boosting, or even simple neural networks) might offer better performance but require more tuning.
-   **No Hyperparameter Tuning**: The current implementation uses default or fixed hyperparameters for TF-IDF and Logistic Regression. Systematic hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV) could improve model performance.
-   **"Multi-Task Learning" Implementation**: The assignment mentions multi-task learning. This implementation trains two separate models. True multi-task learning (where a single model learns multiple tasks simultaneously, potentially sharing layers or representations) is more complex and not implemented here.
-   **Error Propagation**: Errors or inaccuracies in earlier stages of the pipeline (e.g., preprocessing, feature engineering) can affect later stages (model prediction).
-   **Scalability of `combine_features`**: The current `combine_features` function converts the sparse TF-IDF matrix to a dense array. For very large datasets and feature sets, this could be memory-intensive. Using sparse matrices throughout or models that natively handle sparse input would be more scalable.

## Future Enhancements (Bonus Ideas)

-   Implement more sophisticated ML models (e.g., SVM, Random Forest, Gradient Boosting).
-   Perform hyperparameter tuning for models and TF-IDF.
-   Use `spaCy` for more advanced NLP tasks like named entity recognition (NER) for products, dates, and potentially other entities, which might be more robust than regex/list-based methods.
-   Implement true multi-task learning.
-   Add visualizations (ticket distributions, feature importances, confusion matrices) to the Gradio app or a separate report.
-   Allow batch processing of multiple tickets via the Gradio app.
-   Improve error handling and logging throughout the pipeline.
