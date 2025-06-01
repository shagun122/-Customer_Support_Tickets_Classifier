# Customer Support Ticket Analyzer

A machine learning pipeline to classify customer support tickets by issue type and urgency, and extract key entities (product names, dates, complaint keywords). Features an interactive web interface built with Gradio.

## Features

*   **Issue Type Classification:** Predicts the category of the support ticket (e.g., Bug, Feature Request).
*   **Urgency Level Prediction:** Assesses the urgency of the ticket (e.g., High, Medium, Low).
*   **Entity Extraction:** Identifies and extracts:
    *   Product names mentioned in the ticket.
    *   Dates relevant to the issue.
    *   Keywords indicating a complaint or negative sentiment.
*   **Interactive Web UI:** A Gradio interface for easy input and visualization of analysis results.

## Core Components & Tech Stack

*   **Python 3.8+**
*   **Scikit-learn:** For machine learning (Logistic Regression for classification, TF-IDF for text vectorization).
*   **NLTK:** For text preprocessing (tokenization, lemmatization, stopwords) and VADER sentiment analysis.
*   **Pandas:** For data manipulation.
*   **Gradio:** For building the interactive web interface.
*   **Joblib:** For saving and loading trained models.
*   **Key Scripts:**
    *   `app.py`: Gradio web application.
    *   `prediction_pipeline.py`: Core logic for analyzing new tickets.
    *   `model_trainer.py`: Script to train and save ML models.
    *   `text_processor.py`: Text preprocessing functions.
    *   `feature_engineer.py`: Feature creation (TF-IDF, sentiment, length).
    *   `entity_extractor.py`: Entity extraction logic.
    *   `data_utils.py`: Data loading and cleaning utilities.

## Setup and Installation

1.  **Clone the Repository (if applicable) or Download Files:**
    Ensure all project files (`.py`, `.xls`, `requirements.txt`) are in the same directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(The `requirements.txt` file should list all necessary packages like pandas, scikit-learn, nltk, gradio, joblib, openpyxl, datefinder).*

4.  **Download NLTK Resources:**
    The scripts will attempt to download necessary NLTK resources (`stopwords`, `punkt`, `wordnet`, `vader_lexicon`) on first run. You can also pre-download them:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    # nltk.download('punkt_tab') # May be needed by word_tokenize
    ```

## How to Run

1.  **Train the Models:**
    Before using the app, you must train the models. Run:
    ```bash
    python model_trainer.py
    ```
    This will process `ai_dev_assignment_tickets_complex_1000.xls`, train the classifiers, and save them (along with the TF-IDF vectorizer) into the `saved_models/` directory.

2.  **Run the Gradio Web Application:**
    Once models are trained and saved:
    ```bash
    python app.py
    ```
    Open the URL provided in your console (usually `http://127.0.0.1:7860`) in your web browser.

## Project Structure

*   `ai_dev_assignment_tickets_complex_1000.xls`: The dataset used for training.
*   `*.py`: Python scripts for different modules of the pipeline.
*   `saved_models/`: Directory where trained models and the vectorizer are stored.
*   `requirements.txt`: Lists project dependencies.
*   `README.md`: This file.

