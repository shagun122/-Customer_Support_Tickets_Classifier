import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize('tests')
except LookupError:
    nltk.download('wordnet')
try:
    # Check for punkt_tab, which is sometimes needed by word_tokenize indirectly
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# WordNetLemmatizer can benefit from POS tags for more accuracy, 
# but for simplicity, we'll do basic lemmatization.
# If POS tagging is desired:
# try:
#     nltk.pos_tag(['test'])
# except LookupError:
#     nltk.download('averaged_perceptron_tagger')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_text(text: str) -> str:
    """
    Lowercase text and remove special characters (keeping alphanumeric and spaces).
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text) # Keep letters, numbers, and spaces
    text = re.sub(r'\\s+', ' ', text).strip() # Replace multiple spaces with single and strip
    return text

def tokenize_text(text: str) -> list:
    """
    Tokenize text into words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens: list) -> list:
    """
    Remove stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatize a list of tokens.
    """
    # Example of more accurate lemmatization with POS tagging (optional enhancement)
    # def get_wordnet_pos(word):
    #     """Map POS tag to first character lemmatize() accepts"""
    #     tag = nltk.pos_tag([word])[0][1][0].upper()
    #     tag_dict = {"J": nltk.corpus.wordnet.ADJ,
    #                 "N": nltk.corpus.wordnet.NOUN,
    #                 "V": nltk.corpus.wordnet.VERB,
    #                 "R": nltk.corpus.wordnet.ADV}
    #     return tag_dict.get(tag, nltk.corpus.wordnet.NOUN) # Default to noun
    # return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    # Simpler lemmatization (defaults to noun if POS not specified)
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text_pipeline(text: str) -> str:
    """
    Complete text preprocessing pipeline.
    """
    if not isinstance(text, str):
        return ""
    normalized_text = normalize_text(text)
    tokens = tokenize_text(normalized_text)
    tokens_no_stopwords = remove_stopwords(tokens)
    lemmatized_tokens_list = lemmatize_tokens(tokens_no_stopwords)
    return ' '.join(lemmatized_tokens_list)

if __name__ == '__main__':
    # Example Usage
    sample_ticket_text = "My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404 on 2023-03-15."
    
    print(f"Original Text: {sample_ticket_text}")
    
    normalized = normalize_text(sample_ticket_text)
    print(f"Normalized: {normalized}")
    
    tokens = tokenize_text(normalized)
    print(f"Tokens: {tokens}")
    
    tokens_no_stop = remove_stopwords(tokens)
    print(f"Tokens (no stopwords): {tokens_no_stop}")
    
    lemmatized_tokens = lemmatize_tokens(tokens_no_stop)
    print(f"Lemmatized Tokens: {lemmatized_tokens}")
    
    processed_text = preprocess_text_pipeline(sample_ticket_text)
    print(f"Fully Preprocessed Text: {processed_text}")

    sample_ticket_text_2 = "The AlphaWidget is not working as expected. It's showing a strange message."
    processed_text_2 = preprocess_text_pipeline(sample_ticket_text_2)
    print(f"\nOriginal Text 2: {sample_ticket_text_2}")
    print(f"Fully Preprocessed Text 2: {processed_text_2}")

    # Test with non-string input
    processed_non_string = preprocess_text_pipeline(123)
    print(f"\nProcessed non-string: '{processed_non_string}'")
    
    processed_empty_string = preprocess_text_pipeline("")
    print(f"Processed empty string: '{processed_empty_string}'")
