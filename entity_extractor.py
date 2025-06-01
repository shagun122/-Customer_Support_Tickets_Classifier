import re
import pandas as pd
from datefinder import find_dates # For robust date finding
# For product list generation, we'd typically load the data
# from data_utils import load_data, clean_data 

# --- Configuration for Entity Extraction ---
# This list would ideally be comprehensive or dynamically generated
# For now, a sample list. In a full pipeline, this would come from the 'product' column.
DEFAULT_PRODUCT_LIST = [
    "AlphaWidget", "BetaService", "GammaConnector", "DeltaPlatform", 
    "SuperProduct X1", "OmegaSuite", "ProductA", "ProductB"
]

COMPLAINT_KEYWORDS = [
    "abandon", "abysmal", "adverse", "afraid", "aggressive", "alarm", "angry", "annoy", 
    "anxious", "appall", "arrogant", "ashamed", "atrocious", "awful", "bad", "berate", 
    "bewilder", "bizarre", "blame", "bland", "bomb", "bore", "bother", "break", "broken", 
    "cancel", "cannot", "can't", "clumsy", "complain", "complaint", "confuse", "corrupt", 
    "crap", "crazy", "critical", "cry", "damage", "dead", "deceive", "defect", "defective", 
    "delay", "delayed", "denial", "deny", "deplorable", "depress", "derelict", "despair", 
    "destroy", "detest", "difficult", "dirty", "disappoint", "disaster", "disapprove", 
    "disbelief", "discontent", "discredit", "disgrace", "disgust", "dislike", "dismal", 
    "displease", "dispute", "dissatisf", "distort", "distress", "distrust", "doubt", 
    "dreadful", "dumb", "dump", "dupe", "embarrass", "empty", "enrage", "error", "explode", 
    "fail", "failure", "fake", "false", "fault", "faulty", "fear", "feeble", "fight", "filthy", 
    "flop", "fool", "forbid", "forget", "forgot", "fraud", "freeze", "frighten", "frustrate", 
    "furious", "garbage", "ghastly", "glitch", "grave", "greed", "grief", "grimy", "gross", 
    "guilt", "hang", "harass", "hard", "harm", "hate", "hazard", "helpless", "hesitate", 
    "horrible", "hostile", "hurt", "hysteria", "idiot", "ignore", "ill", "impossible", 
    "improper", "inability", "inaccurate", "inadequate", "incompetent", "incorrect", 
    "inferior", "infuriate", "inhibit", "injure", "insecure", "insincere", "insipid", 
    "insolent", "insult", "interfere", "interrupt", "intimidate", "irate", "ironic", 
    "irritate", "issue", "issues", "jealous", "junk", "lag", "late", "lie", "lose", "loss", 
    "lousy", "mad", "malign", "mean", "mess", "misbehave", "mischief", "miscreant", 
    "miserable", "misery", "misfortune", "mislead", "miss", "mistake", "mock", "monster", 
    "nag", "nasty", "naughty", "negative", "neglect", "nervous", "nonsense", "not working", 
    "object", "objection", "obnoxious", "obscene", "obstruct", "offend", "outrage", "overload", 
    "pain", "panic", "pathetic", "penalty", "peril", "perish", "perturb", "pessimism", 
    "petty", "pity", "plague", "poor", "poverty", "powerless", "pressing", "problem", 
    "problems", "prohibit", "protest", "provoke", "punish", "quit", "rage", "rattle", 
    "reject", "renege", "repel", "reprimand", "repulse", "resent", "resign", "restrain", 
    "restrict", "revenge", "revoke", "ridicule", "rigid", "risk", "rot", "rude", "ruin", 
    "sabotage", "sad", "sarcasm", "savage", "scam", "scandal", "scare", "scream", "severe", 
    "shaky", "shame", "shock", "shoddy", "shoot", "shortage", "sick", "sinister", "skeptic", 
    "slam", "slash", "sluggish", "smash", "smear", "smelly", "snare", "snub", "sorry", 
    "spite", "stall", "stalled", "steal", "stiff", "stink", "stole", "stop", "stopped", 
    "strange", "stress", "strict", "strife", "strike", "stubborn", "stuck", "stun", "stupid", 
    "substandard", "suck", "suffer", "suspect", "suspicion", "tense", "terrible", "terror", 
    "threat", "thwart", "timid", "tolerate", "torture", "toxic", "trap", "trash", "trick", 
    "trouble", "unable", "unacceptable", "unapproved", "unaware", "unbearable", "unbelievable", 
    "uncertain", "unclear", "uncomfortable", "uncontrolled", "undermine", "undesirable", 
    "uneasy", "unfair", "unfortunate", "unhappy", "unhealthy", "uniformed", "uninspired", 
    "unjust", "unlawful", "unlicensed", "unlucky", "unpleasant", "unprofessional", 
    "unqualified", "unreliable", "unsafe", "unsatisf", "unstable", "unsuccessful", 
    "unsupported", "unsure", "untamed", "unwanted", "unwelcome", "unwilling", "unwise", 
    "upset", "urgent", "useless", "vague", "vanish", "vex", "vibrate", "vicious", "violate", 
    "violence", "volatile", "vomit", "vulnerable", "warn", "warning", "waste", "weak", 
    "weary", "weird", "wicked", "wild", "wipeout", "woe", "worthless", "wound", "wreak", 
    "wreck", "wrong", "yell"
]


def extract_product_names(text: str, product_list: list = None) -> list:
    """
    Extracts product names from text based on a provided list.
    Uses regex for case-insensitive matching of whole words.
    """
    if not isinstance(text, str):
        return []
    if product_list is None:
        product_list = DEFAULT_PRODUCT_LIST
        
    found_products = set()
    # Sort product list by length descending to match longer names first (e.g., "SuperProduct X1" before "SuperProduct")
    sorted_product_list = sorted(product_list, key=len, reverse=True)

    for product_name in sorted_product_list:
        # Regex to find whole word, case insensitive
        # Escape special regex characters in product name
        escaped_product_name = re.escape(product_name)
        pattern = r"\\b" + escaped_product_name + r"\\b"
        if re.search(pattern, text, re.IGNORECASE):
            # Store the canonical product name from the list
            found_products.add(product_name) 
    return list(found_products)

def extract_dates(text: str) -> list:
    """
    Extracts dates from text using the datefinder library.
    Returns dates as strings in 'YYYY-MM-DD' format.
    """
    if not isinstance(text, str):
        return []
    try:
        dates_generator = find_dates(text, source=True)
        extracted_dates = []
        for date_obj, _ in dates_generator: # datefinder returns (datetime_obj, source_text)
            extracted_dates.append(date_obj.strftime('%Y-%m-%d'))
        return list(set(extracted_dates)) # Return unique dates
    except Exception: # Catch any error from datefinder
        return []


def extract_complaint_keywords(text: str, keyword_list: list = None) -> list:
    """
    Extracts predefined complaint keywords from text.
    Uses regex for case-insensitive matching of whole words.
    """
    if not isinstance(text, str):
        return []
    if keyword_list is None:
        keyword_list = COMPLAINT_KEYWORDS
        
    found_keywords = set()
    # Preprocess text slightly for keyword matching (lowercase)
    processed_text = text.lower() 

    for keyword in keyword_list:
        # Regex to find whole word
        pattern = r"\\b" + re.escape(keyword.lower()) + r"\\b"
        if re.search(pattern, processed_text):
            found_keywords.add(keyword) # Store original casing from keyword_list
    return list(found_keywords)

def extract_all_entities(text: str, product_list: list = None) -> dict:
    """
    Extracts all defined entities (products, dates, complaint keywords) from text.
    """
    if not isinstance(text, str):
        return {"products": [], "dates": [], "complaint_keywords": []}
        
    products = extract_product_names(text, product_list)
    dates = extract_dates(text)
    complaints = extract_complaint_keywords(text)
    
    return {
        "products": products,
        "dates": dates,
        "complaint_keywords": complaints
    }

if __name__ == '__main__':
    sample_ticket_text_1 = "My SuperProduct X1 is BROKEN after the latest Update!!! I need help ASAP. The error code is E404 on 2023-03-15. This is a disaster."
    sample_ticket_text_2 = "The AlphaWidget is not working as expected since 1st April 2024. It's showing a strange message and I am very unhappy. Also, my GammaConnector seems faulty."
    sample_ticket_text_3 = "Everything is great with BetaService! Just a quick question about billing for May 2023."
    sample_ticket_text_4 = "I have an issue with ProductA. It stopped working yesterday." # Yesterday is relative, datefinder might not get it without context.

    # In a real scenario, product_list would be dynamically loaded from the dataset
    # For example:
    # df_full = load_data('ai_dev_assignment_tickets_complex_1000.xls')
    # if not df_full.empty:
    #     df_cleaned = clean_data(df_full)
    #     dynamic_product_list = list(df_cleaned['product'].unique())
    #     # Filter out 'Unknown' if it was used as a placeholder
    #     dynamic_product_list = [p for p in dynamic_product_list if p != 'Unknown' and pd.notna(p)]
    # else:
    #     dynamic_product_list = DEFAULT_PRODUCT_LIST
    # print(f"Using dynamic product list: {dynamic_product_list[:5]}...") # Print first 5

    print("--- Entities from Sample Ticket 1 ---")
    entities1 = extract_all_entities(sample_ticket_text_1) # Uses DEFAULT_PRODUCT_LIST
    print(entities1)

    print("\\n--- Entities from Sample Ticket 2 ---")
    # Example with a custom product list for this call
    custom_list = ["AlphaWidget", "GammaConnector", "BetaService"]
    entities2 = extract_all_entities(sample_ticket_text_2, product_list=custom_list)
    print(entities2)

    print("\\n--- Entities from Sample Ticket 3 ---")
    entities3 = extract_all_entities(sample_ticket_text_3)
    print(entities3)
    
    print("\\n--- Entities from Sample Ticket 4 ---")
    entities4 = extract_all_entities(sample_ticket_text_4)
    print(entities4)

    # Test with empty or non-string input
    print("\\n--- Entities from Empty String ---")
    entities_empty = extract_all_entities("")
    print(entities_empty)

    print("\\n--- Entities from Non-String Input ---")
    entities_non_string = extract_all_entities(12345)
    print(entities_non_string)
