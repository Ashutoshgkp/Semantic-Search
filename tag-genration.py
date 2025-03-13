import spacy
from keybert import KeyBERT
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample blog content
blog_content = """
Team-building is crucial for any workplace, and "21 Conversations" is a fun and easy way to break the ice. By encouraging open-ended discussions, this game helps teams build trust, improve communication, and foster collaboration. Whether used in meetings or casual work settings, it can lead to meaningful connections among colleagues. The key is to create a safe space where everyone feels comfortable sharing. Try incorporating these questions into your next team session and watch as bonds strengthen!
"""

# Function to extract nouns using spaCy
def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns

# Function to extract keywords using KeyBERT
def extract_keywords(text, top_n=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [keyword[0] for keyword in keywords]

# Custom stopwords
custom_stopwords = ["key", "way", "game", "questions"]

# Extract nouns and keywords
nouns = extract_nouns(blog_content)
keywords = extract_keywords(blog_content)

# Filter out custom stopwords
nouns = [noun for noun in nouns if noun.lower() not in custom_stopwords]
keywords = [keyword for keyword in keywords if keyword.lower() not in custom_stopwords]

# Combine and rank the most frequent terms
all_terms = nouns + keywords
term_frequencies = Counter(all_terms)

# Get the top 10 most frequent terms as tags
tags = term_frequencies.most_common(10)
print("Generated Tags:", [tag[0] for tag in tags])