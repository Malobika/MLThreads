import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

text = "Hello, welcome to the world of Natural Language Processing. Let's start with some basics!"

# Process the text
doc = nlp(text)

print("Tokenization:")
for token in doc:
    print(token.text)

print("\nPart of Speech Tagging:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
