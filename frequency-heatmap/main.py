import pandas as pd
import string
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os
import platform
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load CSV file
df = pd.read_csv('dally.csv', header=None, names=['paragraph'])

# Character list (including alternate names)
character_aliases = {
    'Clarissa Dalloway': ['clarissa', 'mrs dalloway', 'clarissas'],
    'Richard Dalloway': ['richard', 'richard dalloway', 'richards'],
    'Hugh Whitbread': ['hugh', 'hugh whitbread', 'whitbread', 'hughs'],
    'Peter Walsh': ['peter', 'peter walsh', 'walsh', 'peters'],
    'Lady Bruton': ['lady bruton', 'bruton'],
    'Sally Seton': ['sally', 'sally seton', 'seton', 'sallys', 'rosseter', 'lady rosseter'],
    'Elizabeth Dalloway': ['elizabeth', 'miss dalloway', 'elizabeths'],
    'Doris Kilman': ['kilman', 'miss kilman', 'doris'],
    'Septimus Warren Smith': ['septimus', 'mr warren smith'],
    'Lucrezia Warren Smith': ['rezia', 'lucrezia', 'mrs warren smith'],
    'Dr. Holmes': ['dr holmes', 'holmes', 'rezias'],
    'Sir William Bradshaw': ['bradshaw', 'sir william bradshaw', 'william', 'bradshaws']
}

# Words to track
words_to_check = ["old", "people", "felt", "always", "without", "rather", "course", "party", "day", "lunch", "grey", "man"]

# Flatten character names for filtering
all_character_names = {name.lower() for names in character_aliases.values() for name in names}

# Clean text function
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))

# Clean paragraphs, dropping NaN values
cleaned_paragraphs = df['paragraph'].dropna().apply(clean_text)

# Additional stopwords
extra_stops = ["said", "thought", "would", "could", "one", "like", "see", "went", "saw", "come"]
stop_words.update(extra_stops)

# Store word frequencies and sentence counts
word_occurrences = {character: {word: 0 for word in words_to_check} for character in character_aliases.keys()}
character_sentence_counts = {}

# Analyze each character
for character, names in character_aliases.items():
    sentences_with_character = set()  # Avoid duplicate sentences

    # Create regex pattern for character's names
    name_pattern = r'\b(?:' + '|'.join(map(re.escape, names)) + r')\b'

    for paragraph in cleaned_paragraphs:
        sentences = paragraph.split('.')  # Simple sentence splitting

        # Store sentences that mention the character (without double counting)
        for sentence in sentences:
            if re.search(name_pattern, sentence):
                sentences_with_character.add(sentence.strip())

    # Count total unique sentences for this character
    total_sentences = len(sentences_with_character)
    character_sentence_counts[character] = total_sentences

    # Count occurrences of specified words
    for sentence in sentences_with_character:
        for word in words_to_check:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence):
                word_occurrences[character][word] += 1

# Convert to DataFrame
df_matrix = pd.DataFrame(word_occurrences).T.fillna(0)  # Transpose for proper format

# Convert counts to percentages
for character in df_matrix.index:
    total_sentences = character_sentence_counts.get(character, 1)  # Avoid division by zero
    df_matrix.loc[character] = (df_matrix.loc[character] / total_sentences) * 100  # Convert to percentage

# Print as a readable table
print(df_matrix)

# Plot as a heatmap
plt.figure(figsize=(14, 8))  # Adjusted for better readability
sns.heatmap(df_matrix, cmap="Oranges", annot=True, fmt=".1f", linewidths=0.5, vmin=0, vmax=13)

plt.title("Percentage of Sentences Containing Specific Words for Each Character", fontsize=14)
plt.xlabel("Words", fontsize=12)
plt.ylabel("Characters", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()  # Prevent cropping

plt.show()
