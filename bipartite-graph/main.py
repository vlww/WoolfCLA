import pandas as pd
import string
import re
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

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
words_to_check = ["old", "people", "felt", "man", "always", "without", "rather", "course", "day", "party", "lunch", "grey"]

# Clean text function
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))

# Clean paragraphs, dropping NaN values
cleaned_paragraphs = df['paragraph'].dropna().apply(clean_text)

# Store word frequencies and sentence counts per character
word_occurrences = {character: {word: 0 for word in words_to_check} for character in character_aliases.keys()}
character_sentence_counts = {}

# Analyze each character
for character, names in character_aliases.items():
    sentences_with_character = set()  # Avoid duplicate sentences

    # Create regex pattern for the character's names
    name_pattern = r'\b(?:' + '|'.join(map(re.escape, names)) + r')\b'
    for paragraph in cleaned_paragraphs:
        sentences = paragraph.split('.')  # Simple sentence splitting
        for sentence in sentences:
            if re.search(name_pattern, sentence):
                sentences_with_character.add(sentence.strip())

    total_sentences = len(sentences_with_character)
    character_sentence_counts[character] = total_sentences

    # Count occurrences of each specified word
    for sentence in sentences_with_character:
        for word in words_to_check:
            if re.search(r'\b' + re.escape(word) + r'\b', sentence):
                word_occurrences[character][word] += 1

# Convert to DataFrame (rows: characters, columns: words)
df_matrix = pd.DataFrame(word_occurrences).T.fillna(0)
for character in df_matrix.index:
    total_sentences = character_sentence_counts.get(character, 1)
    df_matrix.loc[character] = (df_matrix.loc[character] / total_sentences) * 100  # Percentage conversion

# ---------------- Build Bipartite Graph with Force-Directed Layout ----------------

# Create a new graph
G = nx.Graph()

# List nodes for characters and words
characters = list(df_matrix.index)
words = list(df_matrix.columns)

# Add nodes to the graph with a node attribute for clarity
for c in characters:
    G.add_node(c, type='character')
for w in words:
    G.add_node(w, type='word')

# Set an edge threshold to filter out very low connections (adjust as needed)
threshold = 1

# Add edges between characters and words if the percentage exceeds the threshold
for c in characters:
    for w in words:
        weight = df_matrix.loc[c, w]
        if weight >= threshold:
            G.add_edge(c, w, weight=weight)

# Use spring_layout to position nodes according to connection strength
pos = nx.spring_layout(G, weight='weight', k=0.5, iterations=200, seed=42)

plt.figure(figsize=(12, 10))

# Draw edges with 70% transparency
edges = G.edges(data=True)
edge_weights = [edata['weight'] * 0.3 for (_, _, edata) in edges]
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.5)

# Separate character and word nodes
character_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "character"]
word_nodes = [node for node in G.nodes if G.nodes[node]["type"] == "word"]

# Draw character nodes (square)
nx.draw_networkx_nodes(G, pos, nodelist=character_nodes, node_color="lightblue", node_shape="s", node_size=900)

# Draw word nodes (circle)
nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color="orange", node_shape="o", node_size=600)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.title("Force-Directed Bipartite Graph of Character-Word Connections", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
