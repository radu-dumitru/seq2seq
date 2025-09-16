"""
- movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance

    example: L868 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ The "real you".


- movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2','lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content

    example: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L271', 'L272', 'L273', 'L274', 'L275']

"""

import ast
import re

CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "who's": "who is",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mustn't": "must not",
    "n't": " not",
    "'re": " are",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "'s": " is"
}

# Helps shrink vocabulary size
def expand_contractions(text):
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    return text

def clean_sentence(sentence):
	sentence = sentence.lower().strip()

	# Remove text inside parentheses or brackets
	sentence = re.sub(r"\([^)]*\)", "", sentence)
	sentence = re.sub(r"\[[^)]*\]", "", sentence)

	# Expand contractions
	sentence = expand_contractions(sentence)

	# Keep only letters, numbers, and basic punctuation
	sentence = re.sub(r"[^a-z0-9!?',.]+", " ", sentence)

	# Separate punctuation from words (so "area!" -> "area !")
	sentence = re.sub(r"([!?',.])", r" \1 ", sentence)

	# Normalize punctuation
	sentence = re.sub(r"[.]{2,}", ".", sentence)             # "..." -> "."
	sentence = re.sub(r"[!?]{2,}", lambda m: m.group(0)[0], sentence)  # "!!!" -> "!"
	sentence = re.sub(r"\s+", " ", sentence).strip()         # collapse spaces

	return sentence

conversations = []

with open("data/movie_conversations.txt", "r", encoding="utf-8", errors="ignore") as f:
	lines = f.read().split("\n")
	for line in lines:
		parts = line.split(" +++$+++ ")
		if len(parts) == 4:
			conversations.append(ast.literal_eval(parts[3]))


movie_lines = {}

with open("data/movie_lines.txt" , "r", encoding="utf-8", errors="ignore") as f:
	lines = f.read().split("\n")
	for line in lines:
		parts = line.split(" +++$+++ ")
		if len(parts) == 5:
			movie_lines[parts[0]] = parts[4]

dialogs = []

for conversation in conversations:
	for i in range(len(conversation) - 1):
		if conversation[i] in movie_lines and conversation[i + 1] in movie_lines:
			line1 = clean_sentence(movie_lines[conversation[i]])
			line2 = clean_sentence(movie_lines[conversation[i + 1]])
			
			# Filter by length (2–15 words)
			if (2 <= len(line1.split()) <= 15 and 2 <= len(line2.split()) <= 15):
				dialogs.append((line1, line2))

with open("data/dialogs.txt", "w", encoding="utf-8") as f:
	file_content = ''
	for line1, line2 in dialogs:
		f.write(line1 + '\t' + line2 + '\n')

