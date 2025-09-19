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
from pathlib import Path
import zipfile
import io

INPUT_DIR = Path("data")
MOVIE_LINES = INPUT_DIR / "movie_lines.txt"
MOVIE_CONVERSATIONS = INPUT_DIR / "movie_conversations.txt"
OUTPUT_TXT = "dialogs.txt"
OUTPUT_ZIP = INPUT_DIR / "dialogs.zip"

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
    # keep contractions that are generally unambiguous:
    "'re": " are",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
}

# Compile contraction regex (longest keys first to avoid partial matches)
_contractions_pattern = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, CONTRACTIONS.keys()), key=len, reverse=True)) + r")\b"
)

# Precompile other regexes
_re_paren = re.compile(r"\([^)]*\)")
_re_bracket = re.compile(r"\[[^\]]*\]")   # FIXED: exclude closing bracket, not ')'
_re_keep = re.compile(r"[^a-z0-9!?',.]+")  # keep only these chars (after lowercasing)
_re_multi_dot = re.compile(r"\.{2,}")
_re_multi_punc = re.compile(r"[!?]{2,}")   # collapses sequences of ?/! (or mixed) -> first char
_re_punct_sep = re.compile(r"([!?',.])")
_re_spaces = re.compile(r"\s+")
_re_word = re.compile(r"[a-z0-9]+")

def expand_contractions(text: str) -> str:
    """Replace contractions using a single compiled regex and dict lookup."""
    return _contractions_pattern.sub(lambda m: CONTRACTIONS[m.group(0)], text)

def clean_sentence(sentence: str) -> str:
    """
    Clean a single sentence:
      - lowercase
      - remove parenthesis/bracketed text (non-nested)
      - expand contractions (careful about ambiguous "'s")
      - keep only a-z0-9 and basic punctuation
      - normalize repeated punctuation BEFORE separating punctuation
      - separate punctuation as own tokens (adds spaces around them)
      - collapse whitespace
    """
    s = sentence.lower().strip()

    # remove parenthetical content (non-nested)
    s = _re_paren.sub("", s)
    s = _re_bracket.sub("", s)

    # expand contractions (we lowercased, so keys match)
    s = expand_contractions(s)

    # keep only allowed characters
    s = _re_keep.sub(" ", s)

    # normalize repeated punctuation BEFORE we split punctuation into tokens:
    s = _re_multi_dot.sub(".", s)     # "..." -> "."
    s = _re_multi_punc.sub(lambda m: m.group(0)[0], s)  # "?!?!!" -> "?" (first char)

    # separate punctuation from words (so "area!" -> "area !")
    s = _re_punct_sep.sub(r" \1 ", s)

    # collapse spaces
    s = _re_spaces.sub(" ", s).strip()

    return s

def word_count(sentence: str) -> int:
    """Count "word" tokens (letters/digits only), ignores punctuation tokens."""
    return len(_re_word.findall(sentence))

def load_movie_lines(path: Path):
    """
    Return dict: lineID -> text
    Uses maxsplit to avoid accidentally splitting the text if the separator appears in utterance.
    """
    movie_lines = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split(" +++$+++ ", 4)  # expect 5 parts; limit splits
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[4]
                movie_lines[line_id] = text
    return movie_lines

def load_conversations(path: Path):
    """
    Return list of conversations (each is a list of lineIDs).
    Uses maxsplit to isolate the list-of-utterances safely.
    """
    conversations = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split(" +++$+++ ", 3)
            if len(parts) >= 4:
                rhs = parts[3]
                try:
                    utterance_list = ast.literal_eval(rhs)
                except Exception:
                    # malformed conversation line — skip
                    continue
                if isinstance(utterance_list, list):
                    conversations.append(utterance_list)
    return conversations

def build_dialog_pairs(movie_lines: dict, conversations: list, min_words=2, max_words=15):
    dialogs = []
    for conv in conversations:
        # pair each utterance with the next (chronological)
        for a, b in zip(conv, conv[1:]):
            if a in movie_lines and b in movie_lines:
                s1 = clean_sentence(movie_lines[a])
                s2 = clean_sentence(movie_lines[b])
                # count words ignoring punctuation tokens
                if min_words <= word_count(s1) <= max_words and min_words <= word_count(s2) <= max_words:
                    dialogs.append((s1, s2))
    return dialogs

def main():
    movie_lines = load_movie_lines(MOVIE_LINES)
    conversations = load_conversations(MOVIE_CONVERSATIONS)
    dialogs = build_dialog_pairs(movie_lines, conversations)
    # write output as a zip file containing dialogs.txt
    with zipfile.ZipFile(OUTPUT_ZIP, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        with io.StringIO() as buf:
            for l1, l2 in dialogs:
                buf.write(l1)
                buf.write("\t")
                buf.write(l2)
                buf.write("\n")
            zf.writestr(OUTPUT_TXT, buf.getvalue())
    print(f"Produced {len(dialogs)} dialog pairs into {OUTPUT_ZIP} ({OUTPUT_TXT} inside)")

if __name__ == "__main__":
    main()