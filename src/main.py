from collections import Counter
import json
import os

max_number_lines = 10000
max_vocab_size = 8000 

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

word2idx_file = "data/word2idx.json"
idx2word_file = "data/idx2word.json"

data = []

with open("data/dialogs.txt", "r") as f:
    for i, line in enumerate(f):
        if i == max_number_lines:
            break

        q, a = line.strip().split("\t")
        data.append((q, a))

print(f"Loaded {len(data)} dialog pairs")

if os.path.exists(word2idx_file) and os.path.exists(idx2word_file):
    print("Loading vocab from files...")
    with open(word2idx_file, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    with open(idx2word_file, "r", encoding="utf-8") as f:
        idx2word = {int(k): v for k, v in json.load(f).items()}
else:
    print("Building vocab from scratch...")
    
    word_counter = Counter()
    for q, a in data:
        word_counter.update(q.split())
        word_counter.update(a.split())

    print(f"Total unique words (before trimming): {len(word_counter)}")

    word2idx = {
        SOS_TOKEN: 0,
        EOS_TOKEN: 1,
        UNK_TOKEN: 2,
    }

    most_common = word_counter.most_common(max_vocab_size - len(word2idx))
    for idx, (word, _) in enumerate(most_common, start=len(word2idx)):
        word2idx[word] = idx

    idx2word = {i: w for w, i in word2idx.items()}

    with open("data/word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    with open("data/idx2word.json", "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False, indent=2)

    print("Saved vocab files.")

print(f"Final vocab size: {len(word2idx)}")
print("Sample vocab:", list(word2idx.items())[:10])