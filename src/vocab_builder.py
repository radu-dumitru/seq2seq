import os
import json
from collections import Counter
from constants import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class VocabBuilder:
    def __init__(self):
        self.word2idx_file = "data/word2idx.json"
        self.idx2word_file = "data/idx2word.json"

    def build_vocab(self, data, max_vocab_size):
        if os.path.exists(self.word2idx_file) and os.path.exists(self.idx2word_file):
            print("Loading vocab from files...")
            with open(self.word2idx_file, "r", encoding="utf-8") as f:
                word2idx = json.load(f)
            with open(self.idx2word_file, "r", encoding="utf-8") as f:
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

            with open(self.word2idx_file, "w", encoding="utf-8") as f:
                json.dump(word2idx, f, ensure_ascii=False, indent=2)

            with open(self.idx2word_file, "w", encoding="utf-8") as f:
                json.dump(idx2word, f, ensure_ascii=False, indent=2)

            print("Saved vocab files.")

        vocab_size = len(word2idx)
        print(f"Final vocab size: {len(word2idx)}")
        print("Sample vocab:", list(word2idx.items())[:10])

        return (word2idx, idx2word)

    def load_vocab(self):
        if os.path.exists(self.word2idx_file) and os.path.exists(self.idx2word_file):
            with open(self.word2idx_file, "r", encoding="utf-8") as f:
                word2idx = json.load(f)
            with open(self.idx2word_file, "r", encoding="utf-8") as f:
                idx2word = {int(k): v for k, v in json.load(f).items()}
            return word2idx, idx2word
        else:
            raise FileNotFoundError("Vocab files not found. Train first to build and save vocabulary.")