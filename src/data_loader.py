from constants import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
import os
import json

class DataLoader:
    def __init__(self):
        self.filename="data/data.txt"
        self.word2idx_file = "data/word2idx.json"
        self.idx2word_file = "data/idx2word.json"
        
        self.word2idx = None
        self.idx2word = None

        if os.path.exists(self.word2idx_file) and os.path.exists(self.idx2word_file):
            with open(self.word2idx_file, "r", encoding="utf-8") as f:
                self.word2idx = json.load(f)
            with open(self.idx2word_file, "r", encoding="utf-8") as f:
                self.idx2word = {int(k): v for k, v in json.load(f).items()}

    def sentence_to_indices(self, tokens):
        indices = [self.word2idx.get(SOS_TOKEN)] + [self.word2idx.get(w) for w in tokens] + [self.word2idx.get(EOS_TOKEN)]
        return indices

    def load_data(self):
        src_tokens = []
        tgt_tokens = []
        X, Y = [], []
        unique_words = []
        
        with open(self.filename, "r", encoding="utf-8") as f:
            for x in f:
                parts = x.split("\t")
                src = parts[0].split()
                tgt = parts[1].split()
                unique_words.extend(src)
                unique_words.extend(tgt)
                src_tokens.append(src)
                tgt_tokens.append(tgt)

        unique_words = sorted(set(unique_words))
        print('number of unique words: ', len(unique_words))
        all_words = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + unique_words
        self.word2idx = {w: i for i, w in enumerate(all_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        with open(self.word2idx_file, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

        with open(self.idx2word_file, "w", encoding="utf-8") as f:
            json.dump(self.idx2word, f, ensure_ascii=False, indent=2)

        for x, y in zip(src_tokens, tgt_tokens):
            X.append(self.sentence_to_indices(x))
            Y.append(self.sentence_to_indices(y))

        return X, Y
