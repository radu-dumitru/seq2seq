import random
from constants import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class DataLoader:
    def __init__(self):
        self.word2idx = None
        self.idx2word = None

    def sentence_to_indices(self, tokens):
        indices = [self.word2idx.get(SOS_TOKEN)] + [self.word2idx.get(w) for w in tokens] + [self.word2idx.get(EOS_TOKEN)]
        return indices

    def load_data(self):
        filename="data/data.txt"
        src_tokens = []
        tgt_tokens = []
        X, Y = [], []
        unique_words = []
        
        with open(filename, "r", encoding="utf-8") as f:
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

        for x, y in zip(src_tokens, tgt_tokens):
            X.append(self.sentence_to_indices(x))
            Y.append(self.sentence_to_indices(y))

        return X, Y
