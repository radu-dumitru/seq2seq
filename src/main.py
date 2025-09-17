from collections import Counter
import json
import os
import numpy as np
from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM

max_number_lines = 10000
max_vocab_size = 8000
embedding_dim = 100
hidden_size = 128

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

word2idx_file = "data/word2idx.json"
idx2word_file = "data/idx2word.json"

data = []

learning_rate = 1e-2
num_epochs = 10

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

vocab_size = len(word2idx)
print(f"Final vocab size: {len(word2idx)}")
print("Sample vocab:", list(word2idx.items())[:10])

def tokenize(text):
    return text.split()

def words_to_indices(words, word2idx):
    return [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]

embedding = np.random.randn(len(word2idx), embedding_dim) * 0.01

encoder = EncoderLSTM(embedding, hidden_size, learning_rate=learning_rate)
decoder = DecoderLSTM(embedding, hidden_size, vocab_size, learning_rate=learning_rate)

for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    
    total_loss = 0.0

    for i, (q, a) in enumerate(data):
        enc_in = words_to_indices(tokenize(q), word2idx) + [word2idx[EOS_TOKEN]]

        dec_in = [word2idx[SOS_TOKEN]] + words_to_indices(tokenize(a), word2idx)
        dec_out = words_to_indices(tokenize(a), word2idx) + [word2idx[EOS_TOKEN]]

        h0 = np.zeros((hidden_size, 1))
        c0 = np.zeros((hidden_size, 1))

        xs_enc, hs_enc, cs_enc, os_enc, zs_enc = encoder.forward(enc_in, h0, c0)
        hT = hs_enc[len(xs_enc) - 1]
        cT = cs_enc[len(xs_enc) - 1]

        xs_dec, hs_dec, cs_dec, os_dec, zs_dec, ys_dec, ps_dec = decoder.forward(dec_in, hT, cT)

        dWx_dec, dWh_dec, db_dec, dWhy_dec, dby_dec, dE_dec, dh_enc_from_dec, dc_enc_from_dec, loss = decoder.backward(
            xs_dec, hs_dec, cs_dec, os_dec, zs_dec, ys_dec, ps_dec, dec_out, dec_in
        )

        dWx_enc, dWh_enc, db_enc, dE_enc, _, _ = encoder.backward(
            xs_enc, hs_enc, cs_enc, os_enc, zs_enc, dh_enc_from_dec, dc_enc_from_dec, enc_in
        )

        encoder.update_params((dWx_enc, dWh_enc, db_enc, dE_enc))
        decoder.update_params((dWx_dec, dWh_dec, db_dec, dWhy_dec, dby_dec, dE_dec))

        dE_total = dE_enc + dE_dec
        used_idxs = np.unique(np.array(enc_in + dec_in, dtype=np.int32))
        embedding[used_idxs] -= learning_rate * dE_total[used_idxs]

        total_loss += loss

        print(total_loss)
