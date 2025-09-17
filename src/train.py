import numpy as np
from data_loader import DataLoader
from vocab_builder import VocabBuilder
from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
from utils import tokenize, words_to_indices
from constants import SOS_TOKEN, EOS_TOKEN
import os

def save_model_params(embedding, encoder, decoder):
    np.savez(
        "data/model.npz",
        embedding=embedding,
        encoder_Wx=encoder.Wx,
        encoder_Wh=encoder.Wh,
        encoder_b=encoder.b,
        decoder_Wx=decoder.Wx,
        decoder_Wh=decoder.Wh,
        decoder_b=decoder.b,
        decoder_Why=decoder.Why,
        decoder_by=decoder.by
    )

max_number_lines = 10000
max_vocab_size = 8000
embedding_dim = 100
hidden_size = 128
learning_rate = 1e-2
num_epochs = 10
checkpoint_interval = 100

data_loader = DataLoader()
data = data_loader.load_data(max_number_lines)

vocab_builder = VocabBuilder()
word2idx, idx2word = vocab_builder.build_vocab(data, max_vocab_size)
vocab_size = len(word2idx)

if os.path.exists("data/model.npz"):
    params = np.load("data/model.npz", allow_pickle=True)
    embedding = params["embedding"]
else:
    embedding = np.random.randn(len(word2idx), embedding_dim) * 0.01

encoder = EncoderLSTM(embedding, hidden_size, learning_rate)
decoder = DecoderLSTM(embedding, hidden_size, vocab_size, learning_rate)

for epoch in range(num_epochs):
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

        if (i + 1) % checkpoint_interval == 0:
            avg_loss = total_loss / checkpoint_interval
            print(f"Epoch {epoch+1} iter {i+1}/{len(data)} — avg loss: {avg_loss:.4f}")
            total_loss = 0.0
            save_model_params(embedding, encoder, decoder)
    
    print(f"Epoch {epoch+1} finished")
    save_model_params(embedding, encoder, decoder)
