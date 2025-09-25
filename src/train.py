import numpy as np
from data_loader import DataLoader
from vocab_builder import VocabBuilder
from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
from utils import tokenize, words_to_indices, clip_grad_norm_, save_model_params, load_model_params
from hyperparams import HParams
from constants import SOS_TOKEN, EOS_TOKEN
import os
from adam_optimizer import Adam

hp = HParams()
step = 0

data_loader = DataLoader()
data = data_loader.load_data(hp.max_number_lines)

vocab_builder = VocabBuilder()
word2idx, idx2word = vocab_builder.build_vocab(data, hp.max_vocab_size)
vocab_size = len(word2idx)

embedding = np.random.randn(len(word2idx), hp.embedding_dim) * 0.01

optimizer = Adam(hp.learning_rate)

encoder = EncoderLSTM(embedding, hp.hidden_size, hp.learning_rate)
decoder = DecoderLSTM(embedding, hp.hidden_size, vocab_size, hp.learning_rate)

embedding_loaded, step = load_model_params(encoder, decoder, optimizer)
if embedding_loaded is not None:
    embedding = embedding_loaded
    encoder.embedding = embedding
    decoder.embedding = embedding

optimizer.add_parameters([
    *encoder.parameters(),
    *decoder.parameters(),
])

for epoch in range(hp.num_epochs):
    total_loss = 0.0

    for i, (q, a) in enumerate(data):
        step += 1

        enc_in = words_to_indices(tokenize(q), word2idx) + [word2idx[EOS_TOKEN]]

        dec_in = [word2idx[SOS_TOKEN]] + words_to_indices(tokenize(a), word2idx)
        dec_out = words_to_indices(tokenize(a), word2idx) + [word2idx[EOS_TOKEN]]

        h0 = np.zeros((hp.hidden_size, 1))
        c0 = np.zeros((hp.hidden_size, 1))

        (hT, cT) = encoder.forward(enc_in, (h0, c0))
        (hT_dec, cT_dec) = decoder.forward(dec_in, (hT, cT))

        dh_enc_from_dec, dc_enc_from_dec, loss = decoder.backward(dec_out)
        encoder.backward(dh_enc_from_dec, dc_enc_from_dec)

        clip_grad_norm_(optimizer.params, 5.0)
        optimizer.step(step)

        dE_total = encoder.dE + decoder.dE
        # clip gradients
        np.clip(dE_total, -5, 5, out=dE_total)
        used_idxs = np.unique(np.array(enc_in + dec_in, dtype=np.int32))
        embedding = optimizer.update_embedding("embedding", embedding, dE_total, used_idxs, step)

        total_loss += loss

        if (i + 1) % hp.checkpoint_interval == 0:
            avg_loss = total_loss / hp.checkpoint_interval
            print(f"Epoch {epoch+1} iter {i+1}/{len(data)} — avg loss: {avg_loss:.4f}")
            total_loss = 0.0
            save_model_params(embedding, encoder, decoder, optimizer, step)
    
    print(f"Epoch {epoch+1} finished")
    save_model_params(embedding, encoder, decoder, optimizer, step)
