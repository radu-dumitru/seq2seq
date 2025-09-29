from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
from hyperparams import HParams
from adam_optimizer import Adam
from data_builder import DataBuilder
from utils import clip_grad_norm_, save_model_params, load_model_params
import numpy as np

hp = HParams()
step = 0

data_builder = DataBuilder()
X, Y = data_builder.generate_large_dataset()
vocab_size = len(data_builder.word2idx)

embedding = np.random.randn(len(data_builder.word2idx), hp.embedding_dim) * 0.01

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

    for i, (q, a) in enumerate(zip(X, Y)):
        step += 1

        enc_in = q

        dec_in = a[:-1]
        dec_out = a[1:]

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

        # ensure encoder / decoder keep the same reference (defensive)
        encoder.embedding = embedding
        decoder.embedding = embedding

        total_loss += loss

        if (i + 1) % hp.checkpoint_interval == 0:
            avg_loss = total_loss / hp.checkpoint_interval
            print(f"Epoch {epoch+1} iter {i+1}/{len(X)} — avg loss: {avg_loss:.4f}")
            total_loss = 0.0
            save_model_params(embedding, encoder, decoder, optimizer, step)
    
    print(f"Epoch {epoch+1} finished")
    save_model_params(embedding, encoder, decoder, optimizer, step)


