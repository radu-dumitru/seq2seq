from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
import numpy as np
from utils import tokenize, words_to_indices
from constants import SOS_TOKEN, EOS_TOKEN
from beam_search import beam_search
from hyperparams import HParams
from data_builder import DataBuilder

hp = HParams()
data_builder = DataBuilder()
word2idx = data_builder.word2idx
idx2word = data_builder.idx2word
vocab_size = len(word2idx)

params = np.load("data/model.npz", allow_pickle=True)
embedding = params["embedding"]
encoder = EncoderLSTM(embedding, hp.hidden_size)
decoder = DecoderLSTM(embedding, hp.hidden_size, vocab_size)
encoder.load_state_dict(params["encoder_state"].item())
decoder.load_state_dict(params["decoder_state"].item())

while True:
    user_prompt = input("You: ").strip()

    enc_in = [word2idx[SOS_TOKEN]] + words_to_indices(tokenize(user_prompt), word2idx) + [word2idx[EOS_TOKEN]]

    h0 = np.zeros((hp.hidden_size, 1))
    c0 = np.zeros((hp.hidden_size, 1))

    hT, cT = encoder.forward(enc_in, (h0, c0))

    response = beam_search(decoder, hT, cT, word2idx, idx2word)
    print("Bot:", " ".join(response))
