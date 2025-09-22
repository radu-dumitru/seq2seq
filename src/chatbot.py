from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
from data_loader import DataLoader
from vocab_builder import VocabBuilder
import numpy as np
from utils import tokenize, words_to_indices
from constants import SOS_TOKEN, EOS_TOKEN
from beam_search import beam_search

embedding_dim = 100
hidden_size = 128
max_vocab_size = 8000
max_number_lines = 10000

data_loader = DataLoader()
data = data_loader.load_data(max_number_lines)

vocab_builder = VocabBuilder()
word2idx, idx2word = vocab_builder.build_vocab(data, max_vocab_size)
vocab_size = len(word2idx)

params = np.load("data/model.npz", allow_pickle=True)
embedding = params["embedding"]

encoder = EncoderLSTM(embedding, hidden_size)
decoder = DecoderLSTM(embedding, hidden_size, vocab_size)

while True:
    user_prompt = input("You: ").strip()

    enc_in = words_to_indices(tokenize(user_prompt), word2idx) + [word2idx[EOS_TOKEN]]

    h0 = np.zeros((hidden_size, 1))
    c0 = np.zeros((hidden_size, 1))

    xs_enc, hs_enc, cs_enc, os_enc, zs_enc = encoder.forward(enc_in, h0, c0)
    hT = hs_enc[len(xs_enc) - 1]
    cT = cs_enc[len(xs_enc) - 1]

    response = beam_search(decoder, hT, cT, word2idx, idx2word)
    print("Bot:", " ".join(response))