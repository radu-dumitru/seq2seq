from encoder_lstm import EncoderLSTM
from decoder_lstm import DecoderLSTM
from vocab_builder import VocabBuilder
import numpy as np
from utils import tokenize, words_to_indices
from constants import SOS_TOKEN, EOS_TOKEN
from beam_search import beam_search
from hyperparams import HParams

hp = HParams()
vocab_builder = VocabBuilder()
word2idx, idx2word = vocab_builder.load_vocab()
vocab_size = len(word2idx)

params = np.load("data/model.npz", allow_pickle=True)
embedding = params["embedding"]
encoder = EncoderLSTM(embedding, hp.hidden_size)
decoder = DecoderLSTM(embedding, hp.hidden_size, vocab_size)
encoder.load_state_dict(params["encoder_state"].item())
decoder.load_state_dict(params["decoder_state"].item())

while True:
    user_prompt = input("You: ").strip()

    enc_in = words_to_indices(tokenize(user_prompt), word2idx) + [word2idx[EOS_TOKEN]]

    h0 = np.zeros((hp.hidden_size, 1))
    c0 = np.zeros((hp.hidden_size, 1))

    hT, cT = encoder.forward(enc_in, (h0, c0))

    response = beam_search(decoder, hT, cT, word2idx, idx2word)
    print("Bot:", " ".join(response))