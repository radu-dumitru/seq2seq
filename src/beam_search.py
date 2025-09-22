from constants import SOS_TOKEN, EOS_TOKEN
import numpy as np

def beam_search(decoder, h_init, c_init, word2idx, idx2word, beam_size = 3, max_len = 50, expand_topk = 50):
    h0, c0, p0 = decoder.forward_step(word2idx[SOS_TOKEN], h_init, c_init)
    p0 = p0.ravel()  # ensure 1D
    """
    np.argsort returns the indices that would sort the array in ascending order
    Since we negated the values, the largest probabilities become the smallest negatives
    we take only the top beam_size indices
    """
    top_idxs = np.argsort(-p0)[:beam_size]

    sequences = []

    for idx in top_idxs:
        prob = float(p0[idx])
        """
        each word has a probability
        the probability of the whole sequence is the product of the probabilities at each step
        multiplying many numbers < 1 quickly shrinks toward 0
        to avoid this, we work in log-space
        we used max(prob, 1e-12) to avoid np.log(0) (which is -inf)
        """
        logp = np.log(max(prob, 1e-12))
        sequences.append({
            "seq": [word2idx[SOS_TOKEN], int(idx)],
            "score": logp,
            "h": h0.copy(),
            "c": c0.copy()
        })

    for _ in range(max_len - 1):
        all_candidates = []

        if all(s["seq"][-1] == word2idx[EOS_TOKEN] for s in sequences):
            break

        for s in sequences:
            last_token = s["seq"][-1]
            if last_token == word2idx[EOS_TOKEN]:
                all_candidates.append(s)
                continue

            h_next, c_next, p_next = decoder.forward_step(last_token, s["h"], s["c"])
            p_next = p_next.ravel()
            idxs = np.argsort(-p_next)[:expand_topk]

            for idx in idxs:
                prob = float(p_next[idx])
                logp = np.log(max(prob, 1e-12))
                candidate = {
                    "seq": s["seq"] + [int(idx)],
                    "score": s["score"] + logp,
                    "h": h_next.copy(),
                    "c": c_next.copy()
                }
                all_candidates.append(candidate)

        sequences = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:beam_size]

    best = max(sequences, key=lambda x: x["score"])

    """
    the score of a sequence is typically the sum of log probabilities of its tokens
    log probabilities are negative (since probabilities are ≤ 1)
    every time you add another token, you add another negative number
    that means longer sequences accumulate more negative log-probs => lower scores
    as a result, the search will prefer shorter sequences
    to counteract this, you can normalize or penalize the score by sequence length

    Goal: pick the best sequence from sequences while correcting for length bias.
    x["score"]: this is the sum of log-probabilities over the tokens generated so far. Because log-probs are negative, “larger” (less negative) is better.
    len(x["seq"]) - 1: your seq includes the SOS token at the start. Subtracting 1 counts only the generated tokens after SOS (this might include EOS if it has been appended).
    max(1, ...): prevents division by zero for sequences that haven’t generated any token beyond SOS. So the denominator is at least 1.
    Division = length normalization: score / length converts summed log-prob (which favors short sequences) into an average log-prob per token, making different-length sequences comparable.
    In effect, this chooses the sequence with the highest average log-probability per generated token (ignoring the SOS token)

    Suppose two candidates:
    A: score = -3.0 over 3 tokens → avg = -1.0
    B: score = -2.2 over 2 tokens → avg = -1.1
    Raw sums pick B (-2.2 > -3.0). With normalization, A wins (-1.0 > -1.1) because its per-token quality is better.
    """
    best = max(sequences, key=lambda x: x["score"] / max(1, (len(x["seq"]) - 1)))

    tokens = [idx2word[idx] for idx in best["seq"] if idx not in (word2idx[SOS_TOKEN], word2idx[EOS_TOKEN])]
    return tokens

    