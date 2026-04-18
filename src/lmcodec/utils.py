import numpy as np


def stabilize_probabilities(probs: np.ndarray, vocab_size: int) -> np.ndarray:
    min_prob = 1e-10
    probs = np.maximum(probs, min_prob)

    if len(probs) > vocab_size:
        probs = probs[:vocab_size]
    elif len(probs) < vocab_size:
        padding = np.full(vocab_size - len(probs), min_prob)
        probs = np.concatenate([probs, padding])

    probs /= probs.sum()
    return probs
