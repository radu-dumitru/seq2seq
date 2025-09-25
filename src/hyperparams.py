from dataclasses import dataclass

@dataclass(frozen=True)
class HParams:
    max_number_lines: int = 10000
    max_vocab_size: int = 8000
    embedding_dim: int = 100
    hidden_size: int = 128
    learning_rate: float = 1e-2
    num_epochs: int = 10
    checkpoint_interval: int = 100


