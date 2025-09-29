from dataclasses import dataclass

@dataclass(frozen=True)
class HParams:
    embedding_dim: int = 50
    hidden_size: int = 100
    learning_rate: float = 1e-3
    num_epochs: int = 15
    checkpoint_interval: int = 100
