from dataclasses import dataclass

@dataclass(frozen=True)
class HParams:
    embedding_dim: int = 100
    hidden_size: int = 150
    learning_rate: float = 1e-3
    num_epochs: int = 20
    checkpoint_interval: int = 100
