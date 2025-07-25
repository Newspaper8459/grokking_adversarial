from pathlib import Path
from typing import Literal

from omegaconf import DictConfig


class WandbConfig:  # noqa: D101
  project: str

class TrainConfig:  # noqa: D101
  num_epochs: int
  batch_size: int
  learning_rate: float
  weight_decay: float
  loss: Literal['ce', 'mse']
  label_smoothing: float
  subset_size: int
  grad_norm: float
  initialization_scale: float

class ValConfig:  # noqa: D101
  batch_size: int

class TransformerConfig:  # noqa: D101
  layers: int
  width: int
  heads: int

class MLPConfig:  # noqa: D101
  hidden_dim: int
  hidden_layers: int

class Config(DictConfig):  # noqa: D101
  debug_mode: bool
  seed: int
  input_dir: str
  input_path: Path
  output_dir: str
  output_path: Path
  device: str
  wandb: WandbConfig
  num_workers: int
  train: TrainConfig
  val: ValConfig
  model: str
  history: bool
  transformer: TransformerConfig
  mlp: MLPConfig
