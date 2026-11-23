from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

@dataclass
class ModelConfig:
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3
    bidirectional: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"

@dataclass
class DataConfig:
    max_seq_len: int = 57
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    pad_lemma: str = "<PAD>"
    unk_lemma: str = "<UNK>"
    pad_token_id: int = 0
    unk_token_id: int = 1
    pad_lemma_id: int = 0
    unk_lemma_id: int = 1

@dataclass
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    checkpoints_dir: Path = Path("checkpoints")
    reports_dir: Path = Path("reports")

@dataclass
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathsConfig

def load_config(config_path: str = "config.yaml") -> ProjectConfig:
    path = Path(config_path)
    if not path.exists():
        cfg_dict = {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    model_cfg = ModelConfig(**cfg_dict.get("model", {}))
    train_cfg = TrainingConfig(**cfg_dict.get("training", {}))
    data_cfg = DataConfig(**cfg_dict.get("data", {}))

    paths_dict = cfg_dict.get("paths", {})
    paths_cfg = PathsConfig(
        raw_dir=Path(paths_dict.get("raw_dir", "data/raw")),
        processed_dir=Path(paths_dict.get("processed_dir", "data/processed")),
        artifacts_dir=Path(paths_dict.get("artifacts_dir", "artifacts")),
        checkpoints_dir=Path(paths_dict.get("checkpoints_dir", "checkpoints")),
        reports_dir=Path(paths_dict.get("reports_dir", "reports")),
    )

    # Création automatique des répertoires requis
    for p in [paths_cfg.raw_dir, paths_cfg.processed_dir, paths_cfg.artifacts_dir, paths_cfg.checkpoints_dir, paths_cfg.reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Réglage du device (fallback CPU si pas de CUDA)
    if train_cfg.device == "cuda" and (not HAS_TORCH or not torch.cuda.is_available()):
        train_cfg.device = "cpu"

    # Fixation des graines pour la reproductibilité
    set_seed(train_cfg.seed)

    return ProjectConfig(
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
        paths=paths_cfg
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
