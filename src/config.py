# CELL 2 - PROJECT CONFIGURATION (V4: SIGLIP-2 & MERGED DATASET)
# ================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import logging
import torch

# ================================================================
# LOGGER
# ================================================================
LOGGER = logging.getLogger("TRAKE")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

# ================================================================
# DATASET & MERGED PATHS
# ================================================================
@dataclass
class DatasetConfig:

    # User must provide their own dataset path
    ROOT: Path = Path("./data")

    # User must provide their own feature files
    MERGED_EMBEDDING_PATH: Path = Path(
        "./data/merged_embeddings.npy"
    )

    MERGED_METADATA_PATH: Path = Path(
        "./data/merged_metadata.csv"
    )
    
@dataclass
class OutputConfig:

    ROOT: Path = Path("./output")
    CACHE_DIR: Path = Path("./output/cache")
    INDEX_DIR: Path = Path("./output/index")
    RESULT_DIR: Path = Path("./output/results")

# ================================================================
# MODEL 
# ================================================================
@dataclass
class ModelConfig:
    CLIP_MODEL: str = "google/siglip2-so400m-patch16-384" 
    CLIP_PRETRAIN: str = "webli" 

@dataclass
class RetrievalConfig:
    FEATURE_DIM: int = 1152  
    VISUAL_TOP_K: int = 100
    TEXT_TOP_K: int = 50
    TOP_N_VIDEO: int = 20 

@dataclass
class DanteConfig:
    TOP_M_PER_EVENT: int = 20
    MIN_GAP: int = 1
    MAX_GAP: int = 1000
    SIMILARITY_WEIGHT: float = 0.70
    TEMPORAL_WEIGHT: float = 0.30
    TOP_SEQUENCE: int = 10

@dataclass
class FusionConfig:
    VISUAL_WEIGHT: float = 0.60
    TEXT_WEIGHT: float = 0.40
    RRF_K: int = 60

# ================================================================
# SYSTEM
# ================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SystemConfig:
    DEVICE: str = DEVICE
    RANDOM_SEED: int = SEED
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4
    USE_FP16: bool = True

@dataclass
class Config:
    DATASET: DatasetConfig = field(default_factory=DatasetConfig)
    OUTPUT: OutputConfig = field(default_factory=OutputConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    RETRIEVAL: RetrievalConfig = field(default_factory=RetrievalConfig)
    DANTE: DanteConfig = field(default_factory=DanteConfig)
    FUSION: FusionConfig = field(default_factory=FusionConfig)
    SYSTEM: SystemConfig = field(default_factory=SystemConfig)

CFG = Config()

CFG.OUTPUT.CACHE_DIR.mkdir(parents=True, exist_ok=True)
CFG.OUTPUT.INDEX_DIR.mkdir(parents=True, exist_ok=True)
CFG.OUTPUT.RESULT_DIR.mkdir(parents=True, exist_ok=True)
