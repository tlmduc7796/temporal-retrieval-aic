from pathlib import Path

import numpy as np
import pandas as pd


class DataManager:
    """
    Manage video metadata and precomputed visual embeddings.

    Expected inputs:
        - metadata CSV
        - embeddings .npy

    The dataset paths are provided by the user and are not hard-coded.
    """

    def __init__(
        self,
        embedding_path,
        metadata_path,
    ):
        self.embedding_path = Path(embedding_path)
        self.metadata_path = Path(metadata_path)

        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {self.embedding_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.merged_embeddings = np.load(
            self.embedding_path
        ).astype(np.float32)

        self.merged_metadata = pd.read_csv(
            self.metadata_path
        )

        self._detect_columns()
        self._build_video_index()

    def _detect_columns(self):
        """Automatically detect video ID and keyframe path columns."""

        columns = self.merged_metadata.columns.tolist()

        if "video_id" in columns:
            self.vid_col = "video_id"
        elif "video" in columns:
            self.vid_col = "video"
        else:
            self.vid_col = columns[0]

        if "keyframe_path" in columns:
            self.path_col = "keyframe_path"
        elif "path" in columns:
            self.path_col = "path"
        elif "image_path" in columns:
            self.path_col = "image_path"
        else:
            self.path_col = columns[-1]

    def _build_video_index(self):
        """Build mapping from video ID to embedding/metadata indices."""

        self.video_to_indices = (
            self.merged_metadata
            .groupby(self.vid_col)
            .groups
        )

    def load_keyframes(self, video_id):
        """Return keyframe paths belonging to a video."""

        if video_id not in self.video_to_indices:
            return []

        indices = self.video_to_indices[video_id]

        paths = self.merged_metadata.loc[
            indices, self.path_col
        ].tolist()

        return [Path(str(path)) for path in paths]

    def load_clip_features(self, video_id):
        """Return precomputed visual embeddings for a video."""

        if video_id not in self.video_to_indices:
            return None

        indices = self.video_to_indices[video_id]

        return self.merged_embeddings[indices]

    def get_video_ids(self):
        """Return all available video IDs."""

        return sorted(self.video_to_indices.keys())