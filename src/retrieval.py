import faiss  
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from dataclasses import dataclass
from src.config import CFG
from src.config import LOGGER

@dataclass
class VisualRetrievalResult:
    video_id: str
    frame_idx: int
    similarity: float
    keyframe_path: str
    event_id: int

class VisualRetrievalIndex:
    def __init__(self, loader, cfg=CFG):
        self.loader = loader
        self.cfg = cfg
        
        self.dim = self.loader.dm.merged_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.records = []

    def build(self):
        print("=" * 70)
        print("BUILDING VISUAL FAISS INDEX (MERGED BATCH)")
        print("=" * 70)

        embeddings = self.loader.dm.merged_embeddings.copy() 
        df_meta = self.loader.dm.merged_metadata

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-12)

        self.index.add(embeddings)

        self.records = df_meta.to_dict('records')

        print(f"[OK] FAISS loaded {self.index.ntotal} vector embeddings.")
        print("=" * 70)

    def search(self, query_embedding, event_id, top_k=100):
        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query_embedding /= (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0: continue
            
            record = self.records[idx]
            
            vid = record.get("video_id", record.get("video", "UNKNOWN"))
            f_idx = record.get("frame_idx", record.get("frame", record.get("frame_id", 0)))
            k_path = record.get("keyframe_path", record.get("path", record.get("image_path", "")))

            results.append(
                VisualRetrievalResult(
                    video_id=vid,
                    frame_idx=int(f_idx),
                    similarity=float(score),
                    keyframe_path=str(k_path),
                    event_id=event_id
                )
            )
        return results




@dataclass
class TextRetrievalResult:

    video_id: str

    text_type: str

    text: str

    similarity: float


class TextRetrievalIndex:

    def __init__(
        self,
        loader,
        cfg=CFG
    ):

        self.loader = loader

        self.cfg = cfg

        self.records = []

        self.embeddings = None

        self.index = None


    def build(
        self,
        video_ids
    ):

        records = []

        for video_id in video_ids:

            transcript = self.loader.load_transcript(
                video_id
            )

            summary = self.loader.load_summary(
                video_id
            )

            if transcript:

                records.append({
                    "video_id": video_id,
                    "text_type": "transcript",
                    "text": transcript
                })

            if summary:

                records.append({
                    "video_id": video_id,
                    "text_type": "summary",
                    "text": summary
                })

        self.records = records

        if not records:

            LOGGER.warning(
                "No transcript/summary data found."
            )

            return

        texts = [
            r["text"]
            for r in records
        ]

        self.embeddings = \
            self.loader.encode_text(
                texts
            )

        self.index = faiss.IndexFlatIP(
            self.embeddings.shape[1]
        )

        self.index.add(
            self.embeddings
        )

        LOGGER.info(
            f"Text index built: "
            f"{len(records)} records"
        )


    def search(
        self,
        query,
        top_k=50
    ):

        if self.index is None:

            return []

        embedding = self.loader.encode_text(
            [query]
        )

        scores, indices = self.index.search(
            embedding,
            min(
                top_k,
                len(self.records)
            )
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            record = self.records[idx]

            results.append(
                TextRetrievalResult(
                    video_id=record["video_id"],
                    text_type=record["text_type"],
                    text=record["text"],
                    similarity=float(score)
                )
            )

        return results




@dataclass
class MultimodalCandidate:

    video_id: str

    event_id: int

    visual_score: float

    text_score: float

    combined_score: float

    visual_frames: list


class MultimodalCandidateRetriever:

    def __init__(
        self,
        loader,
        visual_index,
        text_index,
        cfg=CFG
    ):

        self.loader = loader
        self.visual_index = visual_index
        self.text_index = text_index
        self.cfg = cfg


    def retrieve_event(
        self,
        event
    ):

        embedding = self.loader.encode_text(
            [event.text]
        )[0]

        visual = self.visual_index.search(
            embedding,
            event.event_id,
            self.cfg.RETRIEVAL.VISUAL_TOP_K
        )

        text = self.text_index.search(
            event.text,
            self.cfg.RETRIEVAL.TEXT_TOP_K
        )

        visual_by_video = defaultdict(list)

        for r in visual:

            visual_by_video[
                r.video_id
            ].append(r)

        text_by_video = {

            r.video_id:
                r.similarity

            for r in text
        }

        candidates = []

        for video_id, frames in visual_by_video.items():

            visual_score = max(
                r.similarity
                for r in frames
            )

            text_score = text_by_video.get(
                video_id,
                0.0
            )

            combined = (
                self.cfg.FUSION.VISUAL_WEIGHT
                * visual_score
                +
                self.cfg.FUSION.TEXT_WEIGHT
                * text_score
            )

            candidates.append(
                MultimodalCandidate(
                    video_id=video_id,
                    event_id=event.event_id,
                    visual_score=visual_score,
                    text_score=text_score,
                    combined_score=combined,
                    visual_frames=frames
                )
            )

        return sorted(
            candidates,
            key=lambda x: x.combined_score,
            reverse=True
        )




class CandidateFusion:
    def __init__(
        self,
        retriever,
        cfg=CFG
    ):

        self.retriever = retriever
        self.cfg = cfg

    def retrieve(
        self,
        events
    ):

        by_video = defaultdict(
            lambda: defaultdict(list)
        )

        for event in events:

            candidates = \
                self.retriever.retrieve_event(
                    event
                )

            for candidate in candidates:

                by_video[
                    candidate.video_id
                ][
                    candidate.event_id
                ].append(
                    candidate
                )

        fused = []

        for video_id, event_dict in by_video.items():

            event_scores = {}

            for event_id, candidates in event_dict.items():

                event_scores[event_id] = max(
                    c.combined_score
                    for c in candidates
                )

            coverage = (
                len(event_scores)
                /
                len(events)
            )

            score = (
                sum(event_scores.values())
                /
                max(len(event_scores), 1)
            )

            score *= (
                0.7 +
                0.3 * coverage
            )

            fused.append({

                "video_id": video_id,

                "score": score,

                "coverage": coverage,

                "events": event_dict
            })

        fused.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return fused[
            :self.cfg.RETRIEVAL.TOP_N_VIDEO
        ]
