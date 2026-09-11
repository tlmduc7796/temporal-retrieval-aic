import numpy as np
from dataclasses import dataclass
from src.config import CFG

@dataclass
class DanteCandidate:
    frame_idx: int
    score: float
    keyframe_path: str

class TRAKE:
    def __init__(self, query_parser, candidate_fusion, dante, frame_generator, loader, cfg=CFG):
        self.query_parser = query_parser
        self.candidate_fusion = candidate_fusion
        self.dante = dante
        self.frame_generator = frame_generator
        self.loader = loader
        self.cfg = cfg

    def _local_search_candidates(self, video_id, events):
        video_features = self.loader.load_clip_features(video_id)
        if video_features is None: return []

        norms = np.linalg.norm(video_features, axis=1, keepdims=True)
        video_features = video_features / (norms + 1e-12)
        
        keyframes = self.loader.dm.load_keyframes(video_id)
        event_candidates = []

        for event in events:
            original_text = getattr(event, "original_text", event.text)
            query_emb = self.loader.encode_text([original_text])[0]
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
            sims = np.dot(video_features, query_emb)
            
            k = min(self.cfg.RETRIEVAL.VISUAL_TOP_K, len(sims))
            top_indices = np.argsort(sims)[::-1][:k]

            candidates = []
            for idx in top_indices:
                if idx < len(keyframes):
                    candidates.append(DanteCandidate(int(keyframes[idx].stem), float(sims[idx]), str(keyframes[idx])))

            candidates.sort(key=lambda x: x.frame_idx)
            event_candidates.append(candidates)
            
        return event_candidates

    def _run_dante(self, video_id, event_candidates):
        if any(len(candidates) == 0 for candidates in event_candidates): return None
        return self.dante.optimize(video_id, event_candidates)

    def search(self, query):
        events = self.query_parser.parse(query)
        if not events: return {"results": []}

        fusion_output = self.candidate_fusion.retrieve(events)
        if not fusion_output: return {"results": []}

        dante_passed_videos = []
        for video_result in fusion_output:
            video_id = video_result.get("video_id") if isinstance(video_result, dict) else getattr(video_result, "video_id", None)
            if not video_id: continue

            event_candidates = self._local_search_candidates(video_id, events)
            if not event_candidates: continue
            
            sequence = self._run_dante(video_id, event_candidates)
            if sequence is None: continue
            
            dante_passed_videos.append({
                "video_id": video_id,
                "sequence": sequence,
                "dante_score": sequence.get("score", 0.0)
            })

        dante_passed_videos.sort(key=lambda x: x["dante_score"], reverse=True)
        top_videos = dante_passed_videos[:5]  

        final_results = []

        for video_data in top_videos:
            video_id = video_data["video_id"]
            sequence = video_data["sequence"]
            
            candidates_for_display = self.frame_generator.generate(sequence, video_id)
            
            ranked_frames = []
            for cand in candidates_for_display:
                visual_score = getattr(cand, "dante_score", getattr(cand, "similarity", 0.0))
                ranked_frames.append({
                    "video_id": video_id, 
                    "frame_idx": cand.frame_idx, 
                    "event_id": cand.event_id,
                    "keyframe_path": cand.keyframe_path, 
                    "visual_score": float(visual_score),
                    "semantic_score": 1.0, 
                    "verified": True,      
                    "final_score": float(visual_score)
                })

            final_results.append({
                "video_id": video_id,
                "score": video_data["dante_score"],
                "events": ranked_frames
            })

        return {"results": final_results}