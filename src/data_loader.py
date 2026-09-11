import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from src.config import LOGGER, CFG
class MultimodalDataLoader:
    def __init__(self, dm, cfg=CFG):
        self.dm = dm
        self.cfg = cfg
        self.device = torch.device(cfg.SYSTEM.DEVICE)

        LOGGER.info("Loading SigLIP-2 Text Encoder via Hugging Face...")
        
        self.processor = AutoProcessor.from_pretrained(cfg.MODEL.CLIP_MODEL)
        self.model = AutoModel.from_pretrained(cfg.MODEL.CLIP_MODEL).to(self.device)
        self.model.eval()

        LOGGER.info("Multimodal DataLoader initialized.")

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
        
        text_outputs = self.model.get_text_features(**inputs)
        
        if not isinstance(text_outputs, torch.Tensor):
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_features = text_outputs.pooler_output
            elif hasattr(text_outputs, 'text_embeds') and text_outputs.text_embeds is not None:
                text_features = text_outputs.text_embeds
            else:
                text_features = text_outputs[0] # Fallback lấy phần tử đầu tiên
        else:
            text_features = text_outputs
            
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float32)

    def load_clip_features(self, video_id):
        return self.dm.load_clip_features(video_id)

    def load_transcript(self, video_id):
        candidates = [
            self.cfg.DATASET.ROOT / "transcripts" / f"{video_id}.txt",
            self.cfg.DATASET.ROOT / "transcript" / f"{video_id}.txt",
            self.cfg.DATASET.ROOT / "transcripts" / f"{video_id}.json",
            self.cfg.DATASET.ROOT / "transcript" / f"{video_id}.json",
        ]
        for path in candidates:
            if not path.exists(): continue
            if path.suffix == ".txt": return path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f: data = json.load(f)
                if isinstance(data, str): return data
                if isinstance(data, dict): return str(data.get("text", data.get("transcript", "")))
        return ""

    def load_summary(self, video_id):
        candidates = [
            self.cfg.DATASET.ROOT / "summaries" / f"{video_id}.txt",
            self.cfg.DATASET.ROOT / "summary" / f"{video_id}.txt",
            self.cfg.DATASET.ROOT / "summaries" / f"{video_id}.json",
            self.cfg.DATASET.ROOT / "summary" / f"{video_id}.json",
        ]
        for path in candidates:
            if not path.exists(): continue
            if path.suffix == ".txt": return path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f: data = json.load(f)
                if isinstance(data, str): return data
                if isinstance(data, dict): return str(data.get("summary", data.get("text", "")))
        return ""

    def load_video(self, video_id):
        return {
            "video_id": video_id,
            "keyframes": self.dm.load_keyframes(video_id),
            "clip_features": self.load_clip_features(video_id),
            "transcript": self.load_transcript(video_id),
            "summary": self.load_summary(video_id)
        }

