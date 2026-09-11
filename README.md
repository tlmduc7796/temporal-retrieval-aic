![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-required-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

# TRAKE V4: AIC Multimodal Video Retrieval System

An open-source multimodal video retrieval system designed to process natural language queries in the form of chronologically ordered event sequences. Developed for the AI Challenge (AIC) 2026.

## System Architecture

TRAKE V4 addresses the retrieval problem through a three-stage pipeline:

1. **Multimodal Embedding & Retrieval** — Uses SigLIP-2 (`google/siglip2-so400m-patch16-384`) to embed both text and images into a shared 1152-dimensional vector space. Retrieval is performed in parallel via FAISS (in-memory `IndexFlatIP`) across two streams: **Visual** (keyframes) and **Text** (transcripts/summaries).

2. **Candidate Fusion** — Combines multimodal scores using a linear weighting scheme (`0.6 × Visual + 0.4 × Text`), then computes event coverage to filter and rank the top-N most promising candidate videos.

3. **DANTE Temporal DP** — A dynamic programming algorithm responsible for temporal alignment. DANTE enforces a minimum frame gap (`min_frame_gap = 2`) and applies a gap penalty function (`lambda_weight = 0.0002`, `ideal_gap = 15`) to stitch together temporally sparse candidate frames into a coherent, ordered event sequence.


## Installation & Usage

### 1. Environment Setup

Requires Python 3.10+ with GPU/CUDA support.

```bash
git clone https://github.com/tlmduc7796/temporal-retrieval-aic.git
cd temporal-retrieval-aic
pip install -r requirements.txt
```

### 2. Data Preparation

Download the dataset (keyframes, transcripts) along with the embedding files (`merged_embeddings.npy`, `merged_metadata.csv`) and place them in the `data/` directory. Make sure the absolute paths are correctly configured in `src/config.py`.

### 3. Running the System

Update the `USER_QUERY` array in `main.py` to define your event query sequence, then run:

```bash
python main.py
```


## Contact

For questions, issues, or collaboration inquiries, please open an [issue](https://github.com/tlmduc7796/temporal-retrieval-aic/issues) or reach out via:

- **Author**: [Minh Duc] — tranleminhduc7796@gmail.com
- **GitHub**: [@tlmduc7796](https://github.com/tlmduc7796)
