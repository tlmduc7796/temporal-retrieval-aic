import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from src.config import LOGGER
DANTE_TOP_K_PER_EVENT = 100
DANTE_MIN_FRAME_GAP = 2
DANTE_IDEAL_GAP = 15 #25 10        # Chỉnh tùy thuộc vào khoảng cách (tính bằng frame/giây) giữa các event trong video
DANTE_LAMBDA_WEIGHT = 0.0002 #0.002 0.5 Tăng trọng số phạt để ép các frame phải tuân thủ trình tự thời gian sát nhau

@dataclass
class DanteCandidate:

    frame_idx: int
    score: float
    keyframe_path: str


class DANTETemporalDP:

    """
    Original DANTE-style temporal dynamic programming.

    Given:
        candidates[event_i]

    Find:
        one frame for every event

    subject to:

        frame_i >= frame_(i-1) + min_frame_gap

    Objective:

        path_score =
            previous_score
            + current_similarity
            - gap_penalty
    """
    def optimize(
        self,
        video_id,
        event_candidates
    ):
    
        converted_candidates = []
    
        for event_results in event_candidates:
    
            # Already converted
            if (
                event_results
                and isinstance(
                    event_results[0],
                    DanteCandidate
                )
            ):
                converted_candidates.append(
                    event_results
                )
                continue
    
            event_converted = []
    
            for result in event_results:
    
                # ----------------------------------------------------
                # VisualRetrievalResult
                # ----------------------------------------------------
    
                frame_idx = getattr(
                    result,
                    "frame_idx",
                    None
                )
    
                if frame_idx is None:
                    frame_idx = getattr(
                        result,
                        "frame",
                        None
                    )
    
                # ----------------------------------------------------
                # Similarity score
                # ----------------------------------------------------
    
                score = getattr(
                    result,
                    "similarity",
                    None
                )
    
                if score is None:
                    score = getattr(
                        result,
                        "score",
                        None
                    )
    
                if score is None:
                    score = getattr(
                        result,
                        "retrieval_score",
                        None
                    )
    
                # ----------------------------------------------------
                # Keyframe path
                # ----------------------------------------------------
    
                keyframe_path = getattr(
                    result,
                    "keyframe_path",
                    ""
                )
    
                # ----------------------------------------------------
                # Dictionary fallback
                # ----------------------------------------------------
    
                if isinstance(result, dict):
    
                    frame_idx = result.get(
                        "frame_idx",
                        result.get("frame")
                    )
    
                    score = result.get(
                        "similarity",
                        result.get(
                            "score",
                            result.get(
                                "retrieval_score",
                                0.0
                            )
                        )
                    )
    
                    keyframe_path = result.get(
                        "keyframe_path",
                        ""
                    )
    
                # ----------------------------------------------------
                # Invalid candidate
                # ----------------------------------------------------
    
                if frame_idx is None:
                    continue
    
                if score is None:
                    score = 0.0
    
                event_converted.append(
    
                    DanteCandidate(
    
                        frame_idx=int(
                            frame_idx
                        ),
    
                        score=float(
                            score
                        ),
    
                        keyframe_path=str(
                            keyframe_path
                        )
    
                    )
    
                )
    
            event_converted.sort(
                key=lambda x: x.frame_idx
            )
    
            converted_candidates.append(
                event_converted
            )
    
        # ------------------------------------------------------------
        # Run actual DANTE DP
        # ------------------------------------------------------------
    
        return self.search(
            event_candidates=converted_candidates,
            video_id=video_id
        )
    def __init__(
        self,

        top_k_per_event: int = DANTE_TOP_K_PER_EVENT,

        min_frame_gap: int = DANTE_MIN_FRAME_GAP,

        ideal_gap: int = DANTE_IDEAL_GAP,

        lambda_weight: float = DANTE_LAMBDA_WEIGHT,
    ):

        self.top_k_per_event = top_k_per_event

        self.min_frame_gap = min_frame_gap

        self.ideal_gap = ideal_gap

        self.lambda_weight = lambda_weight

        LOGGER.info(
            "DANTE Temporal DP initialized."
        )


    # ============================================================
    # GAP PENALTY
    # ============================================================

    def gap_penalty(
        self,
        current_frame: int,
        previous_frame: int,
    ) -> float:

        return (
            self.lambda_weight
            * abs(
                (current_frame - previous_frame)
                - self.ideal_gap
            )
        )


    # ============================================================
    # TEMPORAL DP
    # ============================================================

    def search(
        self,
        event_candidates: List[List[DanteCandidate]],
        video_id: str,
    ) -> Optional[Dict]:

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if not event_candidates:

            LOGGER.warning(
                f"DANTE: no candidates for {video_id}"
            )

            return None


        n_events = len(
            event_candidates
        )


        # --------------------------------------------------------
        # Every event must have at least one candidate
        # --------------------------------------------------------

        for i, candidates in enumerate(
            event_candidates
        ):

            if not candidates:

                LOGGER.warning(
                    f"DANTE: Event {i} has no candidates "
                    f"for video {video_id}"
                )

                return None


        # ========================================================
        # DP TABLE
        # ========================================================

        dp = [

            [
                -1e9
                for _ in candidates
            ]

            for candidates in event_candidates

        ]


        # ========================================================
        # TRACE TABLE
        # ========================================================

        trace = [

            [
                -1
                for _ in candidates
            ]

            for candidates in event_candidates

        ]


        # ========================================================
        # INITIAL EVENT
        # ========================================================

        for j, candidate in enumerate(
            event_candidates[0]
        ):

            dp[0][j] = candidate.score


        # ========================================================
        # DP TRANSITION
        # ========================================================

        for i in range(
            1,
            n_events
        ):

            current_candidates = (
                event_candidates[i]
            )

            previous_candidates = (
                event_candidates[i - 1]
            )


            for j, current in enumerate(
                current_candidates
            ):

                best_score = -1e9

                best_k = -1


                # ------------------------------------------------
                # Check every candidate from previous event
                # ------------------------------------------------

                for k, previous in enumerate(
                    previous_candidates
                ):

                    previous_frame = (
                        previous.frame_idx
                    )

                    current_frame = (
                        current.frame_idx
                    )


                    # ------------------------------------------------
                    # TEMPORAL CONSTRAINT
                    # ------------------------------------------------

                    if (
                        previous_frame
                        + self.min_frame_gap
                        > current_frame
                    ):

                        continue


                    # ------------------------------------------------
                    # GAP PENALTY
                    # ------------------------------------------------

                    penalty = self.gap_penalty(

                        current_frame=current_frame,

                        previous_frame=previous_frame,

                    )


                    # ------------------------------------------------
                    # DANTE PATH SCORE
                    # ------------------------------------------------

                    score_path = (

                        dp[i - 1][k]

                        + current.score

                        - penalty

                    )


                    # ------------------------------------------------
                    # KEEP BEST PREVIOUS STATE
                    # ------------------------------------------------

                    if (
                        score_path
                        > best_score
                    ):

                        best_score = (
                            score_path
                        )

                        best_k = k


                # ------------------------------------------------
                # Store transition
                # ------------------------------------------------

                if best_k != -1:

                    dp[i][j] = (
                        best_score
                    )

                    trace[i][j] = (
                        best_k
                    )


        # ========================================================
        # CHECK FINAL STATE
        # ========================================================

        final_scores = dp[-1]


        if not final_scores:

            return None


        # --------------------------------------------------------
        # Best final candidate
        # --------------------------------------------------------

        last_j = int(
            np.argmax(
                final_scores
            )
        )


        best_score = (
            final_scores[last_j]
        )


        # --------------------------------------------------------
        # No valid temporal path
        # --------------------------------------------------------

        if best_score <= -1e8:

            LOGGER.warning(
                f"DANTE: no valid temporal path "
                f"for {video_id}"
            )

            return None


        # ========================================================
        # BACKTRACKING
        # ========================================================

        path = []

        current_j = last_j


        for i in range(
            n_events - 1,
            -1,
            -1
        ):

            if current_j == -1:

                return None


            candidate = (
                event_candidates[i][current_j]
            )


            path.append(
                candidate
            )


            current_j = (
                trace[i][current_j]
            )


        # Reverse path

        path.reverse()


        # ========================================================
        # BUILD RESULT
        # ========================================================

        frame_sequence = [

            candidate.frame_idx

            for candidate in path

        ]


        # --------------------------------------------------------
        # Keyframe paths
        # --------------------------------------------------------

        keyframe_paths = [

            candidate.keyframe_path

            for candidate in path

        ]


        # --------------------------------------------------------
        # Event → frame mapping
        # --------------------------------------------------------

        event_frames = [

            {
                "event_idx": i,

                "frame_idx": candidate.frame_idx,

                "score": float(
                    candidate.score
                ),

                "keyframe_path":
                    candidate.keyframe_path,

            }

            for i, candidate in enumerate(
                path
            )

        ]


        return {

            "video_id": video_id,

            "frame_sequence":
                frame_sequence,

            "keyframe_paths":
                keyframe_paths,

            "event_frames":
                event_frames,

            "score":
                float(best_score),

        }


# ================================================================
# FAISS → DANTE CANDIDATE CONVERSION
# ================================================================

def build_dante_candidates(
    retrieval_results: List,
    top_k: int = DANTE_TOP_K_PER_EVENT,
) -> List[DanteCandidate]:

    """
    Convert Cell 8/9 retrieval output into the format
    required by DANTE.

    Expected retrieval result fields:

        frame_idx
        similarity
        keyframe_path

    The function is deliberately tolerant of dictionaries
    produced by different V3 retrieval stages.
    """

    candidates = []


    for result in retrieval_results[:top_k]:

        # --------------------------------------------------------
        # Frame index
        # --------------------------------------------------------

        if isinstance(
            result,
            dict
        ):

            frame_idx = result.get(
                "frame_idx"
            )

            if frame_idx is None:

                frame_idx = result.get(
                    "frame"
                )


            # ----------------------------------------------------
            # Similarity score
            # ----------------------------------------------------

            score = result.get(
                "similarity"
            )

            if score is None:

                score = result.get(
                    "score"
                )

            if score is None:

                score = result.get(
                    "retrieval_score",
                    0.0
                )


            # ----------------------------------------------------
            # Keyframe path
            # ----------------------------------------------------

            keyframe_path = result.get(
                "keyframe_path",
                ""
            )


        else:

            raise TypeError(
                "Retrieval result must be a dict."
            )


        if frame_idx is None:

            continue


        candidates.append(

            DanteCandidate(

                frame_idx=int(
                    frame_idx
                ),

                score=float(
                    score
                ),

                keyframe_path=str(
                    keyframe_path
                ),

            )

        )


    # ------------------------------------------------------------
    # DANTE sorts candidates temporally
    # ------------------------------------------------------------

    candidates.sort(
        key=lambda x: x.frame_idx
    )


    return candidates





# ================================================================
# CANDIDATE FRAME
# ================================================================

@dataclass
class CandidateFrame:

    video_id: str

    event_id: int

    frame_idx: int

    keyframe_path: str

    # Original visual retrieval similarity
    similarity: float = 0.0

    # Explicit retrieval score
    retrieval_score: float = 0.0

    # DANTE temporal score
    dante_score: float = 0.0


# ================================================================
# CANDIDATE FRAME GENERATOR
# ================================================================

class CandidateFrameGenerator:

    def __init__(self, data_manager=None):

        self.data_manager = data_manager

        LOGGER.info(
            "Candidate Frame Generator initialized."
        )


    # ============================================================
    # GENERATE
    # ============================================================

    def generate(
        self,
        sequence,
        candidate_video
    ) -> List[CandidateFrame]:

        """
        Convert DANTE temporal sequence into candidate frames.

        Parameters
        ----------
        sequence:
            Dictionary returned by DANTETemporalDP.search()

        candidate_video:
            Video ID.

        Returns
        -------
        List[CandidateFrame]
        """

        # --------------------------------------------------------
        # Validate DANTE output
        # --------------------------------------------------------

        if sequence is None:

            return []


        if not isinstance(
            sequence,
            dict
        ):

            raise TypeError(
                "DANTE sequence must be a dict."
            )


        # --------------------------------------------------------
        # Video ID
        # --------------------------------------------------------

        video_id = sequence.get(
            "video_id",
            candidate_video
        )


        # ========================================================
        # PREFERRED SOURCE:
        # event_frames
        # ========================================================

        event_frames = sequence.get(
            "event_frames",
            []
        )


        candidates = []


        # ========================================================
        # CASE 1:
        # DANTE provides event_frames
        # ========================================================

        if event_frames:

            for event in event_frames:

                event_id = int(
                    event.get(
                        "event_idx",
                        0
                    )
                )

                frame_idx = int(
                    event.get(
                        "frame_idx"
                    )
                )

                score = float(
                    event.get(
                        "score",
                        0.0
                    )
                )

                keyframe_path = str(
                    event.get(
                        "keyframe_path",
                        ""
                    )
                )


                # ------------------------------------------------
                # Recover keyframe path if necessary
                # ------------------------------------------------

                if (
                    not keyframe_path
                    and self.data_manager is not None
                ):

                    try:

                        keyframe_path = (
                            self.data_manager
                            .get_keyframe_path(
                                video_id,
                                frame_idx
                            )
                        )

                    except Exception:

                        keyframe_path = ""


                candidates.append(
                
                    CandidateFrame(
                
                        video_id=video_id,
                
                        event_id=event_id,
                
                        frame_idx=frame_idx,
                
                        keyframe_path=keyframe_path,
                
                        similarity=score,
                
                        retrieval_score=score,
                
                        dante_score=score,
                
                    )
                
                )


        # ========================================================
        # CASE 2:
        # fallback to frame_sequence
        # ========================================================

        else:

            frame_sequence = sequence.get(
                "frame_sequence",
                []
            )

            keyframe_paths = sequence.get(
                "keyframe_paths",
                []
            )

            dante_score = float(
                sequence.get(
                    "score",
                    0.0
                )
            )


            for event_id, frame_idx in enumerate(
                frame_sequence
            ):

                keyframe_path = ""

                if (
                    event_id
                    < len(keyframe_paths)
                ):

                    keyframe_path = str(
                        keyframe_paths[event_id]
                    )


                # ------------------------------------------------
                # Recover keyframe path if necessary
                # ------------------------------------------------

                if (
                    not keyframe_path
                    and self.data_manager is not None
                ):

                    try:

                        keyframe_path = (
                            self.data_manager
                            .get_keyframe_path(
                                video_id,
                                int(frame_idx)
                            )
                        )

                    except Exception:

                        keyframe_path = ""


                candidates.append(

                    CandidateFrame(
                
                        video_id=video_id,
                
                        event_id=event_id,
                
                        frame_idx=int(frame_idx),
                
                        keyframe_path=keyframe_path,
                
                        similarity=dante_score,
                
                        retrieval_score=dante_score,
                
                        dante_score=dante_score,
                
                    )
                
                )


        # ========================================================
        # DEBUG
        # ========================================================

        print("=" * 70)
        print("DANTE → CANDIDATE FRAME GENERATOR")
        print("=" * 70)

        print(
            f"Video       : {video_id}"
        )

        print(
            f"Candidates  : {len(candidates)}"
        )


        for candidate in candidates:

            print(
                f"Event={candidate.event_id} | "
                f"Frame={candidate.frame_idx} | "
                f"DANTE={candidate.dante_score:.6f} | "
                f"{candidate.keyframe_path}"
            )


        print("=" * 70)


        return candidates

