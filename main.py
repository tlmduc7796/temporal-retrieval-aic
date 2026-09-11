import time
# Import các module đã tách
from src.config import CFG
from src.data_manager import DataManager
from src.query_parser import QueryParser
from src.data_loader import MultimodalDataLoader
from src.retrieval import VisualRetrievalIndex, TextRetrievalIndex, MultimodalCandidateRetriever, CandidateFusion
from src.dante import DANTETemporalDP, CandidateFrameGenerator
from src.trake import TRAKE

# 1. Khởi tạo toàn bộ pipeline
dm = DataManager(cfg=CFG)
query_parser = QueryParser()
multimodal_loader = MultimodalDataLoader(dm, cfg=CFG)

visual_index = VisualRetrievalIndex(multimodal_loader, cfg=CFG)
visual_index.build()

text_index = TextRetrievalIndex(multimodal_loader, cfg=CFG)
text_index.build(sorted(list(dm.video_to_indices.keys())))

retriever = MultimodalCandidateRetriever(multimodal_loader, visual_index, text_index, cfg=CFG)
candidate_fusion = CandidateFusion(retriever, cfg=CFG)

dante_dp = DANTETemporalDP()
frame_generator = CandidateFrameGenerator(data_manager=dm)

trake = TRAKE(query_parser, candidate_fusion, dante_dp, frame_generator, multimodal_loader, cfg=CFG)

# 2. Chạy thử truy vấn
USER_QUERY = [
    "Putting quail eggs into a bowl of white flour",
    "Stirring a bowl of yellow egg yolk"
]

print("=" * 90)
print("ĐANG TÌM KIẾM CHO QUERY:")
start_time = time.time()
search_output = trake.search(USER_QUERY)
elapsed_time = time.time() - start_time
results_count = len(search_output.get("results", []))

print(f"⏱️ Hoàn thành tìm kiếm trong {elapsed_time:.2f} giây.")
print(f"🎯 Tìm thấy {results_count} video.")
print("=" * 90)