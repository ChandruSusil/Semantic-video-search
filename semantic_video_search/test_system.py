import os
import shutil
import numpy as np
from processor import FrameExtractor
from embeddings import CLIPEmbedder
from vector_db import VectorIndex
# Mocking cv2 and PIL for smoke test if needed, but we want real test.

def test_system():
    print("Testing System Components...")
    
    # 1. Setup Dummy Data
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
        
    print("Step 1: Frame Extraction (Mocked for speed if no video)")
    # We can't easily create a dummy video, so we will skip video extraction in this purely automated quick test
    # unless we use a known small video. 
    # For now, let's test component instantiation to ensure imports and libraries work.
    
    try:
        extractor = FrameExtractor()
        print("FrameExtractor initialized.")
    except Exception as e:
        print(f"FAILED to init FrameExtractor: {e}")
        
    try:
        embedder = CLIPEmbedder()
        print("CLIPEmbedder initialized (Model loaded).")
    except Exception as e:
        print(f"FAILED to init CLIPEmbedder: {e}")
        return

    try:
        index = VectorIndex()
        print("VectorIndex initialized.")
    except Exception as e:
        print(f"FAILED to init VectorIndex: {e}")
        return
        
    # 2. Test Embedding and Indexing flow with Dummy Image
    print("Step 2: Embedding Flow")
    from PIL import Image
    dummy_img_path = "test_data/dummy.jpg"
    img = Image.new('RGB', (224, 224), color = 'red')
    img.save(dummy_img_path)
    
    try:
        vec = embedder.get_image_embedding(dummy_img_path)
        print(f"Image embedded. Shape: {vec.shape}")
        
        text_vec = embedder.get_text_embedding("A red image")
        print(f"Text embedded. Shape: {text_vec.shape}")
        
        index.add_vectors(vec, [("dummy_path", 1.0)])
        print("Vector added to index.")
        
        results = index.search(text_vec, k=1)
        print("Search performed.")
        print(f"Result: {results}")
        
        assert len(results) > 0
        print("Smoke Test PASSED!")
        
    except Exception as e:
        print(f"Flow FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
