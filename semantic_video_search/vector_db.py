import faiss
import numpy as np
import pickle
import os

class VectorIndex:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Similarity if normalized)
        self.metadata = [] # List to store metadata (e.g., filepath, timestamp) corresponding to vectors

    def add_vectors(self, vectors: np.ndarray, new_metadata: list):
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")
        
        # Faiss expects float32
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.metadata.extend(new_metadata)

    def search(self, query_vector: np.ndarray, k: int = 5):
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dimension}, got {query_vector.shape[1]}")

        query_vector = query_vector.astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "score": float(distances[0][i])
                })
        
        return results

    def save_index(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self, folder_path: str):
        index_path = os.path.join(folder_path, "index.faiss")
        metadata_path = os.path.join(folder_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False
