from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingModel:
    def __init__(self):
        self.model = None
        self.model_name = 'keepitreal/vietnamese-sbert'
        
    def load_model(self):
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_embedding(self, text: Union[str, List[str]]) -> List[List[float]]:
        if self.model is None:
            self.load_model()
            
        if isinstance(text, str):
            text = [text]
            
        try:
            embeddings = self.model.encode(text)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

embedding_model = EmbeddingModel()