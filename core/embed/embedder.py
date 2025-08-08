from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBED_MODEL


class Embedder:
    def __init__(self):
        pass

    def embed(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        return embeddings
