from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from config import TOKEN_MODEL


class Chunker:
    def __init__(self):
        pass

    def _count_tokens(self, text):
        tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL)

        return len(tokenizer.encode(text))

    def chunk_documents(self, docs, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens  # 用 token 數切割
        )

        return splitter.split_documents(docs)
