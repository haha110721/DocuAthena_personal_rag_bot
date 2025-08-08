from langchain_community.vectorstores import FAISS
import os


class VectorStore:
    def __init__(self, path, embedder):
        self.path = path
        self.embedder = embedder.embed()
        self.store = None

    def load(self):
        if os.path.exists(self.path):
            self.store = FAISS.load_local(self.path, self.embedder, allow_dangerous_deserialization=True)
            print(f"向量資料庫已從 {self.path} 載入")
        else:
            print("尚未建立向量資料庫，將於首次加入資料時建立")

        return self.store

    def add_and_save(self, docs):
        if self.store:
            self.store.add_documents(docs)
        else:
            self.store = FAISS.from_documents(docs, self.embedder)

        self.store.save_local(self.path)
        print(f"向量資料庫已儲存到 {self.path}/")

    def similarity_search(self, query, k=3):
        if not self.store:
            self.load()

        results = self.store.similarity_search(query, k=k)

        return results
