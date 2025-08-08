import os
import json
from config import META_PATH, VECTORSTORE_PATH
from core.embed.chunker import Chunker
from core.embed.embedder import Embedder
from core.notion.notion_loader import NotionLoader
from core.vectorstore.vectorstore import VectorStore


class NotionSyncer:
    def __init__(self, notion_api_key, database_id):
        self.meta_path = META_PATH
        self.loader = NotionLoader(notion_api_key, database_id)
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vdb = VectorStore(VECTORSTORE_PATH, self.embedder)

    def load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                return json.load(f)

        return {}

    def save_meta(self, meta):
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def sync(self):
        docs = self.loader.load_documents()
        meta = self.load_meta()

        new_docs = []
        for doc in docs:
            print("⚙️ DEBUG doc.metadata: ", doc.metadata)

            pid = doc.metadata.get("id")
            last_edit = doc.metadata.get("last_edited_time")
            print(f"pid: {pid}, last_edit: {last_edit}")

            if not pid or not last_edit:
                continue

            if pid not in meta or meta[pid] != last_edit:
                new_docs.append(doc)
                meta[pid] = last_edit

        if not new_docs:
            print("沒有新頁面或異動")

            return

        print(f"偵測到 {len(new_docs)} 筆異動，正在更新")
        chunks = self.chunker.chunk_documents(new_docs)
        self.vdb.load()
        self.vdb.add_and_save(chunks)
        self.save_meta(meta)
        print(f"已新增 {len(chunks)} 個 chunks")
