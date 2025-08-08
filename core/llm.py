from transformers import pipeline
from core.embed.embedder import Embedder
from core.vectorstore.vectorstore import VectorStore
from config import VECTORSTORE_PATH


class RAGQA:
    def __init__(self):
        self.embedder = Embedder()
        self.vdb = VectorStore(VECTORSTORE_PATH, self.embedder)
        self.llm = pipeline(
            "text-generation",
            model="Qwen/Qwen2-0.5B-Instruct",
            device_map="cpu"
        )

    def ask(self, question):
        docs = self.vdb.similarity_search(query=question)
        # print("\n===== 檢索到的內容 =====\n")
        # for i, doc in enumerate(docs):
        #     print(f"[{i+1}] {doc.page_content}\n")

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
            你是一個根據文件內容回答問題的助理。

            以下是檢索到的相關內容：{context}
            問題：{question}

            請根據上述內容，給出完整且有條理的答案，如果文件中沒有提到，就說「文件中沒有提到」。
        """

        result = self.llm(prompt, max_length=1024, do_sample=True)
        output = result[0]["generated_text"]

        # 去掉 prompt 部分，只保留生成內容
        answer = output[len(prompt):].strip()
        return answer
