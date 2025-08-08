from core.notion.notion_syncer import NotionSyncer
from core.llm import RAGQA
from config import NOTION_API_KEY, NOTION_DATABASE_ID


def main():
    syncer = NotionSyncer(notion_api_key=NOTION_API_KEY, database_id=NOTION_DATABASE_ID)
    syncer.sync()  # 檢查更新並建立向量庫

    rag = RAGQA()

    while True:
        question = input("請輸入你的問題（輸入 'exit' 離開）：")

        if question.lower() == "exit":
            print("再見～")
            break

        answer = rag.ask(question)
        print(f"回答：{answer}\n")


if __name__ == "__main__":
    main()
