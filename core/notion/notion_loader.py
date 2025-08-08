from langchain_community.document_loaders import NotionDBLoader
import requests


class NotionLoader:
    def __init__(self, notion_api_key, database_id):
        self.notion_api_key = notion_api_key
        self.database_id = database_id

    def load_documents(self):
        loader = NotionDBLoader(
            integration_token=self.notion_api_key,
            database_id=self.database_id
        )
        docs = loader.load()

        print(f"共載入 {len(docs)} 筆 Notion 頁面內容")

        for doc in docs:
            notion_id = doc.metadata.get("id")
            meta = self._get_notion_page_metadata(self.notion_api_key, notion_id)
            if meta:
                doc.metadata["last_edited_time"] = meta["last_edited_time"]

        return docs
 
    def _get_notion_page_metadata(self, notion_api_key, notion_id):
        url = f"https://api.notion.com/v1/pages/{notion_id}"
        headers = {
            "Authorization": f"Bearer {notion_api_key}",
            "Notion-Version": "2022-06-28",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"無法取得頁面 {notion_id} metadata: {response.text}")
            return None

        data = response.json()

        return {
            "id": data["id"],
            "last_edited_time": data["last_edited_time"]
        }
