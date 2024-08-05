from chatbot.document_loader import DocumentLoader
from chatbot.vector_store import VectorStore
from chatbot.chat_model import ChatModel
from langchain_community.chat_message_histories import ChatMessageHistory


class Chatbot:
    def __init__(self, route):
        self.route = route
        self.retriever = self.setup_retriever()
        self.chat_model = ChatModel()
        self.chat_history = ChatMessageHistory()

    def setup_retriever(self):
        # 根據路徑類型選擇適當的加載器
        if self.route.endswith('.pdf'):
            documents = DocumentLoader.load_pdf(self.route)
        elif self.route.endswith('.txt'):
            documents = DocumentLoader.load_txt(self.route)
        elif self.route.startswith('http'):
            documents = DocumentLoader.load_web(self.route)
        else:
            raise ValueError("Unsupported document type")
        
        # 創建向量存儲
        vector_store = VectorStore.create_from_documents(documents)
        return vector_store.as_retriever(k=4)

    def chat(self, input_text):
        docs = self.retriever.invoke(input_text)
        self.chat_history.add_user_message(input_text)
        response = self.chat_model.generate_response(self.chat_history.messages, docs)
        print(response)
        self.chat_history.add_ai_message(response)
        self.trim_chat_history()

    def trim_chat_history(self):
        # 保留最後4條消息
        if len(self.chat_history.messages) > 4:
            self.chat_history.messages = self.chat_history.messages[-4:]

# 使用示例
if __name__ == "__main__":

    print("請輸入欲參考的文字或PDF檔案路徑或是參考網頁的網址 : ")
    input_information = input()
    chatbot = Chatbot(input_information)
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        chatbot.chat(user_input)