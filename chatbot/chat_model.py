from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatModel:
    def __init__(self):
        self.chat = ChatOpenAI(model="gpt-3.5-turbo-1106")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        self.document_chain = create_stuff_documents_chain(self.chat, self.prompt)

    def generate_response(self, messages, context):
        return self.document_chain.invoke({
            "messages": messages,
            "context": context,
        })