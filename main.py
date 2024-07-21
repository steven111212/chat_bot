from chatbot.data_load import Dataload

print("請輸入欲參考的文字或PDF檔案路徑或是參考網頁的網址 : ")
input_information = input()
test = Dataload(input_information)
retriever = test.chunk_embedding()
print("嗨! 我是你的助手Steven，你想問什麼問題呢 : ")
while True:
    input_text = input()
    test.chat(input_text)