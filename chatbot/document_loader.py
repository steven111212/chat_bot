from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document




class DocumentLoader:

    @staticmethod
    def load_pdf(route):

        raw_pdf_elements = partition_pdf(
            filename= route,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf= False,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure= False,
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            # Chunking params to aggregate text blocks
            # Attempt to create a new chunk 3800 chars
            # Attempt to keep chunks > 2000 chars
            # Hard max on chunks
            max_characters=800,
            new_after_n_chars=600,
            combine_text_under_n_chars=300,
        )
        all_splits = []
        for e in raw_pdf_elements:
            doc = Document(
            page_content = e.text,
            metadata = {
                'type': 'text'
            }
        ) 
            all_splits.append(doc)
        return all_splits

    @staticmethod
    def load_txt(route):
        all_splits = []
        with open(route, 'r', encoding='utf-8') as file:
            text = file.read()
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(text)
        for e in texts:
            doc = Document(
            page_content = e,
            metadata = {
                'type': 'text'
            }
        ) 
            all_splits.append(doc)
        return all_splits

    @staticmethod
    def load_web(url):
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        return all_splits

