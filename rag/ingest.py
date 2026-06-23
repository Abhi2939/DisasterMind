import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATASET_DIR = "rag/dataset"
VECTOR_DB_DIR = "rag/vector_db"

DOCS = [
    {"file":"IMD Cyclone.pdf","disaster_type":"Cyclone","source":"IMD"},
    {"file":"IMD Earthquake.pdf","disaster_type":"Earthquake","source":"IMD"},
    {"file":"NDMA Cyclone.pdf","disaster_type":"Cyclone","source":"NDMA"},
    {"file":"NDMA Earthquake.pdf","disaster_type":"Earthquake","source":"NDMA"},
]

#load pdf
def load_pdf_text(filepath:str)->str:

    pages = []

    with pdfplumber.open(filepath) as pdf:
        for i,page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i+1,text))
    
    return pages

def build_documents() -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 120,
        separators= ["\n\n","\n"," "," . "]
    )

    all_docs = []

    for doc_meta in DOCS:
        path = os.path.join(DATASET_DIR,doc_meta["file"])
        if not os.path.exists(path):
            print(f"Path not found at {path}")
            continue

        pages = load_pdf_text(path)
        for page_num,page_text in pages:
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                all_docs.append(
                    Document(
                        page_content=chunk,
                        metadata = {
                            "source":doc_meta["source"],
                            "disaster_type":doc_meta["disaster_type"],
                            "file":doc_meta["file"],
                            "page":page_num,
                        }
                    )
                )
    
    return all_docs

def ingest():
    docs = build_documents()
    print(f"Total chunks to embed: {len(docs)}")

    if not docs:
        print("No documents found — check DATASET_DIR and filenames before continuing.")
        return None

    embeddings = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    print(f"Ingested {len(docs)} chunks into {VECTOR_DB_DIR}")
    return vectordb


if __name__ == "__main__":
    ingest()