import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DATASET_DIR = "rag/dataset"
VECTOR_DB = "rag/vector_db"

DOCS = {
    "file":"IMD Cyclone.pdf","disaster_type":"Cyclone","source":"IMD",
    "file":"IMD Earthquake.pdf","disaster_type":"Earthquake","source":"IMD",
    "file":"NDMA Cyclone.pdf","disaster_type":"Cyclone","source":"NDMA",
    "file":"NDMA Earthquake.pdf","disaster_type":"Earthquake","source":"NDMA"
}

#load pdf
def load_pdf(filepath:str)->str:

    pages = []

    with pdfplumber.open(filepath) as pdf:
        for i,page in enumerate(pdf.pages):
            text = page.extrcat_text()
            if text:
                pages.append((i+1,text))
    
    return pages

def build_document() -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 120,
        separators= ["\n \n","\n"," "," . "]
    )

    all_docs = []

    for doc_meta in DOCS:
        path = os.join(DATASET_DIR,doc_meta["file"])
        if not os.path.exists(path):
            print(f"Path not found at {path}")
            continue

        


