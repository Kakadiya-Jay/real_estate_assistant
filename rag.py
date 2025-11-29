import os
import shutil
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path("./resources/vector_store/")
COLLECTION_NAME = "real-estate-research"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            max_tokens=512,
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
            collection_metadata={"hnsw:batch_size": 10},
        )


def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initializing Components"
    initialize_components()

    yield "reset old data..."
    # Safely clear previous data
    try:
        vector_store.reset_collection()
    except:
        pass

    yield f"Scraping {len(urls)} URLs (Text Mode)..."
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Splitting documents..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(data)

    yield f"Embedding {len(docs)} document chunks..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅\nProcessing Complete. Ready for questions!"


def generate_answer(question):
    if not vector_store:
        raise RuntimeError("Components not initialized. Please process URLs first.")

    system_prompt = (
        "You are a Real Estate research assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, state that you don't know."
        "\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    result = rag_chain.invoke({"input": question})

    # Extract Sources
    context_docs = result.get("context", [])
    unique_sources = {doc.metadata.get("source", "Unknown") for doc in context_docs}
    sources = "\n".join(f"- {s}" for s in sorted(unique_sources))

    return result["answer"], sources


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2025/11/29/silver-hit-record-highs-in-2025-and-still-has-further-to-run.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]
    for status in process_urls(urls):
        print(status)
    ans, src = generate_answer("What is the outlook for silver?")
    print(ans)
