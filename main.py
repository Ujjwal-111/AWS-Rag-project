import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()


# 2. Load PDF, create chunks, and index them
@st.cache_resource
def create_rag():
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "document_loader",
        "STATS.pdf"
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings()

    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)

    return vectorstore, len(docs), len(chunks)


# 3. Initialize Groq
@st.cache_resource
def create_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
    )


# 4. Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Statistical Techniques. "
        "Answer the question accurately based ONLY on the provided context. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])


# 5. Streamlit interface
st.set_page_config(page_title="Statistics RAG Assistant", page_icon="📘")

st.title("📘 Statistical Techniques RAG Assistant")
st.write("Ask questions about the content of your PDF.")

with st.spinner("Loading PDF and preparing the RAG system..."):
    vectorstore, page_count, chunk_count = create_rag()
    llm = create_llm()

st.success(f"PDF ready: {page_count} pages, {chunk_count} chunks indexed.")

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. What are the normal equations for least squares?"
)

if st.button("Get Answer", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the PDF and generating an answer..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": context,
            "question": question,
        })

    st.subheader("Answer")
    st.write(response)

    with st.expander("View retrieved context"):
        for i, doc in enumerate(retrieved_docs, start=1):
            page_num = doc.metadata.get("page", "Unknown")
            st.markdown(f"**Chunk {i} — PDF page {page_num + 1 if isinstance(page_num, int) else page_num}**")
            st.write(doc.page_content)