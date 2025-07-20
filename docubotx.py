import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

# 🧠 Set Hugging Face Token (⚠️ never share this publicly)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # ✅ Your actual token

# 🌟 Streamlit UI
st.title("📄 Ask Questions From PDF")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # 1. Load PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # 2. Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = text_splitter.split_documents(documents)

    # 3. Convert to embeddings using HuggingFace model (no OpenAI needed)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(pages, embeddings)

    # 4. Ask question
    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = db.similarity_search(query)

        # 5. Load model (text2text supported task only)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            model_kwargs={"temperature": 0.5, "max_length": 256},
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            task="text2text-generation"  # ✅ required
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)

        st.write("📌 **Answer:**")
        st.success(answer)
