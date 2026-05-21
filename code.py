from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.prompts import PromptTemplate
import streamlit as st
import pymupdf
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# chat-history
# cross-encoder ranker
# RAGAS

llm= ChatGroq(
    model= "llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

model= llm
parser= StrOutputParser()
st.header('Research Tool')

def gen_list(x):
    return x.content.split('\n')

@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

@st.cache_resource
def process_pdfs(file_tuple):
    embeddings = load_embeddings()
    all_docs = []
    
    for file in file_tuple:
        pdf_doc = pymupdf.open(stream=file.read(), filetype="pdf")
        for page in pdf_doc:
            normal_text = page.get_text()
            image_list = page.get_images()

            if image_list:
                # Page has images — run OCR to catch text inside them
                tp = page.get_textpage_ocr()
                ocr_text = page.get_text(textpage=tp)
                # Merge both — normal text is clean, OCR catches image text
                text = ocr_text
            else:
                text = normal_text
            
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"source": f"{file.name} — Page {page.number + 1}"}
                ))
    
    index = faiss.IndexFlatL2(384)
    vs = FAISS(embedding_function=embeddings, 
               index=index,
               docstore=InMemoryDocstore(), 
               index_to_docstore_id={}
               )
    vs.add_documents(all_docs)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    return vs,bm25_retriever


embeddings= load_embeddings()

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        vector_store,bm25_retriever = process_pdfs(tuple(uploaded_files))

    bm25_retriever.k=2
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever= EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # query handling
    length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

    query= st.chat_input("Ask something")

    if query:

        prompt_to_multi= PromptTemplate(
            template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to helpthe user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}""",
            input_variables=['question']
            
        )

        queries= prompt_to_multi | model | RunnableLambda(gen_list)
        answer= queries.invoke({"question": query}) #list of queries

        all_reldocs= []
        for q in answer:
            doc_temp= retriever.invoke(q)
            all_reldocs.extend(doc_temp)

        results = list({doc.page_content: doc for doc in all_reldocs}.values())            

        texts =[]
        for doc in results:
            texts.append(doc.page_content)

        combined_text= ' '.join(texts)

        template= PromptTemplate(
            template = """You are a highly accurate research assistant.Use ONLY the information provided in the context below to answer the user's question.Guidelines:- Stay strictly relevant to the provided context.- Do not make up information.- If the answer is not present in the context, clearly say:  "The provided documents do not contain enough information to answer this question."- Provide a clear and well-structured explanation.- Keep the response length according to:  {length_input}Context:{text}Question:{query}Answer:""",
            input_variables=['text','query','length_input']
        )

        prompt= template.invoke({
            'text': combined_text,
            'query': query,
            'length_input': length_input 
        })

        result= model.invoke(prompt)
        st.write(result.content)
        with st.expander("Sources"):
            for doc in results:
                st.write(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")

else:
    st.info("Upload a PDF to get started")