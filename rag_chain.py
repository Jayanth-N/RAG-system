# rag_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import settings

def create_rag_chain(vector_store, llm_model: str, k: int):
    """Creates the RAG chain for querying."""
    llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=settings.GEMINI_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.
    Provide the answer and then list the sources with their page number.

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever, document_chain)