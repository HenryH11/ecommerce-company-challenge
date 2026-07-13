import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

# Cargamos el entorno para asegurar que la API Key esté disponible
load_dotenv()

def create_rag_chain(vector_db):
    """
    Construye y retorna la cadena RetrievalQA (Agente RAG).
    Extrae los 3 fragmentos más relevantes de FAISS y los envía a Gemini.
    """
    # Recuperamos de forma segura la API Key desde la memoria local
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Inicializamos el Modelo de Lenguaje oficial moderno de Google
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )
    
    # Ensamblamos la cadena de recuperación tal como lo pide el challenge
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True  # Habilita el retorno de metadatos (páginas y archivos)
    )
    
    return qa_chain