from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embeddings):
    """
    Convierte los fragmentos de texto en vectores usando FAISS
    y guarda el índice en el directorio local 'vectorstore/'.
    """
    print("Generando embeddings e indexando en FAISS local...")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Guardamos localmente el índice de forma persistente en tu carpeta vectorstore/
    vector_db.save_local("vectorstore")
    print("¡Índice FAISS guardado de forma persistente en 'vectorstore/'!")
    return vector_db