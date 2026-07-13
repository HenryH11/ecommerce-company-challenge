from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder="docs"):
    documents = []
    # Verificar si la carpeta existe para evitar errores
    if not os.path.exists(folder):
        print(f"Error: La carpeta '{folder}' no existe.")
        return documents
        
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            documents.extend(loader.load())
    return documents