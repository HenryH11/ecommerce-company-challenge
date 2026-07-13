import os
from dotenv import load_dotenv

# 1. CARGA OBLIGATORIA DEL ENTORNO ANTES QUE CUALQUIER OTRA IMPORTACIÓN
load_dotenv()

# 2. IMPORTACIONES DE MÓDULOS DEL PROYECTO
from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings.embeddings_config import get_embeddings
from src.embeddings.vector_store import create_vector_store
from src.rag.rag_agent import get_rag_agent

def main():
    print("=== 🚀 eCOMMERCE COMPANY CHALLENGE - DÍA 4 ===")

    # 1. Carga y fragmentación (Días 2 y 3)
    documents = load_documents("docs")
    chunks = split_documents(documents)
    
    print(f"PDF pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}\n")
    
    # 2. Embeddings e Indexación FAISS (Día 4)
    embeddings = get_embeddings()
    vector_db = create_vector_store(chunks, embeddings)
    
    # 3. Construcción del Agente RAG
    rag_chain = get_rag_agent(vector_db)
    
    # 4. Consulta de Prueba (Políticas del eCommerce)
    query = "How can I return a product?"
    print(f"\nConsultando al agente: '{query}'")
    
    response = rag_chain.invoke({"input": query})
    
    # 5. Mostrar la respuesta en consola
    print("\n" + "="*60)
    print("🤖 RESPUESTA DEL AGENTE:")
    print("="*60)
    print(response["answer"])
    print("="*60)
    
    # 6. Mostrar Fuentes y Referencias (Requisito del Entregable)
    print("\n📄 FUENTES DE REFERENCIA:")
    print("-" * 60)
    seen_sources = set()
    for doc in response["context"]:
        source_file = os.path.basename(doc.metadata.get('source', 'Desconocido.pdf'))
        # Se suma 1 porque la indexación interna de páginas inicia en 0
        page_num = doc.metadata.get('page', 0) + 1
        
        source_id = f"{source_file}_p{page_num}"
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            print(f"Fuente: {source_file}")
            print(f"Página: {page_num}")
            print("-" * 60)

if __name__ == "__main__":
    main()