import os
import time
from datetime import datetime
from dotenv import load_dotenv

# 1. CARGA OBLIGATORIA DEL ENTORNO ANTES QUE CUALQUIER OTRA IMPORTACIÓN
load_dotenv()

# 2. IMPORTACIONES DE MÓDULOS REALES DEL PROYECTO
from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings.embeddings_config import get_embeddings
from src.embeddings.vector_store import create_vector_store
from src.rag.rag_chain import create_rag_chain
from src.chat import ask_question

def main():
    print("=== 🚀 eCOMMERCE COMPANY CHALLENGE - INICIANDO DÍA 5 ===")

    # 1. Ingesta y Fragmentación (Días 2 y 3)
    documents = load_documents("docs")
    chunks = split_documents(documents)
    
    print(f"PDF pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}\n")
    
    # 2. Embeddings e Indexación FAISS (Día 4)
    embeddings = get_embeddings()
    vector_db = create_vector_store(chunks, embeddings)
    
    # 3. Construcción de la cadena RAG (Día 5)
    print("Ensamblando cadena RAG (RetrievalQA)...")
    rag = create_rag_chain(vector_db)
    
    # -------------------------------------------------------------
    # PASO 7: Lista de Pruebas Automatizadas
    # -------------------------------------------------------------
    questions = [
        "How can I request a refund?",
        "How long does shipping take?",
        "How is my personal data protected?",
        "Can I cancel an order?",
        "How do I contact customer support?"
    ]
    
    print(f"\nIniciando batería de pruebas automáticas ({len(questions)} consultas)...")
    
    total_start_time = time.time()
    respuestas_exitosas = 0
    documentos_recuperados_totales = 0
    
    for question in questions:
        print("\n" + "="*70)
        print(f"QUESTION: {question}")
        print("="*70)
        
        # Invocación del controlador pasando por Gemini y FAISS
        response = ask_question(rag, question)
        
        # Conteo para métricas
        if response and "result" in response:
            respuestas_exitosas += 1
            documentos_recuperados_totales += len(response.get("source_documents", []))
        
        print("\nAnswer")
        print()
        print(response["result"])
        print()
        
        print("Sources")
        print()
        fuentes_vistas = set()
        for doc in response.get("source_documents", []):
            nombre_archivo = os.path.basename(doc.metadata.get('source', 'Documento.pdf'))
            numero_pagina = doc.metadata.get('page', 0) + 1
            id_fuente = f"{nombre_archivo}_p{numero_pagina}"
            
            if id_fuente not in fuentes_vistas:
                fuentes_vistas.add(id_fuente)
                print(nombre_archivo)
                print(f"page {numero_pagina}")
                print()
        print("-" * 70)
        
    total_end_time = time.time()
    tiempo_promedio = (total_end_time - total_start_time) / len(questions)

    # -------------------------------------------------------------
    # PASO 9: Registro Automatizado de Evidencias (Logs)
    # -------------------------------------------------------------
    log_path = "logs/rag_execution.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions Tested: {len(questions)}\n")
        f.write(f"Successful Answers: {respuestas_exitosas}\n")
        f.write(f"Source Documents Retrieved: {documentos_recuperados_totales}\n")
        f.write(f"Average Response Time: {tiempo_promedio:.2f} seconds\n")
        
    print(f"\n✔ ¡Batería de pruebas completada con éxito!")
    print(f"✔ Evidencias registradas de forma automática en: '{log_path}'")

if __name__ == "__main__":
    main()