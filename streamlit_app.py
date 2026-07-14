import os
import time
import logging
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# 1. CARGA OBLIGATORIA DEL ENTORNO ANTES DE LAS IMPORTACIONES TECH
load_dotenv()

from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings.embeddings_config import get_embeddings
from src.embeddings.vector_store import create_vector_store
from src.rag.rag_chain import create_rag_chain

# --- PASO 10: REGISTRO ENRIQUECIDO DE LOGS (Sección 8 del Challenge) ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/rag_execution.txt", # Sincronizado con tu archivo real de logs
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

def registrar_ejecucion_log(query, answer, fuentes, tiempo_ejecucion):
    """Registra de manera estructurada los eventos de auditoría técnica"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = (
        f"Date: {timestamp}\n"
        f"Question:\n{query}\n"
        f"Response:\n{answer}\n"
        f"Sources:\n{fuentes if fuentes else 'None'}\n"
        f"Time:\n{tiempo_ejecucion:.1f} sec\n"
        f"{'='*40}\n"
    )
    logging.info(log_entry)

# --- PASO 11: FAVICON Y CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="eCommerce AI Assistant", page_icon="🤖", layout="centered")

# --- PASO 3 y 4: CREAR SIDEBAR CORPORATIVO Y EJEMPLOS ---
with st.sidebar:
    st.title("About")
    st.write("""
    eCommerce Company Challenge
    AI Corporate Assistant
    
    Powered by:
    • Gemini 2.5 Flash
    • LangChain
    • FAISS
    • Streamlit
    """)
    st.divider()
    st.subheader("Example Questions")
    st.write("""
    • How can I return a product?
    • How long does shipping take?
    • Can I cancel my order?
    • How is my information protected?
    • What payment methods are accepted?
    """)

# --- PASO 2: ENCABEZADO PROFESIONAL ---
st.title("🤖 eCommerce Company Challenge")
st.subheader("Corporate AI Assistant")
st.markdown("""
Welcome!

This AI assistant allows employees to search information from internal corporate documentation.

Supported documents:
• Privacy Policy
• Refund & Returns Policy
• Shipping & Delivery Guide
• Terms & Conditions
• Frequently Asked Questions
""")

# --- PASO 5: HISTORIAL DE CONVERSACIÓN (Inicialización de Estado) ---
if "history" not in st.session_state:
    st.session_state.history = []

# Inicialización optimizada del pipeline RAG con caché de Streamlit
@st.cache_resource
def inicializar_pipeline_rag():
    try:
        if not os.path.exists("docs") or not os.listdir("docs"):
            st.error("❌ Error Crítico: La carpeta 'docs/' está vacía o no existe.")
            return None
        
        # Flujo modular de tu árbol src
        documents = load_documents("docs")
        chunks = split_documents(documents)
        embeddings = get_embeddings()
        vector_db = create_vector_store(chunks, embeddings)
        return create_rag_chain(vector_db)
    except Exception as e:
        st.error(f"❌ Error durante la inicialización del pipeline: {e}")
        return None

with st.spinner("Initializing knowledge base and loading AI models..."):
    rag_chain = inicializar_pipeline_rag()

if rag_chain is None:
    st.stop()

# Caja de entrada de texto para la consulta del colaborador
question = st.text_input("Ask a question", key="input_box")
buscar_button = st.button("Ask")

if buscar_button:
    # --- PASO 6: VALIDACIÓN DE PREGUNTAS VACÍAS ---
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # --- PASO 7: MANEJO DE ERRORES EN LA EJECUCIÓN ---
        try:
            start_time = time.time()
            
            # Invocar cadena clásica de recuperación
            response = rag_chain.invoke({"query": question})
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Adaptación para leer de tu motor de respuesta clásico
            result_text = response.get("answer", response.get("result", ""))
            source_docs = response.get("context", response.get("source_documents", []))
            
            # --- PASO 8: MENSAJE CUANDO NO ENCUENTRE RESPUESTA ---
            # Si Gemini indica que no posee la información en sus manuales oficiales
            info_not_found = (
                "don't know" in result_text.lower() or 
                "no dispongo de esa información" in result_text.lower() or
                "lo siento" in result_text.lower()
            )
            
            # Desplegar respuesta en formato de éxito visual
            st.success("Answer")
            st.write(result_text)
            
            # Procesar y limpiar metadatos de las fuentes extraídas de FAISS
            fuentes_log = []
            if not info_not_found and source_docs:
                # --- PASO 9: MEJORAR VISUALIZACIÓN DE FUENTES ---
                st.divider()
                st.subheader("Sources")
                seen_sources = set()
                
                for doc in source_docs:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown_Document.pdf'))
                    page_num = doc.metadata.get('page', 0) + 1
                    source_id = f"{source_file}_p{page_num}"
                    
                    if source_id not in seen_sources:
                        seen_sources.add(source_id)
                        fuentes_log.append(f"{source_file} page {page_num}")
                        
                        st.markdown(f"""
                        **Document**
                        {source_file}
                        **Page**
                        {page_num}
                        ---
                        """)
            elif info_not_found:
                st.info("The requested information was not found in the available documentation.")
            
            # --- PASO 5: HISTORIAL DE CONVERSACIÓN (Guardar Registro) ---
            st.session_state.history.append({
                "question": question,
                "answer": result_text
            })
            
            # --- PASO 10: REGISTRAR PREGUNTAS EN LOGS ---
            registrar_en_log_str = ", ".join(fuentes_log) if fuentes_log else "None"
            registrar_ejecucion_log(question, result_text, registrar_en_log_str, execution_time)
            
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# --- PASO 5: MOSTRAR HISTORIAL DE CONVERSACIÓN REVERTIDO ---
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")
    for item in reversed(st.session_state.history):
        st.write(f"**Question:** {item['question']}")
        st.write(f"**Answer:** {item['answer']}")
        st.markdown("---")