import os
import streamlit as st
from dotenv import load_dotenv

# Primero cargamos de forma obligatoria las variables del archivo .env
load_dotenv()

# Paso 4. Importar librerías de tu árbol src real (Reutilizando tu backend validado)
from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings.embeddings_config import get_embeddings
from src.embeddings.vector_store import create_vector_store
from src.rag.rag_chain import create_rag_chain

# Paso 5. Configurar la página (Título de pestaña e ícono oficial)
st.set_page_config(
    page_title="eCommerce Company Challenge",
    page_icon="🛒",
    layout="centered"
)

# Paso 6. Título principal en la interfaz gráfica
st.title("🛒 eCommerce Company Challenge")
st.subheader("Corporate AI Assistant")

# Paso 7. Descripción oficial de la documentación soportada
st.markdown("""
This AI assistant answers questions using the company's internal documentation.

**Supported documents:**
- Privacy Policy
- Refund Policy
- Shipping Guide
- FAQ
- Terms & Conditions
""")
st.markdown("---")

# Paso 8. Cargar el RAG una sola vez (Optimización con Caché)
@st.cache_resource
def load_agent():
    """
    Carga los documentos desde la carpeta 'docs', genera los fragmentos
    e indexa la base vectorial FAISS de forma interna.
    Mantiene el pipeline en memoria para evitar reprocesamiento.
    """
    if not os.path.exists("docs") or not os.listdir("docs"):
        return None
    
    # Orquestación de tu backend verificado usando la carpeta real 'docs'
    documents = load_documents("docs")
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_db = create_vector_store(chunks, embeddings)
    
    # Retorna la cadena RAG con Gemini 2.5 Flash
    return create_rag_chain(vector_db)

# Inicializar el agente una sola vez
with st.spinner("Loading corporate knowledge base..."):
    rag = load_agent()

if rag is None:
    st.error("❌ Error: Directory 'docs/' is empty or missing. Please add your corporate PDFs.")
    st.stop()

    st.markdown("---")

# Paso 9. Caja de texto para recibir la consulta del colaborador
question = st.text_input(
    "Ask a question"
)

# Paso 10. Botón de acción para ejecutar la búsqueda
if st.button("Ask"):
    # Validación de seguridad interna: evitar llamadas si el usuario no escribió nada
    if question.strip() != "":
        
        # Paso 11. Spinner interactivo mientras el agente trabaja en el backend
        with st.spinner("Searching documentation..."):
            
            # Paso 12. Obtener respuesta invocando la cadena RAG con tu variable 'rag'
            response = rag.invoke({"query": question})
            
            # Paso 13. Mostrar respuesta con formato de éxito visual
            st.success("Answer")
            st.write(response["result"])

            # Paso 14. Mostrar fuentes citadas en la consulta
            st.divider()
            st.subheader("Sources")
            
            seen_sources = set()
            for doc in response["source_documents"]:
                # Extraemos de forma segura el nombre del archivo de la ruta completa
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown_Document.pdf'))
                # Sumamos 1 porque la indexación interna de páginas en LangChain inicia en 0
                page_num = doc.metadata.get('page', 0) + 1
                
                source_id = f"{source_file}_p{page_num}"
                if source_id not in seen_sources:
                    seen_sources.add(source_id)
                    st.write(f"📄 {source_file}")
                    st.write(f"Page {page_num}")
                    st.markdown("---")
    else:
        st.warning("Please enter a question.")