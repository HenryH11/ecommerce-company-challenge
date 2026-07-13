import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_agent(vector_db):
    """
    Une el recuperador de FAISS con Gemini de forma segura y robusta.
    """
    # Configuramos el retriever para traer los 3 fragmentos más relevantes
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Extraemos la clave de forma segura desde la memoria del sistema (.env)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # ACTUALIZACIÓN: Inicializar usando gemini-2.5-flash y el parámetro google_api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    
    system_prompt = (
        "Eres el asistente virtual inteligente de nuestra compañía de eCommerce.\n"
        "Tu objetivo es responder de forma clara y directa a las dudas de los usuarios.\n"
        "Utiliza única y exclusivamente el contexto proporcionado (extraído de nuestras políticas oficiales).\n"
        "Si la respuesta no se encuentra en el contexto, indica amablemente que no dispones de ella.\n\n"
        "Contexto disponible:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)