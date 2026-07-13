def ask_question(chain, question):
    """
    Controlador que invoca la cadena RAG pasando la consulta
    bajo la estructura nativa de la cadena clásica.
    """
    response = chain.invoke(
        {"query": question}
    )
    return response