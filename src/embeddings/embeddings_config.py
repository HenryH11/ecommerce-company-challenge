from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Retorna un modelo de embeddings que corre 100% en local y gratis
    sin necesidad de usar API Keys externas.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )