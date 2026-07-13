from src.loaders.pdf_loader import load_documents

# Invocamos la función apuntando a tu carpeta 'docs'
documents = load_documents("docs")

print("===================================")
print(f"Documents loaded: {len(documents)}")
print("===================================")
print()

if len(documents) > 0:
    print("--- Contenido de la primera página detectada ---")
    # Accedemos al contenido textual de la primera página cargada
    print(documents[0].page_content)
else:
    print("❌ No se detectaron documentos PDF en la carpeta 'docs/'.")