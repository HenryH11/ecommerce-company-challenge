from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents

# Cargamos desde tu carpeta real de PDFs 'docs'
documents = load_documents("docs")

# Ejecutamos tu función de fragmentación
chunks = split_documents(documents)

print(f"PDF pages: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print()

if len(chunks) > 0:
    print("--- Contenido del primer chunk ---")
    print(chunks[0].metadata)
else:
    print("❌ No se pudieron generar chunks.")