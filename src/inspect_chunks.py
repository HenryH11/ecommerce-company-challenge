from src.loaders.pdf_loader import load_documents
from src.text_splitter import split_documents

documents = load_documents("docs")
chunks = split_documents(documents)

for i, chunk in enumerate(chunks[:5]):
    print("="*50)
    print(f"Chunk {i+1}")
    print(chunk.metadata)
    print()
    print(chunk.page_content)