from backend.chroma_index import load_pdfs_and_index

# Test wheter documents are loaded
docs = load_pdfs_and_index()

if docs:
    collection = docs.get()
    print(f"✅ {len(collection['ids'])} documents in chroma loaeded")
else:
    print("❌ No documents loaded!")
