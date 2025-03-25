from backend.chroma_index import query_chroma

query = "Was ist NCP Secure Enterprise Management?"
results = query_chroma(query)

print("🔍 Suchanfrage:", query)
print("🔍 Gefundene Ergebnisse:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result}")
