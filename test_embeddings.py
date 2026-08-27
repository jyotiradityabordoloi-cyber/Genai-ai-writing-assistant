from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Employees receive 20 days of annual leave."

embedding = model.encode(text)

print("Embedding created!")
print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])