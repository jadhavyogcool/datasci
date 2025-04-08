import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

data = {
    'context': [
        "Artificial Intelligence enables machines to think like humans.",
        "Machine Learning is a subset of AI that learns from data.",
        "Deep Learning uses neural networks with many layers.",
        "Natural Language Processing allows computers to understand text.",
        "AI is used in healthcare, finance, and autonomous vehicles."
    ]
}
df = pd.DataFrame(data)

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = embed_model.encode(df['context'].tolist(), convert_to_numpy=True)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

generator = pipeline("text-generation", model="gpt2")

def answer_question(question, top_k=2):
    question_embedding = embed_model.encode([question], convert_to_numpy=True)
    _, I = index.search(question_embedding, k=top_k)
    
    relevant_texts = df.iloc[I[0]]['context'].tolist()
    prompt = "Context: " + " ".join(relevant_texts) + f"\nQuestion: {question}\nAnswer:"

    output = generator(prompt, max_new_tokens=100, do_sample=True)[0]['generated_text']
    return output[len(prompt):].strip()

if __name__ == "__main__":
    print("=== RAG with DataFrame ===")
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        answer = answer_question(user_question)
        print("\nAnswer:", answer)
