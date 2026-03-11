import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from services.whatsapp_msg import whatsapp_response
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = "app/data/apparel_store_faq.csv"
model = SentenceTransformer("all-MiniLM-L6-v2")

questions = []
answers = []

with open(DATA_PATH, newline="", encoding="utf-8") as f:
  reader = csv.DictReader(f)
  for row in reader:
    questions.append(row["question"])
    answers.append(row["answer"])

question_embed = model.encode(questions)

conversation_history: dict[str, list] = {}

def get_history(sender: str) -> list:
  if sender not in conversation_history:
    conversation_history[sender] = []
  return conversation_history[sender]

def update_history(sender: str, role: str, content: str):
  if sender not in conversation_history:
    conversation_history[sender] = []
  conversation_history[sender].append({
    "role":role,
    "content":content
  })
  if len(conversation_history[sender]) > 10:
    conversation_history[sender] = conversation_history[sender][-10:]
  
def retrieve_contexts(question:str, top_k:int = 3):
  user_embed = model.encode([question])
  similarities = cosine_similarity(user_embed, question_embed)[0]
  best_indices = similarities.argsort()[-top_k:][::-1]
  return [answers[i] for i in best_indices]

def build_contexts(contexts: list[str]) -> str:
  return "\n".join(f"{ctx}" for ctx in contexts)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question:str, contexts_str: str, sender: str):
  history = get_history(sender)


  system_prompt = f"""
  Kamu adalah admin bagian customer service di sebuah toko baju bernama ApparelBC, sebagai customer service, kamu menjawab dengan ramah dan tidak singkat.

  Aturan: 
  - Awali chat dengan sapaan ramah HANYA untuk pertama kali chat, jika sudah sapa sebelumnya TIDAK perlu sapa lagi, hanya jawab pertanyaan.
  - Beri emoji yang sesuai saat chat untuk memberikan kesan friendly.
  - Jawab HANYA berdasarkan konteks informasi dibawah.
  - Jangan menambah informasi diluar konteks.
  - Parafrase informasi diperbolehkan, tapi tidak boleh sampai mengubah makna atau fakta yang ada.
  - Jika jawaban tidak ditemukan, maka jawab tidak tahu. Dan jelaskan bahwa pertanyaan akan diteruskan dan  dijawab oleh admin manusia.
  - Format jawaban dengan rapi, hanya gunakan 1 * di kanan dan kiri teks untuk meng-highligth kata penting.
  

  Informasi konteks:
  {contexts_str}
  """

  # Build messages with history
  messages = [{"role": "system", "content": system_prompt}]
  for msg in history:
    messages.append({"role": msg["role"], "content": msg["content"]})
 
  messages.append({"role": "user", "content": question})
  
  response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages, # type: ignore
    temperature=0.5
  )

  content = response.choices[0].message.content
  assert content is not None

  # Save history
  update_history(sender, "user", question)
  update_history(sender, "assistant", content)

  print(history)

  return content

def chat_with_rag(question: str, sender: str = "default"):
  contexts = retrieve_contexts(question)
  contexts_str = build_contexts(contexts)
  answer = generate_answer(question, contexts_str, sender)

  return answer
 