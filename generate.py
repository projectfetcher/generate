import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# LOGGING
log_file = "logs.txt"



log_handle = open(log_file, "a", encoding="utf-8")
def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("AI Generation Started – Flan-T5 + MiniLM (CPU)")

# INPUT
site_url   = os.getenv("SITE_URL", "")
topics_str = os.getenv("TOPICS", "")
site_desc  = os.getenv("SITE_DESC", "a general blog")

topics = [t.strip() for t in topics_str.split(",") if t.strip()][:15]
if len(topics) < 15:
    topics += ["General Tips"] * (15 - len(topics))

log(f"Site: {site_url} | Topics: {topics}")

# MODEL & TOKENIZER
device = torch.device("cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)
log("Flan-T5-Large loaded")

# SENTENCE TRANSFORMER
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
log("MiniLM loaded")

def generate_article(title):
    log(f"Generating: {title}")
    prompt = f"Write a 400-600 word blog post about \"{title}\" for {site_desc}. Include intro, tips, examples, conclusion."
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=600, temperature=0.7, do_sample=True)
    article = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    log(f"Done: {title} ({len(article.split())} words)")
    return article

# GENERATE
articles = []
progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

for i, title in enumerate(topics, 1):
    progress.update({"current": title, "done": i-1, "percent": int((i-1)/len(topics)*100)})
    with open("progress.json", "w") as f: json.dump(progress, f)
    articles.append({"title": title, "content": generate_article(title)})
    progress.update({"done": i, "percent": int(i/len(topics)*100)})
    with open("progress.json", "w") as f: json.dump(progress, f)

# SAVE
with open("articles.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)
progress["percent"] = 100
with open("progress.json", "w") as f: json.dump(progress, f)
log("All done!")
log_handle.close()
