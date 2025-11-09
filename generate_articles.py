#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import random

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("AI Blog Generator Started – Flan-T5-Large + MiniLM (CPU)")

# ---------- HARDCODED SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com: Your gateway to top jobs in Mauritius. "
    "Explore vacancies in tourism, finance, IT, and more from leading employers. "
    "Post resumes, apply easily, and advance your career on the island."
)
log(f"Site Description (hardcoded): {site_desc}")

# ---------- MODEL & TOKENIZER ----------
log("Loading google/flan-t5-large ...")
device = torch.device("cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("Flan-T5-Large loaded on CPU")

# ---------- SENTENCE TRANSFORMER ----------
log("Loading all-MiniLM-L6-v2 for title uniqueness...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer loaded")

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles: int = 15):
    log(f"Generating {num_titles} unique blog titles for: {site_desc}")
    prompt = (
        f"Generate {num_titles} diverse, engaging, and unique blog post titles "
        f"for a website described as '{site_desc}'. "
        f"Each title should be 6-12 words, SEO-friendly, and cover different angles. "
        f"Return only a numbered list. No duplicates. No explanations."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2
        )

    raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    log(f"Raw title output:\n{raw}")

    # Parse titles
    titles = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and any(c.isalnum() for c in line):
            clean = line.split(".", 1)[-1].split(":", 1)[-1].strip(' "\'-')
            if 6 <= len(clean.split()) <= 14:
                titles.append(clean)

    # Deduplicate using cosine similarity
    unique_titles = []
    embeddings = []

    for title in titles:
        if len(unique_titles) >= num_titles:
            break

        emb = similarity_model.encode(title, convert_to_tensor=True)
        if not embeddings:
            unique_titles.append(title)
            embeddings.append(emb)
            continue

        # Compare against all existing embeddings
        sims = [util.cos_sim(emb, e).item() for e in embeddings]
        if not any(s > 0.85 for s in sims):
            unique_titles.append(title)
            embeddings.append(emb)

    # Fill remaining with variations
    while len(unique_titles) < num_titles and unique_titles:
        base = random.choice(unique_titles)
        variation_prompt = (
            f"Create a fresh, unique blog title variation of: \"{base}\". "
            f"Keep same topic area for '{site_desc}' but change wording completely. "
            f"6-12 words. No quotes."
        )
        inputs = tokenizer(variation_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, temperature=1.0, do_sample=True)
        new_title = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if 6 <= len(new_title.split()) <= 14:
            emb = similarity_model.encode(new_title, convert_to_tensor=True)
            sims = [util.cos_sim(emb, e).item() for e in embeddings]
            if not any(s > 0.85 for s in sims):
                unique_titles.append(new_title)
                embeddings.append(emb)

    return unique_titles[:num_titles]

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"Generating article: {title}")
    prompt = (
        f"Write a helpful, detailed blog post titled \"{title}\" "
        f"for a site about {site_desc}. "
        f"Include: introduction, 3–5 practical tips with examples, "
        f"real-world scenario, and a strong conclusion. "
        f"Use friendly, expert tone. Minimum 400 words. Natural paragraphs."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.8,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.15,
            min_length=300
        )
    article = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    word_count = len(article.split())
    if word_count < 100:
        log(f"Warning: Article too short ({word_count} words). Regenerating...")
        return generate_article(title)

    log(f"Generated: {title} ({word_count} words)")
    return article

# ---------- MAIN LOOP ----------
try:
    topics = generate_unique_titles(site_desc, num_titles=15)
    log(f"Final {len(topics)} Unique Titles:\n" + "\n".join([f"- {t}" for t in topics]))

    articles = []
    progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}

    log("Starting article generation loop...")
    for i, title in enumerate(topics, 1):
        progress["current"] = title
        progress["done"] = i - 1
        progress["percent"] = int((i - 1) / len(topics) * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

        content = generate_article(title)
        articles.append({"title": title, "content": content})

        # Update progress
        progress["done"] = i
        progress["percent"] = int(i / len(topics) * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    with open(articles_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    log(f"{articles_file} saved with {len(articles)} articles")

    progress["percent"] = 100
    progress["current"] = "Complete"
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    log("progress.json set to 100%")

    log("All articles generated successfully!")
    print("SUCCESS")

except Exception as e:
    log(f"ERROR: {str(e)}")
    print("FAILED")
finally:
    log_handle.close()
