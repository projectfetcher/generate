#!/usr/bin/env python3
import os
import json
import time
import torch
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
LOG_FILE = "logs.txt"
PROGRESS_FILE = "progress.json"
ARTICLES_FILE = "articles.json"

# Verbose mode
VERBOSE = True

def vlog(msg):
    """Verbose logger"""
    if VERBOSE:
        print(f"[VERBOSE] {msg}")

def log(msg):
    """Normal logger (to file + console)"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------- SAFE SITE DESCRIPTION ----------
default_desc = (
    "Mauritius.mimusjobs.com: Your gateway to top jobs in Mauritius. "
    "Explore vacancies in tourism, finance, IT, and more from leading employers. "
    "Post resumes, apply easily, and advance your career on the island."
)

env_site_desc = os.environ.get("SITE_DESC", "").strip()
site_desc = env_site_desc if env_site_desc else default_desc
log(f"Site Description: {site_desc}")

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Running on device: {device}")

# ---------- MODEL ----------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-base")  # base for faster CPU use
vlog(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
log(f"Model {MODEL_NAME} loaded successfully on {device}")

# ---------- SENTENCE TRANSFORMER ----------
vlog("Loading MiniLM for title uniqueness checking...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer loaded successfully")

# ---------- GENERATION HELPERS ----------
def safe_generate(prompt, max_new_tokens=250, temperature=0.9, top_p=0.95, penalty=1.2, min_length=None):
    """Robust text generator with retries and full verbosity"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    params = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "repetition_penalty": penalty,
        "no_repeat_ngram_size": 3,
    }
    if min_length:
        params["min_length"] = min_length

    for attempt in range(3):
        try:
            vlog(f"Generating text (attempt {attempt+1}) with params: {params}")
            with torch.no_grad():
                output = model.generate(**params)
            text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            vlog(f"Generated text length: {len(text.split())} words")
            return text
        except Exception as e:
            log(f"Error during generation attempt {attempt+1}: {e}")
            time.sleep(2)
    raise RuntimeError("Text generation failed after 3 attempts")

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles=10):
    log(f"Generating {num_titles} titles for: {site_desc}")
    prompt = (
        f"Generate {num_titles} diverse, SEO-friendly blog post titles for a site described as '{site_desc}'. "
        f"Each title must be 6–12 words and unique. Return only a numbered list."
    )
    raw = safe_generate(prompt, max_new_tokens=200)
    vlog(f"Raw title output:\n{raw}")

    titles = []
    for line in raw.split("\n"):
        line = line.strip().lstrip("0123456789. )-").strip()
        if 4 <= len(line.split()) <= 14:
            titles.append(line)

    vlog(f"Parsed {len(titles)} candidate titles before deduplication")

    # Deduplicate
    unique_titles, embeddings = [], []
    for t in titles:
        emb = similarity_model.encode(t, convert_to_tensor=True)
        if not embeddings or all(util.cos_sim(emb, e).item() < 0.83 for e in embeddings):
            unique_titles.append(t)
            embeddings.append(emb)

    vlog(f"Final {len(unique_titles)} unique titles generated")
    return unique_titles[:num_titles]

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str):
    log(f"Generating article for: {title}")
    prompt = (
        f"Write a detailed blog article titled '{title}' for a website about {site_desc}. "
        f"Include introduction, 3–5 practical tips with examples, a real-world scenario, and conclusion. "
        f"Friendly expert tone, 450–700 words."
    )
    article = safe_generate(prompt, max_new_tokens=800, min_length=300)
    return article

# ---------- MAIN ----------
def main():
    log("Starting AI blog generation pipeline...")
    titles = generate_unique_titles(site_desc, num_titles=10)
    log(f"Generated {len(titles)} titles:\n" + "\n".join([f"- {t}" for t in titles]))

    articles = []
    for i, title in enumerate(titles, 1):
        log(f"[{i}/{len(titles)}] Generating article...")
        article = generate_article(title)
        articles.append({"title": title, "content": article})
        with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        progress = {"done": i, "total": len(titles), "current": title, "percent": int(i/len(titles)*100)}
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
        log(f"Article saved: {title}")

    log("✅ All articles generated successfully!")
    print("SUCCESS")

if __name__ == "__main__":
    main()
