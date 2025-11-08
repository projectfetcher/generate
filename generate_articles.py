import os
import json
import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ===================================================================
# USER CONFIG – CHANGE ONLY THIS
# ===================================================================
site_desc = """Jobs Mauritius is the leading online job portal in Mauritius, connecting employers with skilled talent across industries like IT, finance, tourism, and BPO. Featuring daily job listings, resume uploads, career advice, and company reviews, it helps Mauritian job seekers find local and remote opportunities fast."""

# Or for a travel site:
# site_desc = """WanderWorld is a travel inspiration platform featuring destination guides, budget tips, packing lists, and real traveler stories from over 100 countries."""

# Or for e-commerce:
# site_desc = """TechTrendz sells cutting-edge gadgets, laptops, and smart home devices with fast shipping, price match guarantee, and expert reviews."""

NUM_ARTICLES = 15
MIN_WORDS = 500
# ===================================================================

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
log_file = "blog_generator.log"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    log_handle.write(f"[{timestamp}] {msg}\n")
    log_handle.flush()

log("Universal Blog Generator Started")

# -------------------------------------------------------------------
# LOAD MODELS (CPU)
# -------------------------------------------------------------------
device = torch.device("cpu")

# Title model
title_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
title_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
title_model.eval().to(device)
log("Title model loaded")

# Article model
article_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
article_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
article_model.eval().to(device)
log("Article model loaded")

# Similarity model (detect duplicate content)
sim_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
log("Similarity model loaded")

# -------------------------------------------------------------------
# STEP 1: GENERATE UNIQUE TITLES
# -------------------------------------------------------------------
def generate_titles(site_desc: str, n: int = 15) -> list:
    seen = set()
    titles = []
    prompt = f"""
You are a professional blog editor. Generate {n * 2} unique, engaging, and SEO-friendly blog post titles based **only** on this site description.

Site description:
{site_desc}

Rules:
- Titles must be questions, how-to guides, or listicles.
- Each title must be completely different.
- Return **only** a numbered list: 1. Title... 2. Title...
- No explanations, no intro, no extra text.
"""
    log(f"Generating {n * 2} title candidates...")

    while len(titles) < n:
        inputs = title_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = title_model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.9,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.3
            )
        raw = title_tokenizer.decode(output[0], skip_special_tokens=True)

        for line in raw.split("\n"):
            line = line.strip()
            if not line or not re.match(r"^\d+\.", line):
                continue
            candidate = re.split(r"^\d+\.\s*", line, 1)[-1].strip(' "')
            if candidate and candidate not in seen:
                seen.add(candidate)
                titles.append(candidate)
                log(f"Title [{len(titles)}/{n}]: {candidate}")
            if len(titles) >= n:
                break
    return titles

titles = generate_titles(site_desc, NUM_ARTICLES)
log(f"Generated {len(titles)} unique titles")

# -------------------------------------------------------------------
# STEP 2: GENERATE UNIQUE ARTICLES (≥500 words)
# -------------------------------------------------------------------
articles = []
content_embeddings = []

def is_duplicate_content(text: str, threshold: float = 0.88) -> bool:
    if not content_embeddings:
        return False
    emb = sim_model.encode(text, convert_to_tensor=True)
    for prev in content_embeddings:
        if util.cos_sim(emb, prev) > threshold:
            return True
    return False

def generate_article(title: str) -> str:
    prompt = f"""
Write a high-quality, engaging blog post of at least {MIN_WORDS} words with this exact title:

"{title}"

The post is for this website:
{site_desc}

Structure:
- Start with a compelling hook
- Use 4–6 subheadings
- Include practical tips, examples, or insights
- Add bullet points or numbered lists
- End with a strong conclusion and call-to-action

Write naturally, conversationally, and for the target audience. Use bold **key phrases**.
"""
    inputs = article_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    while True:
        with torch.no_grad():
            output = article_model.generate(
                **inputs,
                max_new_tokens=1200,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
        text = article_tokenizer.decode(output[0], skip_special_tokens=True).strip()

        words = len(text.split())
        if words < MIN_WORDS:
            log(f"  Too short ({words} words), retrying...")
            continue

        if is_duplicate_content(text):
            log(f"  Duplicate content detected, regenerating...")
            continue

        # Save embedding
        emb = sim_model.encode(text, convert_to_tensor=True)
        content_embeddings.append(emb)
        return text

# -------------------------------------------------------------------
# MAIN GENERATION LOOP
# -------------------------------------------------------------------
progress = {"total": len(titles), "done": 0, "current": "", "percent": 0}

for idx, title in enumerate(titles, 1):
    # Update progress
    progress.update({"current": title, "done": idx-1, "percent": int((idx-1)/len(titles)*100)})
    with open("progress.json", "w") as f:
        json.dump(progress, f)

    log(f"[{idx}/{len(titles)}] Generating article: {title}")
    content = generate_article(title)
    word_count = len(content.split())

    articles.append({
        "title": title,
        "content": content,
        "word_count": word_count,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    log(f"  Done: {word_count} words")

    # Update progress
    progress.update({"done": idx, "percent": int(idx/len(titles)*100)})
    with open("progress.json", "w") as f:
        json.dump(progress, f)

# -------------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------------
final_output = {
    "site_description": site_desc,
    "generated_on": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_articles": len(articles),
    "min_words_per_article": MIN_WORDS,
    "articles": articles
}

with open("blog_articles.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

with open("titles.txt", "w", encoding="utf-8") as f:
    for t in titles:
        f.write(t + "\n")

progress["percent"] = 100
with open("progress.json", "w") as f:
    json.dump(progress, f)

log("All done! Files: blog_articles.json, titles.txt, progress.json")
log_handle.close()

# -------------------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------------------
print("\n" + "="*70)
print("GENERIC BLOG GENERATOR – COMPLETE")
print(f"Site: {site_desc.split('.')[0][:50]}...")
print(f"Articles: {len(articles)} × ≥{MIN_WORDS} words")
print(f"Output: blog_articles.json")
print("="*70)
