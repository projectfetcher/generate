#!/usr/bin/env python3
import os
import json
import torch
import re
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"

log_handle = open(log_file, "a", encoding="utf-8")
def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("AI Blog Generator Started – Flan-T5-XL + MiniLM (CPU)")

# ---------- SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site: {site_desc[:120]}...")

# ---------- MODELS ----------
log("Loading google/flan-t5-xl (better for list generation)...")
device = torch.device("cpu")
model_name = "google/flan-t5-xl"  # ← BETTER THAN large for structured output
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("Flan-T5-XL loaded on CPU")

log("Loading all-MiniLM-L6-v2...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer loaded")

# ---------- GLOBAL TRACKERS ----------
seen_title_hashes   = set()
title_embeddings    = []
articles            = []
article_embeddings  = []

# ---------- TITLE GENERATION (GUARANTEED TO WORK) ----------
def generate_title_with_fallback(topic_seed: str) -> str:
    prompt = (
        f"Write one SEO-friendly blog title about '{topic_seed}' for a Mauritius job portal.\n"
        f"Site: {site_desc}\n"
        f"Rules: 6–12 words, engaging, no quotes, no numbers at start."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    for temp in [0.9, 1.0, 1.1, 1.2]:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=temp,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.3
            )
        title = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        words = title.split()
        if 6 <= len(words) <= 12 and any(c.isalpha() for c in title):
            return title
    return " ".join(words[:12]) if words else "Default Mauritius Job Tips"

def generate_unique_titles(num_titles: int = 15):
    log(f"Generating {num_titles} unique titles...")
    unique = []
    seeds = [
        "IT jobs in Ebene", "hospitality careers Grand Baie", "finance roles Port Louis",
        "healthcare jobs Mauritius", "remote work for expats", "ATS resume tips",
        "job alerts setup", "tourism industry hiring", "tech startup roles",
        "luxury hotel management", "AI in recruitment", "expat relocation guide",
        "Kreol job search", "French-speaking jobs", "career change over 40"
    ] * 2

    for seed in seeds:
        if len(unique) >= num_titles:
            break
        title = generate_title_with_fallback(seed)
        h = hash(title.lower())
        if h in seen_title_hashes:
            continue
        emb = similarity_model.encode(title, convert_to_tensor=True)
        if title_embeddings and any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
            continue
        unique.append(title)
        seen_title_hashes.add(h)
        title_embeddings.append(emb)
        log(f"  Generated: {title}")

    # Final fill with variation
    while len(unique) < num_titles and unique:
        base = random.choice(unique)
        new_title = generate_title_with_fallback(f"variation of: {base}")
        h = hash(new_title.lower())
        if h not in seen_title_hashes:
            emb = similarity_model.encode(new_title, convert_to_tensor=True)
            if not any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
                unique.append(new_title)
                seen_title_hashes.add(h)
                title_embeddings.append(emb)

    final = unique[:num_titles]
    log(f"FINAL {len(final)} TITLES:")
    for i, t in enumerate(final, 1):
        log(f"  {i}. {t}")
    return final

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"Generating article: {title}")

    prev_context = ""
    if article_embeddings:
        t_emb = similarity_model.encode(title, convert_to_tensor=True)
        sims = [util.cos_sim(t_emb, e).item() for e in article_embeddings]
        if sims:
            idx = sims.index(max(sims))
            prev = articles[idx]["content"].split(". ")[:2]
            prev_context = f"Avoid repeating: \"{' '.join(prev)}.\" "

    prompt = (
        f"{prev_context}"
        f"Write a detailed blog post titled: \"{title}\"\n"
        f"For: {site_desc}\n"
        f"Include:\n"
        f"- Intro with Mauritius context\n"
        f"- 3–5 practical tips with local examples\n"
        f"- Real-world success story\n"
        f"- Conclusion with CTA to mimusjobs.com\n"
        f"Minimum 420 words. Natural tone."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    for attempt in range(6):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1100,
                temperature=0.88,
                do_sample=True,
                top_p=0.94,
                repetition_penalty=1.35,
                min_length=400,
                no_repeat_ngram_size=3
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        wc = len(text.split())

        if wc < 420:
            continue

        h = hash(text.lower()[:600])
        if h in seen_title_hashes:  # reuse seen_article_hashes
            continue

        emb = similarity_model.encode(text, convert_to_tensor=True)
        if article_embeddings and any(util.cos_sim(emb, e).item() > 0.78 for e in article_embeddings):
            continue

        article_embeddings.append(emb)
        seen_title_hashes.add(h)  # reuse set
        log(f"  Article ready ({wc} words)")
        return text

    log("  Using fallback article")
    return text

# ---------- MAIN ----------
try:
    topics = generate_unique_titles(num_titles=15)
    if not topics:
        raise ValueError("Title generation failed")

    progress = {"total": 15, "done": 0, "current": "", "percent": 0}
    log("Starting article loop...")

    for i, title in enumerate(topics, 1):
        progress["current"] = title[:50] + "..."
        progress["done"] = i - 1
        progress["percent"] = int((i - 1) / 15 * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        content = generate_article(title)
        articles.append({"title": title, "content": content})

        progress["done"] = i
        progress["percent"] = int(i / 15 * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

    # Save
    with open(articles_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log(f"SAVED: {len(articles)} articles → {articles_file}")

    progress.update({"percent": 100, "current": "Complete"})
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    log("SUCCESS: All done!")
    print("SUCCESS")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    print("FAILED")
finally:
    log_handle.close()
