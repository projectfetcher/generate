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
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
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
log("Loading all-MiniLM-L6-v2 for uniqueness...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer loaded")

# ---------- GLOBAL RE-PETITION TRACKERS ----------
seen_title_hashes   = set()
seen_article_hashes = set()
title_embeddings    = []
article_embeddings  = []
articles            = []  # final list to save

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles: int = 15):
    log(f"Generating {num_titles} unique blog titles...")
    all_titles = []

    # --- Step 1: Generate more than needed ---
    prompt = (
        f"Generate {num_titles * 2} diverse, engaging, unique blog post titles "
        f"for the site: '{site_desc}'. "
        f"Each title 6-12 words, SEO-friendly, no duplicates. "
        f"Return only a numbered list."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.95,
            do_sample=True,
            top_p=0.96,
            repetition_penalty=1.3,
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    log(f"Raw LLM titles:\n{raw}")

    # --- Step 2: Parse titles ---
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not any(c.isalnum() for c in line):
            continue
        clean = line.split(".", 1)[-1].split(":", 1)[-1].strip(' "\'-')
        if 6 <= len(clean.split()) <= 12:
            all_titles.append(clean)

    # --- Step 3: Deduplicate with hash + cosine ---
    unique = []
    for t in all_titles:
        h = hash(t.lower())
        if h in seen_title_hashes:
            continue
        emb = similarity_model.encode(t, convert_to_tensor=True)
        if title_embeddings and any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
            continue

        unique.append(t)
        seen_title_hashes.add(h)
        title_embeddings.append(emb)
        if len(unique) == num_titles:
            break

    # --- Step 4: Fill gaps with paraphrasing ---
    while len(unique) < num_titles and unique:
        base = random.choice(unique)
        var_prompt = (
            f"Paraphrase this title while keeping the same topic for '{site_desc}'. "
            f"Original: \"{base}\". 6-12 words, completely different wording, no quotes."
        )
        inputs = tokenizer(var_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=1.0,
                do_sample=True,
                top_p=0.94,
                repetition_penalty=1.4,
            )
        new_t = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if 6 <= len(new_t.split()) <= 12:
            h = hash(new_t.lower())
            if h not in seen_title_hashes:
                emb = similarity_model.encode(new_t, convert_to_tensor=True)
                if not any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
                    unique.append(new_t)
                    seen_title_hashes.add(h)
                    title_embeddings.append(emb)

    log(f"Final unique titles ({len(unique)}):")
    for t in unique:
        log(f"  • {t}")
    return unique[:num_titles]

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"Generating article: {title}")

    # --- Build negative context from most similar prior article ---
    prev_context = ""
    if article_embeddings:
        title_emb = similarity_model.encode(title, convert_to_tensor=True)
        sims = [util.cos_sim(title_emb, e).item() for e in article_embeddings]
        if sims:
            most_sim_idx = sims.index(max(sims))
            first_two = " ".join(articles[most_sim_idx]["content"].split(". ")[:2]) + "."
            prev_context = f"Avoid repeating ideas from this previous post: \"{first_two}\" "

    prompt = (
        f"{prev_context}"
        f"Write a fresh, detailed blog post titled \"{title}\" for the site: {site_desc}. "
        f"Structure: introduction → 3–5 practical tips with Mauritius-specific examples → real-world scenario → strong conclusion. "
        f"Friendly expert tone, 400+ words, natural paragraphs, no bullet lists."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    for attempt in range(5):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=900,
                temperature=0.82,
                do_sample=True,
                top_p=0.93,
                repetition_penalty=1.25,
                min_length=350,
                no_repeat_ngram_size=3,
            )
        article = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        wc = len(article.split())

        # --- Hash dedup ---
        h = hash(article.lower()[:500])
        if h in seen_article_hashes:
            log(f"  Hash collision – retry {attempt+1}")
            continue
        seen_article_hashes.add(h)

        # --- Embedding dedup ---
        emb = similarity_model.encode(article, convert_to_tensor=True)
        if article_embeddings and any(util.cos_sim(emb, e).item() > 0.80 for e in article_embeddings):
            log(f"  Too similar to prior – retry {attempt+1}")
            continue

        # --- Word count guard ---
        if wc < 400:
            log(f"  Too short ({wc} words) – retry {attempt+1}")
            continue

        # --- Success ---
        article_embeddings.append(emb)
        log(f"  Article ready ({wc} words)")
        return article

    log("  All retries failed – returning last attempt")
    return article

# ---------- MAIN LOOP ----------
try:
    topics = generate_unique_titles(site_desc, num_titles=15)
    log(f"Final {len(topics)} Unique Titles:\n" + "\n".join([f"- {t}" for t in topics]))

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

    # Save final articles
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
    import traceback
    log(traceback.format_exc())
    print("FAILED")

finally:
    log_handle.close()
