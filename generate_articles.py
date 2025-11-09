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

log("AI Blog Generator Started – Flan-T5-Large + MiniLM (CPU)")

# ---------- SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site Description: {site_desc[:100]}...")

# ---------- MODELS ----------
log("Loading google/flan-t5-large ...")
device = torch.device("cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("Flan-T5-Large loaded")

log("Loading all-MiniLM-L6-v2 ...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer loaded")

# ---------- GLOBAL TRACKERS ----------
seen_title_hashes   = set()
seen_article_hashes = set()
title_embeddings    = []
article_embeddings  = []
articles            = []

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles: int = 15):
    log(f"Generating {num_titles} unique titles...")
    raw_titles = set()

    # --- Prompt that forces clean numbered list ---
    prompt = (
        f"Generate {num_titles * 3} diverse, SEO-friendly blog post titles for a job portal in Mauritius.\n"
        f"Site: {site_desc}\n"
        f"Rules:\n"
        f"- Each title: 6–12 words\n"
        f"- Cover IT, tourism, finance, healthcare, remote work, expats, etc.\n"
        f"- No duplicates\n"
        f"- Return ONLY a numbered list like:\n"
        f"1. First Title Here\n"
        f"2. Second Title Here\n"
        f"Do not add explanations."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.4,
            num_return_sequences=1
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    log(f"Raw LLM output:\n{raw}\n{'-'*50}")

    # --- Extract titles with regex ---
    pattern = re.compile(r'^\d+\.\s*(.+)$', re.MULTILINE)
    for match in pattern.finditer(raw):
        title = match.group(1).strip(' "\'')
        if 6 <= len(title.split()) <= 12:
            raw_titles.add(title)

    # --- Fallback: split by newline if regex fails ---
    if len(raw_titles) < num_titles:
        log("Regex failed, falling back to line split...")
        for line in raw.split('\n'):
            line = line.strip()
            if line and not line[0].isdigit():
                continue
            clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip(' "\'')
            if 6 <= len(clean.split()) <= 12:
                raw_titles.add(clean)

    log(f"Extracted {len(raw_titles)} raw titles")

    # --- Deduplicate with hash + embedding ---
    unique = []
    for t in raw_titles:
        h = hash(t.lower())
        if h in seen_title_hashes:
            continue
        emb = similarity_model.encode(t, convert_to_tensor=True)
        if title_embeddings and max(util.cos_sim(emb, e).item() for e in title_embeddings) > 0.88:
            continue
        unique.append(t)
        seen_title_hashes.add(h)
        title_embeddings.append(emb)
        if len(unique) >= num_titles:
            break

    # --- Paraphrase to fill gaps ---
    while len(unique) < num_titles and unique:
        base = random.choice(unique)
        var_prompt = (
            f"Create a completely new title with different wording but same topic as:\n"
            f'"{base}"\n'
            f"Site: {site_desc}\n"
            f"6–12 words. No quotes. No numbers."
        )
        inputs = tokenizer(var_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=1.1,
                do_sample=True,
                top_p=0.96,
                repetition_penalty=1.5
            )
        new_t = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if 6 <= len(new_t.split()) <= 12:
            h = hash(new_t.lower())
            if h not in seen_title_hashes:
                emb = similarity_model.encode(new_t, convert_to_tensor=True)
                if not title_embeddings or max(util.cos_sim(emb, e).item() for e in title_embeddings) <= 0.88:
                    unique.append(new_t)
                    seen_title_hashes.add(h)
                    title_embeddings.append(emb)

    final = unique[:num_titles]
    log(f"Final {len(final)} unique titles:")
    for i, t in enumerate(final, 1):
        log(f"  {i}. {t}")
    return final

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"Generating article: {title}")

    # Negative context
    prev_context = ""
    if article_embeddings:
        t_emb = similarity_model.encode(title, convert_to_tensor=True)
        sims = [util.cos_sim(t_emb, e).item() for e in article_embeddings]
        if sims:
            idx = sims.index(max(sims))
            prev_snip = " ".join(articles[idx]["content"].split(". ")[:2]) + "."
            prev_context = f"Avoid repeating: \"{prev_snip}\" "

    prompt = (
        f"{prev_context}"
        f"Write a detailed, original blog post titled:\n"
        f'"{title}"\n'
        f"For: {site_desc}\n"
        f"Structure:\n"
        f"- Engaging introduction\n"
        f"- 3–5 practical tips with Mauritius examples\n"
        f"- Real-world scenario\n"
        f"- Strong conclusion with CTA\n"
        f"400+ words. Natural paragraphs. Expert but friendly tone."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    for attempt in range(6):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.85,
                do_sample=True,
                top_p=0.94,
                repetition_penalty=1.3,
                min_length=380,
                no_repeat_ngram_size=3
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        wc = len(text.split())

        # Dedup
        h = hash(text.lower()[:500])
        if h in seen_article_hashes:
            continue
        seen_article_hashes.add(h)

        emb = similarity_model.encode(text, convert_to_tensor=True)
        if article_embeddings and max(util.cos_sim(emb, e).item() for e in article_embeddings) > 0.80:
            continue

        if wc < 400:
            continue

        article_embeddings.append(emb)
        log(f"Article ready ({wc} words)")
        return text

    log("Warning: All retries failed")
    return text

# ---------- MAIN ----------
try:
    topics = generate_unique_titles(site_desc, num_titles=15)
    if not topics:
        raise ValueError("No titles generated!")

    progress = {"total": len(topics), "done": 0, "current": "", "percent": 0}
    log("Starting article generation...")

    for i, title in enumerate(topics, 1):
        progress["current"] = title
        progress["done"] = i - 1
        progress["percent"] = round((i - 1) / len(topics) * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

        content = generate_article(title)
        articles.append({"title": title, "content": content})

        progress["done"] = i
        progress["percent"] = round(i / len(topics) * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    # Save
    with open(articles_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log(f"Saved {len(articles)} articles to {articles_file}")

    progress.update({"percent": 100, "current": "Complete"})
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    log("SUCCESS: All done!")
    print("SUCCESS")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    print("FAILED")
finally:
    log_handle.close()
