#!/usr/bin/env python3
import os
import json
import torch
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

log("AI Blog Generator Started – Flan-T5-Large + MiniLM (Universal)")

# ---------- SITE DESCRIPTION (USER CAN CHANGE) ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site Description: {site_desc[:120]}...")

# ---------- MODELS ----------
log("Loading google/flan-t5-large ...")
device = torch.device("cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("Flan-T5-Large loaded on CPU")

log("Loading all-MiniLM-L6-v2 ...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("MiniLM loaded")

# ---------- GLOBAL STATE ----------
seen_title_hashes   = set()
title_embeddings    = []
seen_article_hashes = set()
article_embeddings  = []
articles            = []

# ---------- TITLE GENERATOR (NO SEEDS) ----------
def generate_diverse_title_prompts() -> list:
    """Ask LLM to suggest 30 diverse topic angles for the site."""
    prompt = (
        f"List 30 diverse, specific blog post topic ideas for a website described as:\n"
        f'"{site_desc}"\n'
        f"Each idea: 4–8 words, no duplicates, cover different user needs, industries, tips, stories.\n"
        f"Return ONLY a numbered list. Example:\n"
        f"1. Resume tips for IT jobs\n"
        f"2. How to find remote work\n"
        f"No explanations."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=1.0,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.4
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    topics = []
    for line in raw.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            topic = line.split('.', 1)[-1].strip(' "\'')
            if 4 <= len(topic.split()) <= 8:
                topics.append(topic)
    log(f"Generated {len(topics)} topic seeds from LLM")
    return topics[:30] or ["job search tips", "career change", "resume building", "interview prep"]

def generate_one_title_from_topic(topic: str) -> str:
    prompt = (
        f"Convert this short topic into a full, engaging, SEO-friendly blog title:\n"
        f'"{topic}"\n'
        f"Site: {site_desc}\n"
        f"Rules:\n"
        f"- 6 to 12 words\n"
        f"- Include location or industry if relevant\n"
        f"- No quotes, no numbers at start\n"
        f"- Example: How to Build an ATS-Friendly Resume in Mauritius"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    for temp in [0.8, 0.9, 1.0, 1.1]:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=temp,
                do_sample=True,
                top_p=0.94,
                repetition_penalty=1.4
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        words = text.split()
        if 6 <= len(words) <= 12 and any(c.isalpha() for c in text):
            return text
    return " ".join(words[:12]) if words else "Essential Career Tips for Job Seekers"

def generate_unique_titles(num_titles: int = 15):
    log(f"Generating {num_titles} unique titles (no seed topics)...")
    topic_seeds = generate_diverse_title_prompts()
    unique = []

    for topic in topic_seeds:
        if len(unique) >= num_titles:
            break
        title = generate_one_title_from_topic(topic)
        h = hash(title.lower())
        if h in seen_title_hashes:
            continue
        emb = similarity_model.encode(title, convert_to_tensor=True)
        if title_embeddings and any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
            continue
        unique.append(title)
        seen_title_hashes.add(h)
        title_embeddings.append(emb)
        log(f"  Title: {title}")

    # Fill gaps with paraphrasing
    while len(unique) < num_titles and unique:
        base = random.choice(unique)
        title = generate_one_title_from_topic(f"variation of: {base}")
        h = hash(title.lower())
        if h not in seen_title_hashes:
            emb = similarity_model.encode(title, convert_to_tensor=True)
            if not any(util.cos_sim(emb, e).item() > 0.88 for e in title_embeddings):
                unique.append(title)
                seen_title_hashes.add(h)
                title_embeddings.append(emb)

    final = unique[:num_titles]
    log(f"FINAL {len(final)} TITLES:")
    for i, t in enumerate(final, 1):
        log(f"  {i}. {t}")
    return final

# ---------- ARTICLE GENERATOR ----------
def generate_article(title: str) -> str:
    log(f"Generating article: {title}")

    prev_context = ""
    if article_embeddings:
        t_emb = similarity_model.encode(title, convert_to_tensor=True)
        sims = [util.cos_sim(t_emb, e).item() for e in article_embeddings]
        if sims:
            idx = sims.index(max(sims))
            prev = " ".join(articles[idx]["content"].split(". ")[:2]) + "."
            prev_context = f"Avoid repeating: \"{prev}\" "

    prompt = (
        f"{prev_context}"
        f"Write a detailed, original blog post titled:\n"
        f'"{title}"\n'
        f"For the site: {site_desc}\n"
        f"Include:\n"
        f"- Engaging intro with local context\n"
        f"- 3–5 practical tips with real examples\n"
        f"- Real-world success story\n"
        f"- Strong conclusion + CTA to the site\n"
        f"Minimum 420 words. Natural, friendly tone."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    for attempt in range(6):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1100,
                temperature=0.9,
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
        if h in seen_article_hashes:
            continue
        seen_article_hashes.add(h)

        emb = similarity_model.encode(text, convert_to_tensor=True)
        if article_embeddings and any(util.cos_sim(emb, e).item() > 0.78 for e in article_embeddings):
            continue

        article_embeddings.append(emb)
        log(f"  Article ready ({wc} words)")
        return text

    log("  Using last attempt")
    return text

# ---------- MAIN ----------
try:
    topics = generate_unique_titles(num_titles=15)
    if len(topics) < 10:
        raise ValueError("Too few titles generated")

    progress = {"total": 15, "done": 0, "current": "", "percent": 0}
    log("Starting article generation...")

    for i, title in enumerate(topics, 1):
        progress["current"] = title[:60] + "..." if len(title) > 60 else title
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
    log(f"SAVED {len(articles)} articles → {articles_file}")

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
