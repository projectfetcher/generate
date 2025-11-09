#!/usr/bin/env python3
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import random
from datetime import datetime
import time

# ---------- CONFIG ----------
log_file = "logs.txt"
progress_file = "progress.json"
articles_file = "articles.json"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    log_handle.write(full_msg + "\n")
    log_handle.flush()

log("AI Blog Generator Started – Using Flan-T5-XL + MiniLM (High Quality Mode)")

# ---------- HARDCODED SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com is a premier job portal dedicated to connecting talent with opportunities across Mauritius's thriving economy. "
    "From IT roles in Ebene Cybercity to luxury hospitality positions in Grand Baie, the platform features thousands of verified listings in tourism, finance, tech, healthcare, and more. "
    "Job seekers can upload resumes, build ATS-friendly profiles, and receive tailored job alerts, while employers benefit from advanced recruitment tools and company branding. "
    "With a mobile-optimized interface, multilingual support (English, French, Kreol), and AI-powered matching, it empowers locals and expatriates alike to advance their careers in one of the Indian Ocean’s most dynamic job markets."
)
log(f"Site Description (hardcoded): {site_desc}")

# ---------- MODEL & TOKENIZER (UPGRADED TO FLAN-T5-XL) ----------
log("Loading google/flan-t5-xl (3B params) – this may take 1-2 minutes on first run...")
device = torch.device("cpu")
model_name = "google/flan-t5-xl"  # Upgraded from large → xl
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()
log("Flan-T5-XL loaded successfully on CPU")

# ---------- SENTENCE TRANSFORMER ----------
log("Loading all-MiniLM-L6-v2 for semantic deduplication...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
log("SentenceTransformer ready")

# ---------- TITLE GENERATION ----------
def generate_unique_titles(site_desc: str, num_titles: int = 15):
    log(f"Generating {num_titles} high-quality, unique blog titles...")
    prompt = (
        f"Generate {num_titles} diverse, engaging, SEO-optimized blog post titles for a job portal: "
        f"'{site_desc}'. "
        f"Each title: 7–11 words, unique angle, actionable or insightful. "
        f"Cover career tips, industry trends, job search hacks, Mauritius-specific insights. "
        f"Return ONLY a numbered list. Example format:\n"
        f"1. How to Land Tech Jobs in Ebene Cybercity Fast\n"
        f"No explanations. No duplicates."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.95,
            do_sample=True,
            top_p=0.96,
            repetition_penalty=1.3,
            num_return_sequences=1
        )
    raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    log(f"Raw LLM Output for Titles:\n{raw}\n")

    # Parse titles
    titles = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not any(c.isalnum() for c in line):
            continue
        # Extract after number
        clean = line.split(".", 1)[-1].split(":", 1)[-1].strip(' "\'-')
        words = clean.split()
        if 7 <= len(words) <= 11 and clean[0].isupper():
            titles.append(clean)

    log(f"Extracted {len(titles)} raw title candidates")

    # Deduplicate semantically
    unique_titles = []
    embeddings = []
    for title in titles:
        if len(unique_titles) >= num_titles:
            break
        emb = similarity_model.encode(title, convert_to_tensor=True)
        if not embeddings:
            unique_titles.append(title)
            embeddings.append(emb)
            log(f"Title {len(unique_titles)}: {title}")
            continue
        sims = [util.cos_sim(emb, e).item() for e in embeddings]
        if not any(s > 0.88 for s in sims):  # Stricter threshold
            unique_titles.append(title)
            embeddings.append(emb)
            log(f"Title {len(unique_titles)}: {title}")

    # Generate variations if needed
    while len(unique_titles) < num_titles and unique_titles:
        base = random.choice(unique_titles)
        var_prompt = (
            f"Create a completely reworded, fresh blog title variation of:\n\"{base}\"\n"
            f"Same topic, different phrasing. 7–11 words. Start with verb or question. "
            f"SEO-friendly. For Mauritius job site."
        )
        inputs = tokenizer(var_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=1.1,
                do_sample=True,
                top_p=0.9
            )
        new_title = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        words = new_title.split()
        if 7 <= len(words) <= 11 and new_title[0].isupper():
            emb = similarity_model.encode(new_title, convert_to_tensor=True)
            sims = [util.cos_sim(emb, e).item() for e in embeddings]
            if not any(s > 0.88 for s in sims):
                unique_titles.append(new_title)
                embeddings.append(emb)
                log(f"Title {len(unique_titles)} (variation): {new_title}")

    final_titles = unique_titles[:num_titles]
    log(f"FINAL {len(final_titles)} UNIQUE TITLES GENERATED:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(final_titles)]))
    return final_titles

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    log(f"\nGenerating article for:\n→ {title}")
    prompt = (
        f"Write a professional, engaging, and actionable blog post titled:\n"
        f"\"{title}\"\n"
        f"For the job portal: {site_desc}\n\n"
        f"Structure:\n"
        f"1. Hook introduction (mention Mauritius job market)\n"
        f"2. 4 practical tips with real examples (local companies, locations)\n"
        f"3. One detailed real-world success story\n"
        f"4. Strong conclusion with CTA to use mimusjobs.com\n\n"
        f"Tone: Friendly expert. 500–700 words. Natural paragraphs. Use bullet points. "
        f"Include keywords: Mauritius jobs, Ebene, Grand Baie, resume tips, career growth."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    for attempt in range(3):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=900,
                temperature=0.82,
                do_sample=True,
                top_p=0.94,
                repetition_penalty=1.18,
                min_length=450
            )
        article = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        word_count = len(article.split())
        
        if word_count >= 500:
            log(f"Article generated successfully ({word_count} words)")
            return article
        else:
            log(f"Attempt {attempt+1}: Too short ({word_count} words). Regenerating...")

    log("Final fallback: Using last generated article despite length")
    return article

# ---------- MAIN LOOP WITH LIVE PROGRESS ----------
try:
    log("="*60)
    log("STARTING TITLE GENERATION PHASE")
    log("="*60)
    titles = generate_unique_titles(site_desc, num_titles=15)

    articles = []
    total = len(titles)
    progress = {"total": total, "done": 0, "current": "", "percent": 0}

    log("="*60)
    log("STARTING ARTICLE GENERATION PHASE")
    log("="*60)

    for i, title in enumerate(titles, 1):
        progress["current"] = title
        progress["done"] = i - 1
        progress["percent"] = int((i - 1) / total * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

        content = generate_article(title)
        articles.append({"title": title, "content": content})

        # Update progress
        progress["done"] = i
        progress["percent"] = int(i / total * 100)
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

        log(f"PROGRESS: {i}/{total} – {progress['percent']}% – Saved: {articles_file}")

    # Save final output
    with open(articles_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log(f"SUCCESS: {articles_file} saved with {len(articles)} articles")

    progress["percent"] = 100
    progress["current"] = "All articles generated!"
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    log("progress.json updated to 100%")
    log("AI BLOG GENERATION COMPLETE!")

    print("\n" + "="*60)
    print("SUCCESS: All 15 articles generated and saved!")
    print(f"→ Check: {articles_file}")
    print(f"→ Logs: {log_file}")
    print("="*60)

except Exception as e:
    log(f"CRITICAL ERROR: {str(e)}")
    import traceback
    log(traceback.format_exc())
    print("FAILED")
finally:
    log_handle.close()
