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

NUM_ARTICLES = 15
MIN_WORDS = 500
VERBOSE = True  # Set to False for quiet mode
# ===================================================================

# -------------------------------------------------------------------
# VERBOSE LOGGING
# -------------------------------------------------------------------
log_file = "blog_generator_verbose.log"
log_handle = open(log_file, "a", encoding="utf-8")

def vlog(msg: str, level: str = "INFO"):
    if not VERBOSE and level != "ERROR":
        return
    timestamp = time.strftime("%H:%M:%S")
    prefix = f"[{timestamp}] [{level.ljust(5)}]"
    print(f"{prefix} {msg}")
    log_handle.write(f"{prefix} {msg}\n")
    log_handle.flush()

vlog("Universal Blog Generator (VERBOSE MODE) Started", "INFO")

# -------------------------------------------------------------------
# LOAD MODELS (CPU) – with token count logging
# -------------------------------------------------------------------
device = torch.device("cpu")
vlog(f"Using device: {device}", "INFO")

# Title model
vlog("Loading title generation model (Flan-T5-Large)...", "INFO")
title_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
title_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
title_model.eval().to(device)
vlog("Title model loaded", "INFO")

# Article model
vlog("Loading article generation model (Flan-T5-Large)...", "INFO")
article_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
article_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
article_model.eval().to(device)
vlog("Article model loaded", "INFO")

# Similarity model
vlog("Loading similarity model (MiniLM-L6-v2)...", "INFO")
sim_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
vlog("Similarity model loaded", "INFO")

# -------------------------------------------------------------------
# STEP 1: GENERATE UNIQUE TITLES (VERBOSE)
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
    vlog(f"Title prompt prepared ({len(prompt.split())} words)", "DEBUG")
    vlog(f"Generating up to {n * 2} title candidates to ensure {n} unique...", "INFO")

    attempt = 0
    while len(titles) < n:
        attempt += 1
        vlog(f"Title generation attempt #{attempt}", "INFO")

        inputs = title_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"]
        vlog(f"Input tokens: {input_ids.shape[1]}", "DEBUG")

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
        vlog(f"Raw model output ({len(raw.split())} words):\n{raw[:500]}{'...' if len(raw) > 500 else ''}", "DEBUG")

        new_count = 0
        for line in raw.split("\n"):
            line = line.strip()
            if not line or not re.match(r"^\d+\.", line):
                continue
            candidate = re.split(r"^\d+\.\s*", line, 1)[-1].strip(' "')
            if not candidate:
                continue

            if candidate in seen:
                vlog(f"Duplicate title skipped: {candidate}", "WARN")
                continue

            seen.add(candidate)
            titles.append(candidate)
            new_count += 1
            vlog(f"Accepted title [{len(titles)}/{n}]: {candidate}", "INFO")

            if len(titles) >= n:
                break

        vlog(f"Added {new_count} new titles in this attempt", "INFO")

    vlog(f"Successfully generated {len(titles)} unique titles", "INFO")
    return titles

titles = generate_titles(site_desc, NUM_ARTICLES)

# -------------------------------------------------------------------
# STEP 2: GENERATE UNIQUE ARTICLES (VERBOSE + SIMILARITY LOG)
# -------------------------------------------------------------------
articles = []
content_embeddings = []

def is_duplicate_content(text: str, threshold: float = 0.88) -> tuple[bool, float]:
    if not content_embeddings:
        return False, 0.0
    emb = sim_model.encode(text, convert_to_tensor=True)
    max_sim = 0.0
    for prev in content_embeddings:
        sim = util.cos_sim(emb, prev).item()
        if sim > max_sim:
            max_sim = sim
    is_dup = max_sim > threshold
    return is_dup, max_sim

def generate_article(title: str, attempt: int = 1) -> str:
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
    vlog(f"Article prompt for '{title}' ({len(prompt.split())} words)", "DEBUG")

    inputs = article_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_tokens = inputs["input_ids"].shape[1]
    vlog(f"Input tokens: {input_tokens}", "DEBUG")

    while True:
        vlog(f"Generating article attempt #{attempt} for: {title}", "INFO")
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
        vlog(f"Generated {words} words", "INFO")

        if words < MIN_WORDS:
            vlog(f"Too short ({words} < {MIN_WORDS}), retrying...", "WARN")
            attempt += 1
            continue

        is_dup, sim_score = is_duplicate_content(text)
        if is_dup:
            vlog(f"Duplicate content detected (similarity: {sim_score:.3f}), regenerating...", "WARN")
            attempt += 1
            continue

        # Save embedding
        emb = sim_model.encode(text, convert_to_tensor=True)
        content_embeddings.append(emb)
        vlog(f"Article accepted: {words} words, similarity: {sim_score:.3f}", "INFO")
        return text

# -------------------------------------------------------------------
# MAIN GENERATION LOOP (VERBOSE PROGRESS)
# -------------------------------------------------------------------
progress = {"total": len(titles), "done": 0, "current": "", "percent": 0}

for idx, title in enumerate(titles, 1):
    # Update progress
    progress.update({"current": title, "done": idx-1, "percent": int((idx-1)/len(titles)*100)})
    with open("progress.json", "w") as f:
        json.dump(progress, f)
    vlog(f"Progress: {progress['done']}/{progress['total']} ({progress['percent']}%)", "INFO")

    vlog(f"[{idx}/{len(titles)}] Generating article: {title}", "INFO")
    content = generate_article(title)
    word_count = len(content.split())

    articles.append({
        "title": title,
        "content": content,
        "word_count": word_count,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    vlog(f"Article saved: {word_count} words", "INFO")

    # Final progress
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
    "verbose_mode": VERBOSE,
    "articles": articles
}

with open("blog_articles.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)
vlog("Saved: blog_articles.json", "INFO")

with open("titles.txt", "w", encoding="utf-8") as f:
    for t in titles:
        f.write(t + "\n")
vlog("Saved: titles.txt", "INFO")

progress["percent"] = 100
with open("progress.json", "w") as f:
    json.dump(progress, f)
vlog("Progress: 100%", "INFO")

vlog("Pipeline completed successfully!", "INFO")
log_handle.close()

# -------------------------------------------------------------------
# FINAL VERBOSE SUMMARY
# -------------------------------------------------------------------
print("\n" + "="*80)
print("VERBOSE BLOG GENERATOR – COMPLETE")
print(f"Site: {site_desc.split('.')[0][:60]}...")
print(f"Articles: {len(articles)} × ≥{MIN_WORDS} words")
print(f"Log: {log_file}")
print(f"Output: blog_articles.json, titles.txt, progress.json")
print("="*80)
