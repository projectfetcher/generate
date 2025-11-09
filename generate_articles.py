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

# Model config: change MODEL_NAME to a smaller model if you run on CPU only.
# If you have a GPU, this large model will perform much better.
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-large")
# if you're CPU-only and slow, try: "google/flan-t5-small" or "google/flan-t5-base"
TITLE_COUNT = 15
MAX_TITLE_GEN_ATTEMPTS = 6
MAX_ARTICLE_RETRIES = 2

# ---------- LOGGING ----------
log_handle = open(LOG_FILE, "a", encoding="utf-8")

def log(msg):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    log_handle.write(line + "\n")
    log_handle.flush()

log("AI Blog Generator Started")

# ---------- HARDCODED SITE DESCRIPTION ----------
site_desc = (
    "Mauritius.mimusjobs.com: Your gateway to top jobs in Mauritius. "
    "Explore vacancies in tourism, finance, IT, and more from leading employers. "
    "Post resumes, apply easily, and advance your career on the island."
)
log(f"Site Description (hardcoded): {site_desc}")

# ---------- DEVICE DETECTION ----------
if torch.cuda.is_available():
    device = torch.device("cuda")
    log("CUDA is available. Using GPU.")
else:
    # If you have many CPU cores, Accelerate or bitsandbytes would help — but keep simple here
    device = torch.device("cpu")
    log("CUDA not available. Using CPU.")

# ---------- LOAD TOKENIZER & MODEL ----------
log(f"Loading model/tokenizer: {MODEL_NAME} (this may take a while)")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    log(f"Model {MODEL_NAME} loaded and moved to {device}")
except Exception as e:
    log(f"ERROR loading model {MODEL_NAME}: {e}")
    raise

# ---------- SENTENCE TRANSFORMER ----------
try:
    # SentenceTransformer takes "cpu" or "cuda" for device string
    st_device = "cuda" if device.type == "cuda" else "cpu"
    log(f"Loading SentenceTransformer (device={st_device}) for title deduplication...")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device=st_device)
    log("SentenceTransformer loaded")
except Exception as e:
    log(f"ERROR loading SentenceTransformer: {e}")
    raise

# ---------- HELPERS ----------
def save_json_atomic(path: str, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(path)

def safe_generate(prompt, max_new_tokens=200, temperature=0.9, top_p=0.95, 
                  do_sample=True, penalty=1.2, min_length=None, num_return_sequences=1):
    """
    Wrapper for model.generate with robust kwargs and retries on OOM / failure.
    Returns decoded string (first sequence by default).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=penalty,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    if min_length:
        gen_kwargs["min_length"] = min_length

    for attempt in range(3):
        try:
            with torch.no_grad():
                out = model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
            # handle multiple return sequences
            if isinstance(out, list) or out.shape[0] > 1:
                results = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
                return results[0]
            return tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except RuntimeError as e:
            # common when GPU OOM or similar
            log(f"Generation runtime error (attempt {attempt+1}/3): {e}")
            torch.cuda.empty_cache()
            time.sleep(1 + attempt)
        except Exception as e:
            log(f"Unexpected generation error (attempt {attempt+1}/3): {e}")
            time.sleep(1 + attempt)
    raise RuntimeError("Generation failed after retries")

# ---------- TITLE GENERATION ----------
def parse_titles(raw_text):
    """
    Extract lines that look like titles, remove leading numbers and punctuation,
    strip extraneous characters, and only keep titles 4-14 words long.
    """
    tlist = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading numbering like "1." or "1)"
        if line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) > 1 and parts[1].strip():
                line = parts[1].strip()
            else:
                # maybe "1) Title"
                parts = line.split(")", 1)
                if len(parts) > 1 and parts[1].strip():
                    line = parts[1].strip()
        # drop quoted output markers
        line = line.strip(' "\'–—-:')
        # ignore lines that are too short or too long
        words = line.split()
        if 4 <= len(words) <= 14 and any(c.isalpha() for c in line):
            tlist.append(line)
    return tlist

def generate_unique_titles(site_desc: str, num_titles: int = 15):
    attempts = 0
    collected = []
    embeddings = []

    base_prompt = (
        f"Generate {max(6, num_titles)} diverse, SEO-friendly blog post titles "
        f"for a website described as: {site_desc}\n"
        f"Requirements: 6-12 words each, varied angles, friendly tone. Return only a numbered list, "
        f"one title per line. No explanations."
    )

    while len(collected) < num_titles and attempts < MAX_TITLE_GEN_ATTEMPTS:
        attempts += 1
        log(f"Title generation attempt {attempts}")
        raw = safe_generate(base_prompt, max_new_tokens=180, temperature=0.9, top_p=0.95, penalty=1.2)
        log(f"Raw title output:\n{raw[:1000]}")  # log only head to avoid megabytes
        parsed = parse_titles(raw)
        log(f"Parsed {len(parsed)} candidate titles")
        for t in parsed:
            if len(collected) >= num_titles:
                break
            # compute embedding and filter duplicates
            emb = similarity_model.encode(t, convert_to_tensor=True)
            if not embeddings:
                collected.append(t)
                embeddings.append(emb)
                continue
            sims = [util.cos_sim(emb, e).item() for e in embeddings]
            if max(sims) < 0.82:
                collected.append(t)
                embeddings.append(emb)

        # if not enough, ask for variations deterministically
        if len(collected) < num_titles:
            # generate variations for existing
            for base in list(collected):
                if len(collected) >= num_titles:
                    break
                var_prompt = (
                    f"Create one fresh, unique blog title variation of: \"{base}\". "
                    f"Keep the topic area but change wording. 6-12 words. Return only the title."
                )
                v = safe_generate(var_prompt, max_new_tokens=40, temperature=0.95, top_p=0.92)
                v = v.strip().strip(' "\'–—-:')
                if 4 <= len(v.split()) <= 14 and any(c.isalpha() for c in v):
                    emb = similarity_model.encode(v, convert_to_tensor=True)
                    sims = [util.cos_sim(emb, e).item() for e in embeddings]
                    if max(sims) < 0.82:
                        collected.append(v)
                        embeddings.append(emb)

    # fallback: if still short, pad by rewording site_desc
    while len(collected) < num_titles:
        fallback = f"Top jobs in Mauritius: how to apply and succeed - variation {len(collected)+1}"
        collected.append(fallback)

    return collected[:num_titles]

# ---------- ARTICLE GENERATION ----------
def generate_article(title: str) -> str:
    """Generate one article. Retry a small bounded number of times if output too short."""
    log(f"Generating article for title: {title}")
    prompt = (
        f"Write a helpful, detailed blog post titled: \"{title}\" for a website about {site_desc}.\n\n"
        "Include: a short introduction, 3-5 practical tips with examples, a real-world scenario, and a strong conclusion. "
        "Use a friendly expert tone, natural paragraphs. Aim for 450-700 words. Do not output section headers only — write prose."
    )

    for attempt in range(1, MAX_ARTICLE_RETRIES + 2):
        out = safe_generate(prompt, max_new_tokens=900, temperature=0.8, top_p=0.92, penalty=1.15, min_length=300)
        word_count = len(out.split())
        log(f"Attempt {attempt}: generated {word_count} words")
        # basic quality checks
        if word_count >= 350:
            return out
        else:
            log("Generated article is too short; retrying with higher temperature and sampling...")
            # tweak prompt and retry (do not recurse indefinitely)
            prompt += "\nPlease expand the post with more concrete details and examples."
    # if still short, return what we have (safer than infinite recursion)
    log("Warning: returning shorter article after retries")
    return out

# ---------- MAIN ----------
def main():
    try:
        titles = generate_unique_titles(site_desc, num_titles=TITLE_COUNT)
        log(f"Generated {len(titles)} titles")
        save_json_atomic(PROGRESS_FILE, {"total": len(titles), "done": 0, "current": "", "percent": 0})

        articles = []
        for i, t in enumerate(titles, start=1):
            log(f"Starting {i}/{len(titles)}: {t}")
            # update progress file early so we have trace if interrupted
            save_json_atomic(PROGRESS_FILE, {"total": len(titles), "done": i-1, "current": t, "percent": int((i-1)/len(titles)*100)})
            try:
                content = generate_article(t)
                articles.append({"title": t, "content": content})
                # save after each article to avoid data loss
                save_json_atomic(ARTICLES_FILE, articles)
                save_json_atomic(PROGRESS_FILE, {"total": len(titles), "done": i, "current": t, "percent": int(i/len(titles)*100)})
                log(f"Saved article {i}: {t}")
            except Exception as e:
                log(f"Failed to generate article for title '{t}': {e}")
                # continue but record a placeholder
                articles.append({"title": t, "content": f"ERROR_GENERATING_ARTICLE: {str(e)}"})
                save_json_atomic(ARTICLES_FILE, articles)

        # final save & progress
        save_json_atomic(ARTICLES_FILE, articles)
        save_json_atomic(PROGRESS_FILE, {"total": len(titles), "done": len(titles), "current": "Complete", "percent": 100})
        log(f"All done. {ARTICLES_FILE} saved with {len(articles)} articles")
        print("SUCCESS")
    except Exception as e:
        log(f"ERROR in main: {e}")
        save_json_atomic(PROGRESS_FILE, {"total": 0, "done": 0, "current": "Failed", "percent": 0})
        print("FAILED")
    finally:
        log_handle.close()

if __name__ == "__main__":
    main()
