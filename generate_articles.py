#!/usr/bin/env python3
# generate_articles.py
import os
import re
import textwrap
import random
import argparse
from transformers import pipeline, set_seed
import torch

# ---------------------------
# Configurable defaults (can be overridden by env or CLI)
# ---------------------------
DEFAULT_MODEL = os.getenv("MODEL_NAME", "distilgpt2")  # light CPU-friendly default
DEFAULT_NUM_TOPICS = int(os.getenv("TOPICS") or 5)
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS") or 400)
DEFAULT_TEMPERATURE = float(os.getenv("TEMP") or 0.85)
DEFAULT_TOP_P = float(os.getenv("TOP_P") or 0.92)
DEFAULT_REP_PENALTY = float(os.getenv("REPETITION_PENALTY") or 1.1)
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_articles")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def device_index():
    # Use GPU if available and USE_GPU env var is set to a true-ish value
    use_gpu_env = os.getenv("USE_GPU", "").lower() in ("1", "true", "yes")
    if torch.cuda.is_available() and use_gpu_env:
        return 0
    return -1

def clean_text(text: str) -> str:
    text = text.strip()
    # remove incomplete last sentence if missing punctuation
    pieces = re.split(r'(?<=[.!?])\s+', text)
    if pieces and not text.endswith(('.', '!', '?')):
        pieces = pieces[:-1]
    text = ' '.join(pieces).strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def sanitize_filename(s: str, maxlen: int = 50) -> str:
    s = s.lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:maxlen] if len(s) > maxlen else s

def parse_titles_from_generated(generated_text: str, prompt_fragment: str, sep: str = '||'):
    """Remove the prompt fragment if present and split on the separator or newlines."""
    txt = generated_text
    # If the model repeated the prompt exactly, remove the first occurrence conservatively
    if prompt_fragment and prompt_fragment in txt:
        txt = txt.split(prompt_fragment, 1)[-1]
    # try separator first
    if sep in txt:
        parts = [p.strip() for p in txt.split(sep) if p.strip()]
    else:
        # fallback: split on lines and extract lines that look like titles
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        parts = []
        for line in lines:
            # strip leading numbering like "1. " or "- "
            line = re.sub(r'^[\-\*\d\.\)\s]+', '', line).strip()
            if len(line.split()) >= 2:
                parts.append(line)
    # sanitize weird outputs: drop lines that are mostly non-letter characters
    valid = []
    for p in parts:
        letters = re.sub(r'[^A-Za-z]', '', p)
        if len(letters) >= max(3, int(0.25 * len(p))):  # require some alphabetic content
            valid.append(p)
    return valid

# ---------------------------
# Generation functions
# ---------------------------
def init_generator(model_name: str, device: int):
    print(f"Loading model: {model_name} (device={'GPU' if device==0 else 'CPU'}) ...")
    gen = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        device=device,
        pad_token_id=50256
    )
    return gen

def generate_topics(generator, description: str, n: int,
                    max_new_tokens: int, temperature: float, top_p: float, attempts: int = 3):
    """
    Try to generate n titles. We use a separator (||) to make parsing reliable.
    We will attempt multiple times if results are insufficient.
    """
    sep = "||"
    # Create a compact explicit prompt
    prompt = (
        f"Produce exactly {n} unique, concise, and engaging blog/article titles "
        f"for this website description. Return only titles separated by '{sep}' with no extra commentary.\n\n"
        f"Description: {description}\n\nTitles: "
    )

    for attempt in range(1, attempts + 1):
        try:
            # We request num_return_sequences = n to get multiple candidates in one call
            out = generator(
                prompt,
                max_new_tokens=120,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,  # one sampled completion; it should include all titles separated by sep
                truncation=True
            )
        except Exception as e:
            print(f"  Generation error (attempt {attempt}): {e}")
            continue

        # The pipeline returns a list of dicts when num_return_sequences>1; normalize to string
        if isinstance(out, list):
            gen_text = out[0].get("generated_text", "")
        else:
            gen_text = str(out)

        titles = parse_titles_from_generated(gen_text, prompt, sep=sep)
        if len(titles) >= n:
            return titles[:n]
        # If not enough, attempt alternative prompt that asks for one title per line
        alt_prompt = (
            f"Generate {n} concise blog/article titles (one per line) for the website described below. "
            f"Return only the titles, one per line, no commentary.\n\nDescription: {description}\n\n"
        )
        try:
            out2 = generator(
                alt_prompt,
                max_new_tokens=120,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                truncation=True
            )
            gen_text2 = out2[0].get("generated_text", "")
            titles2 = parse_titles_from_generated(gen_text2, alt_prompt, sep='\n')
            # merge any unique ones
            for t in titles2:
                if t not in titles:
                    titles.append(t)
            if len(titles) >= n:
                return titles[:n]
        except Exception:
            pass

        print(f"  Attempt {attempt} produced {len(titles)} titles; retrying...")

    # Final fallback: neutral placeholders (non-site-specific), only if absolutely necessary
    if not titles:
        titles = [f"Generated Article {i+1}" for i in range(n)]
    else:
        # If we have fewer than requested, pad with neutral placeholders
        while len(titles) < n:
            titles.append(f"Generated Article {len(titles)+1}")
    return titles[:n]

def generate_article(generator, topic: str, max_new_tokens: int, temperature: float, top_p: float, rep_penalty: float):
    prompt = (
        f"# {topic}\n\n"
        f"Write a clear, helpful, and structured article about \"{topic}\". Include an intro, "
        f"a few subpoints, practical tips, and a short conclusion. Use natural language and no extra meta commentary.\n\nArticle:"
    )
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        do_sample=True,
        truncation=True
    )
    gen_text = out[0].get("generated_text", "")
    # remove the prompt prefix if repeated
    if prompt in gen_text:
        body = gen_text.split(prompt, 1)[-1].strip()
    else:
        body = gen_text[len(prompt):].strip() if gen_text.startswith("#") else gen_text.strip()
    body = clean_text(body)
    # pretty-wrap paragraphs
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent="    ")
    formatted_lines = []
    for line in body.splitlines():
        if line.strip():
            formatted_lines.append(wrapper.fill(line.strip()))
        else:
            formatted_lines.append("")
    formatted_body = "\n".join(formatted_lines)
    return f"# {topic}\n\n{formatted_body}\n"

# ---------------------------
# CLI / Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate articles from a site description.")
    parser.add_argument("--desc", "-d", help="Site description (overrides SITE_DESC env)", default=os.getenv("SITE_DESC"))
    parser.add_argument("--topics", "-t", type=int, help="Number of topics to generate", default=DEFAULT_NUM_TOPICS)
    parser.add_argument("--model", "-m", help="Model name", default=os.getenv("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--out", "-o", help="Output directory", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-tokens", type=int, help="Max new tokens for article body", default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    description = args.desc or os.getenv("SITE_DESC") or "No description provided."
    num_topics = args.topics
    model_name = args.model
    out_dir = args.out
    max_new_tokens = args.max_tokens

    device = device_index()
    generator = init_generator(model_name, device)

    # set seed if provided
    seed_env = os.getenv("SEED")
    if seed_env:
        try:
            set_seed(int(seed_env))
            random.seed(int(seed_env))
        except Exception:
            pass

    print(f"Site description (len={len(description)}): {description[:200]}{'...' if len(description)>200 else ''}\n")
    topics = generate_topics(
        generator,
        description,
        num_topics,
        max_new_tokens=120,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        attempts=3
    )

    print(f"Generated {len(topics)} topics:\n" + "\n".join(f"  • {t}" for t in topics) + "\n")

    os.makedirs(out_dir, exist_ok=True)
    for i, topic in enumerate(topics, start=1):
        article = generate_article(
            generator,
            topic,
            max_new_tokens=max_new_tokens,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            rep_penalty=DEFAULT_REP_PENALTY
        )
        safe = sanitize_filename(topic)
        filename = os.path.join(out_dir, f"article_{i:02d}_{safe or 'topic_'+str(i)}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(article)
        print(f"  Saved: {filename}")

    print(f"\nAll {len(topics)} articles generated in '{out_dir}/'")

if __name__ == "__main__":
    main()
