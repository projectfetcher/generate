# article_generator.py
from transformers import pipeline
import textwrap
import random
import re
import os

# ================================================
# CONFIGURATION – CHANGE ONLY THIS SECTION
# ================================================

# Paste your site description here (one sentence)
SITE_DESCRIPTION = "Mauritius.MimusJobs.com is a specialized job portal dedicated to connecting employers and job seekers in Mauritius, featuring local vacancies in IT, finance, hospitality, and tourism with resume uploads and career tools."

# Model choice (smaller = faster, larger = better)
MODEL_NAME = "gpt2-medium"  # Options: gpt2, gpt2-medium, gpt2-large, distilgpt2

# Generation settings
MAX_LENGTH = 800
TEMPERATURE = 0.85
TOP_P = 0.92
REPETITION_PENALTY = 1.1
NUM_TOPICS_TO_GENERATE = 5
SEED = None  # Set to int for reproducibility

# Output folder
OUTPUT_DIR = "generated_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================
# END OF CONFIG – DO NOT EDIT BELOW
# ================================================

print(f"Loading model: {MODEL_NAME} ...")
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0 if os.getenv("USE_GPU") else -1,
    pad_token_id=50256
)

def clean_text(text: str) -> str:
    text = text.strip()
    # Remove incomplete last sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1 and not text.endswith(('.','!','?')):
        sentences = sentences[:-1]
    text = ' '.join(sentences)
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def generate_topics_from_description(description: str, n: int = 5) -> list[str]:
    prompt = f"""Based on this website description, suggest {n} unique, engaging blog article titles that would attract its target audience:

Description: {description}

Article Titles:
1. """
    
    print("Generating article topics from site description...")
    output = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        truncation=True
    )[0]["generated_text"]

    # Extract numbered list
    raw = output.split("Article Titles:")[1].strip()
    titles = []
    for line in raw.split('\n'):
        match = re.match(r'^\d+\.\s*(.+)', line.strip())
        if match:
            title = match.group(1).strip(' "\'')
            if title:
                titles.append(title)
        if len(titles) >= n:
            break
    return titles[:n] if titles else [f"Exploring {description.split()[0]} Opportunities"]

def generate_article(topic: str) -> str:
    prompt = f"""# {topic}

In today's fast-evolving landscape, {topic.lower()} plays a pivotal role. This article explores"""

    print(f"  Generating: {topic}")
    outputs = generator(
        prompt,
        max_length=MAX_LENGTH,
        num_return_sequences=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        truncation=True
    )

    raw = outputs[0]["generated_text"]
    article_body = raw[len(prompt):].strip()
    article_body = clean_text(article_body)

    wrapper = textwrap.TextWrapper(width=80, subsequent_indent="    ")
    formatted_lines = []
    for line in article_body.splitlines():
        if line.strip():
            formatted_lines.append(wrapper.fill(line.strip()))
        else:
            formatted_lines.append("")
    
    formatted_body = "\n".join(formatted_lines)
    return f"# {topic}\n\n{formatted_body}\n"

# ================================================
# MAIN EXECUTION
# ================================================

if __name__ == "__main__":
    if SEED:
        random.seed(SEED)

    print(f"Site: {SITE_DESCRIPTION}\n")
    
    # Step 1: Generate topics
    topics = generate_topics_from_description(SITE_DESCRIPTION, NUM_TOPICS_TO_GENERATE)
    print(f"Generated {len(topics)} topics:\n" + "\n".join(f"  • {t}" for t in topics) + "\n")

    # Step 2: Generate articles
    for i, topic in enumerate(topics, 1):
        article = generate_article(topic)
        
        safe_name = re.sub(r'[^\w\s-]', '', topic.lower())
        safe_name = re.sub(r'\s+', '_', safe_name)[:50]
        filename = f"{OUTPUT_DIR}/article_{i:02d}_{safe_name}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(article)
        
        print(f"  Saved: {filename}")

    print(f"\nAll {len(topics)} articles generated in '{OUTPUT_DIR}/'")
