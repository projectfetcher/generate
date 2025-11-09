# article_generator.py
from transformers import pipeline, set_seed
import textwrap
import random
import re
import os
import torch

# ================================================
# CONFIGURATION
# ================================================
SITE_DESCRIPTION = (
    "This website connects professionals and organizations through informative content "
    "and valuable insights. It covers topics relevant to its target audience and aims "
    "to inspire growth, learning, and collaboration."
)

MODEL_NAME = "gpt2-medium"  # Options: gpt2, gpt2-medium, gpt2-large, distilgpt2

# Generation settings
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.85
TOP_P = 0.92
REPETITION_PENALTY = 1.1
NUM_TOPICS_TO_GENERATE = 5
SEED = None  # Set to int for reproducibility

OUTPUT_DIR = "generated_articles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================
# INITIALIZE MODEL
# ================================================
device = 0 if torch.cuda.is_available() and os.getenv("USE_GPU") else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

print(f"Loading model: {MODEL_NAME} ...")
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=device,
    pad_token_id=50256
)

if SEED:
    set_seed(SEED)
    random.seed(SEED)

# ================================================
# UTILITIES
# ================================================
def clean_text(text: str) -> str:
    """Remove incomplete last sentences and stray tokens."""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and not text.endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    text = ' '.join(sentences).strip()
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def generate_topics_from_description(description: str, n: int = 5) -> list[str]:
    """Generate a few potential blog post titles from the site description."""
    prompt = (
        f"Generate {n} unique and engaging blog article titles for a website with the following description:\n\n"
        f"{description}\n\n"
        f"Titles:\n"
    )

    print("Generating article topics from site description...")
    output = generator(
        prompt,
        max_new_tokens=120,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05
    )[0]["generated_text"]

    lines = re.split(r'[\n\r]+', output)
    titles = []
    for line in lines:
        m = re.match(r'^\s*(?:\d+[\).\s-]*)?["“”]*(.+?)["“”]*\s*$', line)
        if m:
            title = m.group(1).strip()
            if len(title.split()) >= 3 and len(title) < 120 and not any(bad in title for bad in ['<', '>', '{', '}', '#']):
                titles.append(title)
        if len(titles) >= n:
            break

    # If model fails completely, return minimal safe placeholders
    if not titles:
        titles = [f"Generated Article {i+1}" for i in range(n)]

    return titles[:n]


def generate_article(topic: str) -> str:
    """Generate a full article for a given topic."""
    prompt = (
        f"# {topic}\n\n"
        f"In today's fast-evolving landscape, {topic.lower()} plays an important role. "
        f"This article explores key insights, practical examples, and relevant ideas related to {topic.lower()}."
    )

    print(f"  Generating: {topic}")
    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True
    )[0]["generated_text"]

    article_body = output[len(prompt):].strip()
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
    print(f"Site: {SITE_DESCRIPTION}\n")

    topics = generate_topics_from_description(SITE_DESCRIPTION, NUM_TOPICS_TO_GENERATE)
    print(f"Generated {len(topics)} topics:\n" + "\n".join(f"  • {t}" for t in topics) + "\n")

    for i, topic in enumerate(topics, 1):
        article = generate_article(topic)
        safe_name = re.sub(r'[^\w\s-]', '', topic.lower())
        safe_name = re.sub(r'\s+', '_', safe_name)[:50]
        filename = f"{OUTPUT_DIR}/article_{i:02d}_{safe_name}.md"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(article)
        print(f"  Saved: {filename}")

    print(f"\nAll {len(topics)} articles generated in '{OUTPUT_DIR}/'")
