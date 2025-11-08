import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
log_file = "topic_generation_logs.txt"
log_handle = open(log_file, "a", encoding="utf-8")


def log(msg: str):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()


log("Topic Generation Started – Flan-T5 (CPU) – UNIQUE TITLES")

# ----------------------------------------------------------------------
# SITE DESCRIPTION (fixed – you can also read it from env if you like)
# ----------------------------------------------------------------------
site_desc = """Jobs Mauritius is the leading online job portal in Mauritius, connecting employers with skilled talent across industries like IT, finance, tourism, and BPO. Featuring daily job listings, resume uploads, career advice, and company reviews, it helps Mauritian job seekers find local and remote opportunities fast."""

# ----------------------------------------------------------------------
# MODEL & TOKENIZER
# ----------------------------------------------------------------------
device = torch.device("cpu")
model_name = "google/flan-t5-large"          # you can switch to -base or -small
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)
log("Flan-T5-Large loaded")

# ----------------------------------------------------------------------
# PROMPT (strict – only a numbered list, no extra text)
# ----------------------------------------------------------------------
base_prompt = f"""
You are a career-blog editor for Jobs Mauritius.  
Generate **exactly 30** unique, SEO-friendly blog-post titles (questions or actionable phrases) that are directly inspired by the site description below.

Site description:
{site_desc}

Rules:
- Every title must be different.
- Return **only** a numbered list: 1. Title… 2. Title… etc.
- Do NOT add any intro, conclusion, or examples.
"""

log("Prompt ready")

# ----------------------------------------------------------------------
# GENERATE UNTIL WE HAVE 15 UNIQUE TITLES
# ----------------------------------------------------------------------
needed = 15
seen = set()
topics = []

log(f"Generating titles – need {needed} unique ones…")

while len(topics) < needed:
    inputs = tokenizer(
        base_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.9,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.3,
            num_return_sequences=1,
        )

    raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    log(f"Raw model output:\n{raw}\n{'-'*60}")

    # ---- extract numbered lines -------------------------------------------------
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # remove leading "1.", "2.", etc.
        if "." in line[:5]:
            candidate = line.split(".", 1)[1].strip(' "')
        else:
            candidate = line

        # clean up quotes / punctuation
        candidate = candidate.strip('."\' ')

        if candidate and candidate not in seen:
            seen.add(candidate)
            topics.append(candidate)
            log(f"Added: {candidate}")

        if len(topics) >= needed:
            break

# ----------------------------------------------------------------------
# FALLBACK (should never be needed, but keeps the script safe)
# ----------------------------------------------------------------------
if len(topics) < needed:
    for i in range(len(topics) + 1, needed + 1):
        fallback = f"Career Tips for Mauritius #{i}"
        topics.append(fallback)
        seen.add(fallback)
        log(f"Fallback added: {fallback}")

# ----------------------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------------------
output = {
    "site_description": site_desc,
    "generated_topics": topics[:needed],
    "total_unique_before_truncation": len(seen),
}

with open("generated_topics.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

with open("topics_list.txt", "w", encoding="utf-8") as f:
    for t in topics[:needed]:
        f.write(t + "\n")

log(f"Saved {len(topics[:needed])} unique topics → generated_topics.json & topics_list.txt")
log_handle.close()

# ----------------------------------------------------------------------
# PRETTY PRINT
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL 15 UNIQUE BLOG TOPICS")
print("=" * 60)
for i, t in enumerate(topics[:needed], 1):
    print(f"{i:2}. {t}")
print("=" * 60)
