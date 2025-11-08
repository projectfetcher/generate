import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# LOGGING
log_file = "topic_generation_logs.txt"
log_handle = open(log_file, "a", encoding="utf-8")

def log(msg):
    print(msg)
    log_handle.write(msg + "\n")
    log_handle.flush()

log("Topic Generation Started – Flan-T5 (CPU)")

# INPUT: Site Description
site_desc = """Jobs Mauritius is the leading online job portal in Mauritius, connecting employers with skilled talent across industries like IT, finance, tourism, and BPO. Featuring daily job listings, resume uploads, career advice, and company reviews, it helps Mauritian job seekers find local and remote opportunities fast."""

# MODEL & TOKENIZER
device = torch.device("cpu")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)
log("Flan-T5-Large loaded")

# PROMPT TO GENERATE 15 HIGH-QUALITY BLOG TOPICS
prompt = f"""
Based on this site description, generate 15 engaging, practical, and SEO-friendly blog post titles for a career advice blog on Jobs Mauritius.

Site description:
{site_desc}

Rules:
- Each title must be a question or actionable phrase.
- Focus on job search, career growth, resume tips, interviews, industry trends, remote work, or Mauritius-specific opportunities.
- Make them appealing to job seekers in IT, finance, tourism, and BPO.
- Include local relevance (Mauritius) where natural.
- Do NOT include examples or explanations — only return the list of 15 titles, one per line.
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

log("Generating 15 blog topics...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )

raw_topics = tokenizer.decode(output[0], skip_special_tokens=True).strip()
log(f"Raw output:\n{raw_topics}")

# CLEAN AND EXTRACT TOPICS
topics = []
for line in raw_topics.split("\n"):
    line = line.strip()
    if line and not line.lower().startswith(("here", "list", "1.", "title")):
        # Remove numbering if present
        clean = line.split(".", 1)[-1].strip(' "')
        if clean:
            topics.append(clean)

# Ensure exactly 15 topics
if len(topics) > 15:
    topics = topics[:15]
elif len(topics) < 15:
    topics += [f"Career Tips in Mauritius #{i}" for i in range(len(topics)+1, 16)]

# SAVE TOPICS
output = {
    "site_description": site_desc,
    "generated_topics": topics
}

with open("generated_topics.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

log(f"Generated {len(topics)} topics. Saved to generated_topics.json")

# ALSO SAVE AS PLAIN LIST FOR EASY USE
with open("topics_list.txt", "w", encoding="utf-8") as f:
    for t in topics:
        f.write(t + "\n")

log("Topic list saved to topics_list.txt")
log_handle.close()

print("\n" + "="*50)
print("GENERATED TOPICS:")
print("="*50)
for i, t in enumerate(topics, 1):
    print(f"{i:2}. {t}")
print("="*50)
