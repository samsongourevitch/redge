import json, random, re
from urllib.request import urlopen
from bs4 import BeautifulSoup

URL = "https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/"

# --- load class map from the webpage ---
html = urlopen(URL, timeout=60).read()
soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
rows = table.find_all("tr")[1:]

classes = []
for r in rows:
    tds = r.find_all("td")
    if len(tds) != 2:
        continue
    cid = int(tds[0].get_text(strip=True))
    name = tds[1].get_text(" ", strip=True)
    name = re.sub(r"\s+", " ", name).strip()
    classes.append((cid, name))

assert len(classes) == 1000, f"Expected 1000 classes, got {len(classes)}"

def primary_label(full_name: str) -> str:
    # keep first comma-separated name as the main label
    return full_name.split(",")[0].strip().strip("'").strip()

def article(label: str) -> str:
    w = label.strip().lower()
    if re.match(r"^(honest|hour|heir)\b", w): return "an"
    if re.match(r"^(university|unicorn|european|one)\b", w): return "a"
    return "an" if w[:1] in "aeiou" else "a"

# CLIP / ImageNet-friendly attributes: color + simple size adjective
colors = ["red", "blue", "green", "yellow", "black", "white", "silver", "gold"]
sizes = ["small", "big"]

# keep prompts caption-like and image-grounded ("photo of ...")
styles = [
    "a photo of {art} {label}",
    "a photo of {art} {label} on a plain background",
    "a photo of {art} {label} in natural light",
    "a photo of {art} {label} outdoors",
    "a close-up photo of {art} {label}",
    "a clean studio photo of {art} {label}",
    "a photo of {art} {label}, high detail",
    "a photo of {art} {label} with soft lighting",
    "a photo of {art} {label} with a shallow depth of field",
]

# color templates (often)
color_styles = [
    "a photo of a {color} {label}",
    "a photo of a {color} {label} on a plain background",
    "a photo of a {color} {label} in natural light",
    "a close-up photo of a {color} {label}",
    "a photo of a {color} {label}, high detail",
]

# size templates (sometimes)
size_styles = [
    "a photo of a {size} {label}",
    "a photo of a {size} {label} on a plain background",
    "a close-up photo of a {size} {label}",
    "a clean studio photo of a {size} {label}",
]

# color + size templates (rare but useful)
color_size_styles = [
    "a photo of a {color}, {size} {label}",
    "a close-up photo of a {color}, {size} {label}",
    "a photo of a {color}, {size} {label} on a plain background",
]

rng = random.Random(0)

def make_prompt(full_name: str) -> str:
    label = primary_label(full_name)
    art = article(label)

    u = rng.random()
    # ~45% color, ~25% size, ~10% color+size, rest plain photo styles
    if u < 0.10:
        tpl = rng.choice(color_size_styles)
        return tpl.format(color=rng.choice(colors), size=rng.choice(sizes), label=label)
    elif u < 0.55:
        tpl = rng.choice(color_styles)
        return tpl.format(color=rng.choice(colors), label=label)
    elif u < 0.80:
        tpl = rng.choice(size_styles)
        return tpl.format(size=rng.choice(sizes), label=label)
    else:
        tpl = rng.choice(styles)
        return tpl.format(art=art, label=label)

# build prompts
N = 300
items = []
for _ in range(N):
    cid, full = rng.choice(classes)  # with replacement
    items.append({
        "class_id": cid,
        "class_name": full,
        "prompt": make_prompt(full),
    })

out = "imagenet_prompts_300.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

print(f"Wrote {out} with {len(items)} entries.")
print("Example entries:")
for ex in items[:5]:
    print(ex)