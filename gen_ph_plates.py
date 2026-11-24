"""Generate synthetic Philippine plate crops for LPRNet PH testing.

Styles:
- new: white background, black text (current PH private plates)
- old: green background, white text (older PH private plates)
Conditions for test sets: clean, dirty (mud smudges + noise + blur).
Filenames are the ground-truth plate text (the repo loader reads labels from filenames).
Plate format: 3 letters + 4 digits (current PH car plates).
"""
import os, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BASE = os.path.join(os.path.dirname(__file__), "ph_data")
W, H = 240, 60

def rand_plate(used):
    while True:
        p = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)) + \
            "".join(random.choice("0123456789") for _ in range(4))
        if p not in used:
            used.add(p)
            return p

def render(plate, style="new", dirty=False, train_var=False):
    if style == "new":
        bg = (random.randint(225, 255),) * 3
        fg = (10, 10, 10)
        border = (20, 20, 20)
    else:  # old green plate
        bg = (0, random.randint(95, 125), 55)
        fg = (240, 240, 235)
        border = (230, 230, 230)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([1, 1, W - 2, H - 2], outline=border, width=2)
    text = plate[:3] + " " + plate[3:]
    font = ImageFont.truetype(FONT, 38)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0] + (random.randint(-3, 3) if train_var else 0)
    y = (H - th) // 2 - bbox[1] + (random.randint(-2, 2) if train_var else 0)
    d.text((x, y), text, font=font, fill=fg)
    if train_var:
        ang = random.uniform(-2.5, 2.5)
        img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=bg)
        if random.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
        if random.random() < 0.30:
            add_dirt(img, light=True)
    if dirty:
        add_dirt(img, light=False)
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.3)))
    return img

def add_dirt(img, light=False):
    d = ImageDraw.Draw(img, "RGBA")
    n = random.randint(2, 4) if light else random.randint(5, 10)
    for _ in range(n):
        x, y = random.randint(0, W - 20), random.randint(0, H - 12)
        r = random.randint(4, 14) if light else random.randint(6, 22)
        mud = (101, 78, 50, random.randint(90, 170))
        d.ellipse([x, y, x + r * 2, y + r], fill=mud)
    px = img.load()
    for _ in range(300 if light else 900):
        x, y = random.randint(0, W - 1), random.randint(0, H - 1)
        v = random.randint(0, 90)
        px[x, y] = (v, v, v)

def make(split, count_new, count_old, dirty=False):
    used = make.used
    out = os.path.join(BASE, split)
    os.makedirs(out, exist_ok=True)
    for _ in range(count_new):
        p = rand_plate(used)
        render(p, "new", dirty=dirty, train_var=(split == "train")).save(os.path.join(out, p + ".jpg"), quality=92)
    for _ in range(count_old):
        p = rand_plate(used)
        render(p, "old", dirty=dirty, train_var=(split == "train")).save(os.path.join(out, p + ".jpg"), quality=92)
    print(split, "->", len(os.listdir(out)), "images")

make.used = set()
make("train", 800, 200)          # 1000 training plates
make("test_clean", 200, 0)       # new, clean
make("test_dirty", 200, 0)       # new, dirty/blurry
make("test_old", 0, 120)         # old green plates, clean
print("total unique plates:", len(make.used))
