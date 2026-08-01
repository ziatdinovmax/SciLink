"""Wide figures get their own landscape page in the PDF.

A schematic squeezed into a portrait text column renders about an inch
tall and its labels are unreadable; this checks it is pulled out of the
flow and given the full page width instead.

  conda run -n scilink python tests/test_md_to_pdf_figures.py
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import fitz
from PIL import Image

from scilink.utils.md_to_pdf import markdown_to_pdf, _pull_wide_figures

results = {}


def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def make_png(path, w, h):
    Image.new("RGB", (w, h), "white").save(path)


with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    make_png(d / "wide.png", 3000, 500)      # 6:1
    make_png(d / "square.png", 800, 700)     # 1.14:1
    md = d / "doc.md"
    md.write_text("# Doc\n\nIntro text.\n\n![Wide schematic](wide.png)\n\n"
                  "![Small plot](square.png)\n")

    text, figs = _pull_wide_figures(md.read_text(), d)
    check("wide figure pulled from flow", len(figs) == 1
          and figs[0][0].name == "wide.png")
    check("caption preserved", figs[0][1] == "Wide schematic")
    check("narrow figure left inline", "square.png" in text)
    check("wide figure removed from text", "wide.png" not in text)

    pdf = markdown_to_pdf(md, title="Doc")
    doc = fitz.open(pdf)
    landscape = [p for p in doc if p.rect.width > p.rect.height]
    check("a landscape page was appended", len(landscape) == 1,
          f"({len(doc)} pages)")
    if landscape:
        pg = landscape[0]
        check("figure is on it", len(pg.get_images()) == 1)
        info = pg.get_image_info()
        if info:
            w = info[0]["bbox"][2] - info[0]["bbox"][0]
            check("figure uses the full page width", w > 600,
                  f"({w:.0f}pt = {w/72:.1f}in)")
    portrait_imgs = sum(len(p.get_images()) for p in doc
                        if p.rect.width <= p.rect.height)
    check("narrow figure still inline in the text", portrait_imgs >= 1)


with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    make_png(d / "tall.png", 700, 3000)      # 0.23:1
    md = d / "tall.md"
    md.write_text("# Doc\n\nText.\n\n![Tall chain](tall.png)\n")
    text, figs = _pull_wide_figures(md.read_text(), d)
    check("tall figure also pulled", len(figs) == 1)
    doc = fitz.open(markdown_to_pdf(md, title="Doc"))
    own = [p for p in doc if len(p.get_images()) == 1]
    check("tall figure gets its own page", len(own) == 1, f"({len(doc)} pages)")
    if own:
        r = own[0].rect
        check("that page is portrait", r.height > r.width,
              f"({r.width:.0f}x{r.height:.0f})")

with tempfile.TemporaryDirectory() as t:
    d = Path(t)
    md = d / "plain.md"
    md.write_text("# Plain\n\nNo figures at all.\n")
    doc = fitz.open(markdown_to_pdf(md, title="Plain"))
    check("no spurious pages without figures",
          all(p.rect.width <= p.rect.height for p in doc))

print("=" * 50)
n = sum(results.values())
print(f"PDF FIGURE PAGES: {n}/{len(results)} checks passed")
if n != len(results):
    raise SystemExit(1)
