"""Screenshots for the README: one Fashion-MNIST run, late enough in the reverse process to read as a garment.

    python scripts/readme_shots.py --url http://127.0.0.1:8105/ [--seed 1] [--frac 0.9]

Shows the three panes at the chosen point of a recorded run, in halftone and in pixels mode, at desktop and phone size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def shoot(page, label_has: str, seed: int, frac: float, mode: str, path: Path) -> None:
    page.locator("#model-chips .chip", has_text=label_has).first.click()
    page.wait_for_timeout(300)
    page.locator("#seed-chips .chip", has_text=f"seed {seed}").first.click()
    page.wait_for_timeout(300)
    page.locator("#render-chips .chip", has_text=mode).first.click()
    page.click("#play") if page.evaluate("document.querySelector('#play-ic').getAttribute('d').startsWith('M6')") else None
    page.evaluate(
        f"""() => {{ const r = document.querySelector('#scrub-range'); r.value = Math.round(+r.max * {frac});
        r.dispatchEvent(new Event('input')); }}"""
    )
    page.wait_for_timeout(400)
    page.locator("#stage").scroll_into_view_if_needed()
    page.locator("#stage").screenshot(path=str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8105/")
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    ap.add_argument("--model", default="linear", help="text of the model chip, e.g. linear or cosine")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frac", type=float, default=0.9)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, args in {
            "desktop": dict(viewport={"width": 1440, "height": 900}),
            "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
        }.items():
            page = b.new_context(**args).new_page()
            page.goto(a.url, wait_until="networkidle")
            page.wait_for_function("window.__ready === true", timeout=30000)
            page.add_style_tag(content=".top { display: none !important }")  # the sticky bar would cover the panes in an element shot
            for mode in ("halftone", "pixels"):
                shoot(page, a.model, a.seed, a.frac, mode, out / f"{name}-stage-{mode}.png")
        b.close()


if __name__ == "__main__":
    main()
