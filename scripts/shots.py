"""Load the UI in headless Chromium at desktop and phone sizes, save screenshots, report console errors.

    python scripts/shots.py [--url http://127.0.0.1:8105/] [--out docs/assets] [--live]

Also checks there is no horizontal scroll at phone width and that the scrubber moves the panes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

SECTIONS = ["scrub", "schedules", "compare", "forward", "toy", "training", "maths"]


def run(url: str, out: Path, live: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"console_errors": [], "page_errors": [], "checks": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, ctx_args in {
            "desktop": dict(viewport={"width": 1440, "height": 900}),
            "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True),
        }.items():
            ctx = browser.new_context(**ctx_args)
            page = ctx.new_page()
            page.on("console", lambda m, n=name: m.type == "error" and report["console_errors"].append(f"{n}: {m.text}"))
            page.on("pageerror", lambda e, n=name: report["page_errors"].append(f"{n}: {e}"))
            page.on("requestfailed", lambda r, n=name: report["console_errors"].append(f"{n}: request failed {r.url}"))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.__ready === true", timeout=30000)
            page.wait_for_timeout(1500)
            report["checks"][f"{name}_badge"] = page.inner_text("#mode-badge")
            sw = page.evaluate("document.documentElement.scrollWidth"), page.evaluate("window.innerWidth")
            report["checks"][f"{name}_no_hscroll"] = sw[0] <= sw[1] and (name != "phone" or sw[1] == 390)  # 390: the page did not zoom out to fit something wide
            if name == "phone":  # tap targets: every visible control should be at least 44 px in both directions
                small = page.evaluate(
                    "[...document.querySelectorAll('button, a.btn, input[type=range], select')].filter(e => e.offsetParent)"
                    ".map(e => { const r = e.getBoundingClientRect(); return [e.id || e.textContent.trim().slice(0, 18), Math.round(r.width), Math.round(r.height)]; })"
                    ".filter(([, w, h]) => w < 44 || h < 44)"
                )
                report["checks"]["phone_small_tap_targets"] = small
            # scrubber: park at the middle and at the end, make sure the readout moves
            page.click("#play")  # pause if playing
            page.evaluate(
                "document.querySelector('#scrub-range').value = 0; document.querySelector('#scrub-range').dispatchEvent(new Event('input'))"
            )
            t0 = page.inner_text("#r-t")
            mx = page.evaluate("+document.querySelector('#scrub-range').max")
            page.evaluate(
                f"document.querySelector('#scrub-range').value = {int(mx * 0.72)}; document.querySelector('#scrub-range').dispatchEvent(new Event('input'))"
            )
            page.wait_for_timeout(200)
            t1 = page.inner_text("#r-t")
            report["checks"][f"{name}_scrub_moves_t"] = t0 != t1
            page.evaluate("document.getElementById('stage').scrollIntoView()")
            page.wait_for_timeout(250)
            page.screenshot(path=str(out / f"{name}-hero-mid.png"))
            page.evaluate(
                f"document.querySelector('#scrub-range').value = {mx}; document.querySelector('#scrub-range').dispatchEvent(new Event('input'))"
            )
            page.wait_for_timeout(200)
            page.screenshot(path=str(out / f"{name}-hero-end.png"))
            for sec in SECTIONS[1:]:
                page.evaluate(f"document.getElementById('{sec}').scrollIntoView()")
                page.wait_for_timeout(1400 if sec == "toy" else 700)
                page.screenshot(path=str(out / f"{name}-{sec}.png"))
            if live and name == "desktop":
                page.evaluate("document.getElementById('scrub').scrollIntoView()")
                if page.is_visible("#live-btn"):
                    page.click("#live-btn")
                    page.wait_for_function(
                        "document.querySelector('#run-note').textContent.startsWith('live run')", timeout=120000
                    )
                    report["checks"]["live_run_ok"] = True
                    page.screenshot(path=str(out / "desktop-live.png"))
            ctx.close()
        browser.close()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8105/")
    ap.add_argument("--out", default="shots")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.url, Path(a.out), a.live), indent=1))
