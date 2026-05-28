"""One-off Playwright script to render the two HTML fixtures to PNG.

Run from the support_copilot_demo/assets directory:
    python make_assets.py
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def ensure_playwright():
    try:
        import playwright  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright>=1.40.0"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])


def render(html_name: str, png_name: str, viewport=(720, 360)):
    from playwright.sync_api import sync_playwright

    here = Path(__file__).parent
    html_url = (here / html_name).as_uri()
    png_path = here / png_name

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        page.goto(html_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()
    print(f"wrote {png_path}")


if __name__ == "__main__":
    ensure_playwright()
    render("bank_statement.html", "bank_statement.png", viewport=(620, 280))
    render("help_center_30day.html", "help_center_30day.png", viewport=(720, 320))
