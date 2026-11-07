"""Smoke-check the public Streamlit dashboard for wake and render failures."""

from __future__ import annotations

from dataclasses import dataclass


APP_URL = "https://deadinternetobservatory.streamlit.app/"
EXPECTED_MARKERS = (
    "Dead Internet Observatory",
    "Documents scored · cumulative",
)
FATAL_CONTENT_MARKERS = (
    "Document count could not be loaded",
    "Published dataset snapshot could not be loaded",
)
AUTH_MARKERS = ("sign in", "log in", "authentication")


@dataclass
class DashboardSnapshot:
    final_url: str
    text: str


def evaluate_dashboard_text(text: str, *, final_url: str) -> dict:
    lowered_text = text.lower()
    lowered_url = final_url.lower()
    if "/-/login" in lowered_url or "share.streamlit.io/-/auth/app" in lowered_url:
        raise RuntimeError(f"Dashboard requires authentication at {final_url}")
    if any(marker in lowered_text for marker in AUTH_MARKERS):
        raise RuntimeError("Dashboard appears to require authentication")
    for marker in FATAL_CONTENT_MARKERS:
        if marker.lower() in lowered_text:
            raise RuntimeError(f"Dashboard rendered fatal content marker: {marker}")
    if not all(marker in text for marker in EXPECTED_MARKERS):
        raise RuntimeError(
            "Dashboard did not render expected dashboard markers "
            f"({', '.join(EXPECTED_MARKERS)})"
        )
    return {"ok": True, "final_url": final_url}


def _capture_dashboard(url: str = APP_URL) -> DashboardSnapshot:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        wake_button = page.get_by_role("button", name="Yes, get this app back up!")
        try:
            wake_button.click(timeout=10000)
            print("Clicked Streamlit wake button")
        except PlaywrightTimeoutError:
            print("Wake button not present")

        page.wait_for_timeout(10000)
        frame = next(
            (item for item in page.frames if item.url.rstrip("/").endswith("/~/+")),
            page.main_frame,
        )
        text = frame.locator("body").inner_text(timeout=60000)
        final_url = frame.url or page.url
        browser.close()
        return DashboardSnapshot(final_url=final_url, text=text)


def main() -> None:
    snapshot = _capture_dashboard()
    result = evaluate_dashboard_text(snapshot.text, final_url=snapshot.final_url)
    print("Dashboard data check passed")
    print(f"Final URL: {result['final_url']}")


if __name__ == "__main__":
    main()
