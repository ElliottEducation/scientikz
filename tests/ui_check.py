"""Manual UI smoke test against a running app (streamlit run app.py --server.port 8599)."""
import sys

from playwright.sync_api import sync_playwright

URL = "http://localhost:8599"
KEY = sys.argv[1]
OUT = "/tmp/ui"


def wait(page):
    page.wait_for_timeout(600)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=30000)
    page.wait_for_timeout(500)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL); page.wait_for_selector("text=ScienTikZ"); wait(page)
    page.screenshot(path=f"{OUT}-1-home.png", full_page=True)

    # search -> incline at 45 degrees
    page.get_by_label("🔍 搜索 / Search").fill("斜面 45度"); page.keyboard.press("Enter"); wait(page)
    print("incline heading:", page.locator("h3").first.inner_text())
    print("theta=45 in code:", "rotate=45" in page.locator("[data-testid='stCode']").first.inner_text())

    # formula search -> function graph
    page.get_by_label("🔍 搜索 / Search").fill("y = x^3 - 3x"); page.keyboard.press("Enter"); wait(page)
    print("function heading:", page.locator("h3").first.inner_text())
    print("formula in code:", "(\\x)^3 - 3*\\x" in page.locator("[data-testid='stCode']").first.inner_text())

    # locked diagram
    page.get_by_text("开普勒椭圆轨道 · Kepler orbit").click(); wait(page)
    print("locked notice shown:", page.get_by_text("完整源码与").count() > 0)
    print("PDF button disabled:", page.get_by_role("button", name="⬇ PDF").is_disabled())

    # licence
    page.get_by_label("授权码 / Licence key").fill(KEY); page.keyboard.press("Enter"); wait(page)
    print("unlocked:", page.get_by_text("Pro 已解锁").count() > 0,
          "| PDF enabled:", not page.get_by_role("button", name="⬇ PDF").is_disabled())
    page.screenshot(path=f"{OUT}-2-kepler-pro.png", full_page=True)

    # bad input -> friendly error, no crash
    page.get_by_text("函数图像 · Function graph").click(); wait(page)
    print("formula kept after switching:", page.get_by_role("textbox", name="f(x) =").input_value())
    page.get_by_role("textbox", name="f(x) =").fill("import os"); page.keyboard.press("Enter"); wait(page)
    print("error shown:", page.locator("[data-testid='stAlert']").first.inner_text()[:60])
    page.get_by_role("textbox", name="f(x) =").fill("1/x"); page.keyboard.press("Enter"); wait(page)
    page.screenshot(path=f"{OUT}-3-hyperbola.png", full_page=True)

    mobile = b.new_page(viewport={"width": 390, "height": 844})
    mobile.goto(URL); mobile.wait_for_selector("text=TikZ"); wait(mobile)
    mobile.screenshot(path=f"{OUT}-4-mobile.png")
    print("page errors:", errors)
    b.close()
