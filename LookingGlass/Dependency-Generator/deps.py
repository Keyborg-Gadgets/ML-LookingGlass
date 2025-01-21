import os
import asyncio
from playwright.async_api import async_playwright
import json

async def run(playwright):
    browser = await playwright.chromium.connect_over_cdp("http://localhost:9223")
    context = browser.contexts[0]
    page = context.pages[0]  # Assuming the first page is the one we want

    # Enable CDP network domain
    client = await context.new_cdp_session(page)
    await client.send("Network.enable")

    # List to store all network requests
    network_requests = []

    # Event handler for network requests
    async def on_request(event):
        network_requests.append(event)

    client.on("Network.requestWillBeSent", on_request)
    # client.on("Network.responseReceived", on_request)

    await page.wait_for_url("https://developer.nvidia.com/", timeout=0)

    print("Login successful")

    download_path = os.getenv('DOWNLOAD_PATH', os.getcwd())

    await page.goto("https://developer.nvidia.com/nvidia-tensorrt-8x-download")
    await page.click('//*[@id="content"]/div/section/form/b/input')
    await page.click('//*[@id="accordion"]/div[1]/div[1]/a')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    har_file_path = os.path.join(script_dir, "network_requests.har")
    print("doot")
    print(har_file_path)
    await page.route_from_har(har_file_path, update=True)

    async def intercept_route(route, request):
        if "hmac" in request.url:
            print(f"URL: {request.url}")
            print("Headers:")
            for header, value in request.headers.items():
                print(f"{header}: {value}")
            await route.abort()  # Cancel the download
        else:
            await route.continue_()
       

    await page.route("**/*", intercept_route)

    await page.click('//*[@id="trt861"]/div/ul[3]/li[2]/a')

    await page.wait_for_timeout(1000)  

    har_data = {
        "log": {
            "version": "1.2",
            "creator": {
                "name": "Playwright",
                "version": "1.0"
            },
            "entries": network_requests
        }
    }

    with open(har_file_path, "w") as har_file:
        json.dump(har_data, har_file)

    print(f"HAR file saved as {har_file_path}")


async def main():
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == "__main__":
    asyncio.run(main())