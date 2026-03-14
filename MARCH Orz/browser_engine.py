from playwright.sync_api import sync_playwright
from PIL import Image
import io
import base64
import time

class VisionBrowser:
    def __init__(self, headless=False):
        print("--- Booting VisionBrowser Engine ---")
        self.playwright = sync_playwright().start()
        
        # FIX: Force the physical window to be 1080p, and disable info bars
        self.browser = self.playwright.chromium.launch(
            headless=headless,
            args=[
                '--window-size=1920,1080',
                '--disable-infobars',
                '--force-device-scale-factor=1' # Kills Windows UI scaling interference
            ]
        )
        
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1 
        )
        self.page = self.context.new_page()
        print("Browser initialized at strict 1920x1080.")

    def navigate(self, url):
        print(f"Navigating to {url}...")
        # networkidle ensures the page actually loads before we take a screenshot
        self.page.goto(url, wait_until="networkidle") 
        
    def capture_vision_state(self, resize_dim=(512, 512)):
        """
        Captures the screen entirely in RAM.
        Returns the Base64 string for the LLM, and the raw image bytes for OpenCV.
        Zero hard-drive I/O.
        """
        # 1. Take native screenshot in RAM
        raw_bytes = self.page.screenshot(type='jpeg', quality=100)
        
        # 2. Open in Pillow directly from memory
        image = Image.open(io.BytesIO(raw_bytes))
        
        # 3. Downscale for the fast 4B model
        image.thumbnail(resize_dim, Image.Resampling.LANCZOS)
        
        # 4. Save back to a memory buffer and encode for Ollama
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", optimize=True, quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return b64_str, raw_bytes 

    def click(self, x, y):
        print(f"Executing DOM-independent click at X:{x}, Y:{y}")
        self.page.mouse.click(x, y)
        
    def type_text(self, text):
        print(f"Typing text: '{text}'")
        # Slight delay mimics human typing so bot-detectors don't freak out
        self.page.keyboard.type(text, delay=50) 
        self.page.keyboard.press("Enter")

    def close(self):
        print("Shutting down browser...")
        self.browser.close()
        self.playwright.stop()

# --- Quick Sanity Test ---
if __name__ == "__main__":
    agent = VisionBrowser(headless=False)
    
    agent.navigate("https://www.youtube.com/@PewDiePie/videos")
    time.sleep(2) # Give thumbnails an extra second to pop in
    
    # Test the RAM screenshot
    llm_b64, cv2_bytes = agent.capture_vision_state()
    print(f"Screenshot captured in RAM. Base64 string length: {len(llm_b64)}")
    
    # Test a blind click right in the middle of the screen
    agent.click(960, 540)
    time.sleep(3)
    
    agent.close()