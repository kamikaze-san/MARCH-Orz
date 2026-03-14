import cv2
import numpy as np
import ollama
import re
from browser_engine import VisionBrowser

# --- 1. The OpenCV Sniper ---
def snap_to_element(image_bytes, qwen_x_norm, qwen_y_norm, search_radius=60):
    """Takes RAM image bytes and normalized coordinates, returns exact pixel."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    
    rough_x = int((qwen_x_norm / 1000.0) * width)
    rough_y = int((qwen_y_norm / 1000.0) * height)
    
    x_min, y_min = max(0, rough_x - search_radius), max(0, rough_y - search_radius)
    x_max, y_max = min(width, rough_x + search_radius), min(height, rough_y + search_radius)
    roi = img[y_min:y_max, x_min:x_max]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return rough_x, rough_y
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        local_cx, local_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        local_cx, local_cy = x + w // 2, y + h // 2
        
    return x_min + local_cx, y_min + local_cy

# --- 2. The 4B Brain ---
def get_action_coordinate(prompt, base64_image):
    """Passes the RAM-based image to Qwen 4B."""
    system_prompt = (
        "Output ONLY bounding box coordinates for the requested UI element. "
        "Format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>"
    )
    
    response = ollama.chat(
        model='qwen3-vl:4b-instruct', # USING THE FAST 4B MODEL
        messages=[
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user', 
                'content': prompt, 
                # Passing the Base64 image directly. No hard drive required.
                'images': [base64_image] 
            }
        ],
        options={'temperature': 0.0}
    )
    
    raw_text = response['message']['content'].strip()
    print(f"   [Qwen 4B Output]: {raw_text}")
    
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', raw_text)
    if not match: 
        match = re.search(r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', raw_text)
        
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return (x1 + x2) // 2, (y1 + y2) // 2
    return None, None

# --- 3. The Execution Loop ---
if __name__ == "__main__":
    agent = VisionBrowser(headless=False)
    
    print("\n--- Starting Autonomous Action ---")
    agent.navigate("https://www.youtube.com/@PewDiePie/videos")
    
    print("\nCapturing screen state...")
    llm_b64, cv_bytes = agent.capture_vision_state()
    
    user_goal = "click the latest video"
    print(f"Asking Qwen 4B to: '{user_goal}'")
    
    rough_x, rough_y = get_action_coordinate(user_goal, llm_b64)
    
    if rough_x and rough_y:
        print("Snapping to exact UI element with OpenCV...")
        exact_x, exact_y = snap_to_element(cv_bytes, rough_x, rough_y)
        
        agent.click(exact_x, exact_y)
        print("Action Complete! Leaving browser open for 5 seconds to verify...")
        agent.page.wait_for_timeout(5000)
    else:
        print("AI failed to find the target.")
        
    agent.close()