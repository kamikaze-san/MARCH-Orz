import cv2
import numpy as np
import easyocr
import time

# Initialize the neural network ONCE at the top of the file.
# Setting gpu=True allows it to leverage your NVIDIA card for massive speed gains.
print("Loading EasyOCR Model into memory...")
reader = easyocr.Reader(['en'], gpu=True)

def find_text_and_click(image_bytes, target_word, qwen_x_norm, qwen_y_norm, search_radius=250):
    start_time = time.time()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    
    rough_x = int((qwen_x_norm / 1000.0) * width)
    rough_y = int((qwen_y_norm / 1000.0) * height)
    
    print(f"Scanning for '{target_word}' near actual pixel X:{rough_x}, Y:{rough_y}...")
    
    # 1. CROP (Crucial for speed! Running PyTorch on a 1080p image is slow. 
    # Running it on a tiny 500x500 cropped square is nearly instant.)
    x_min, y_min = max(0, rough_x - search_radius), max(0, rough_y - search_radius)
    x_max, y_max = min(width, rough_x + search_radius), min(height, rough_y + search_radius)
    roi = img[y_min:y_max, x_min:x_max]
    
    # 2. EasyOCR Magic
    # detail=1 returns the bounding box, the text, and the confidence score
    results = reader.readtext(roi, detail=1)
    
    target_word_lower = target_word.lower()
    best_x, best_y = None, None
    closest_distance = float('inf')
    
    for (bbox, text, prob) in results:
        found_text = text.lower()
        
        # Check if the target word is inside the found text block
        if target_word_lower in found_text and len(found_text) > 2:
            # bbox is a list of 4 coordinates: [top-left, top-right, bottom-right, bottom-left]
            tl_x, tl_y = bbox[0]
            br_x, br_y = bbox[2]
            
            # Find the exact center of the text inside the cropped image
            center_x_roi = int((tl_x + br_x) / 2)
            center_y_roi = int((tl_y + br_y) / 2)
            
            # Map back to global 1080p screen coordinates
            center_x = x_min + center_x_roi
            center_y = y_min + center_y_roi
            
            dist = ((center_x - rough_x) ** 2 + (center_y - rough_y) ** 2) ** 0.5
            
            if dist < closest_distance:
                closest_distance = dist
                best_x, best_y = center_x, center_y
                
    if best_x and best_y:
        print(f"-> EasyOCR Target locked at X:{best_x}, Y:{best_y} in {time.time() - start_time:.3f}s!")
        return best_x, best_y
        
    print(f"-> EasyOCR Could not find '{target_word}' near the intended target.")
    return None, None