"""
perception.py — State Extraction & UI Element Detection

Converts a raw screenshot into structured page state and a UI element inventory.
All functions accept a pre-decoded OpenCV image (decoded ONCE in the main loop)
to avoid redundant cv2.imdecode calls.

Public API:
    extract_page_state(img, width, height) -> dict
    detect_ui_elements(img, width, height) -> list[dict]
"""

import cv2
import numpy as np
import easyocr

# ──────────────────────────────────────────────
# Singleton OCR reader (loaded once at import)
# ──────────────────────────────────────────────
print("[Perception] Loading EasyOCR model …")
_ocr_reader = easyocr.Reader(['en'], gpu=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _classify_page_type(texts):
    """Simple heuristic page-type classifier based on visible text."""
    joined = " ".join(t.lower() for t in texts)
    if any(kw in joined for kw in ("sign in", "log in", "username", "password", "email")):
        return "login"
    if any(kw in joined for kw in ("search results", "showing results", "results for")):
        return "search_results"
    if any(kw in joined for kw in ("subscribe", "views", "watch", "video")):
        return "video"
    if any(kw in joined for kw in ("cart", "checkout", "add to cart", "buy now", "price")):
        return "shopping"
    if any(kw in joined for kw in ("home", "welcome", "explore", "trending")):
        return "homepage"
    return "unknown"


def _classify_element_type(text, bbox, img_region):
    """
    Heuristic element-type classifier.
    Uses aspect ratio, size, text content, and visual features.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    aspect = w / max(h, 1)
    text_lower = text.lower().strip()

    # Button heuristics: short text, compact bbox
    button_keywords = (
        "submit", "search", "sign in", "log in", "subscribe",
        "ok", "cancel", "next", "back", "done", "apply",
        "add to cart", "buy", "checkout", "close",
    )
    if text_lower in button_keywords or (len(text_lower) < 20 and 1.5 < aspect < 8 and h < 60):
        return "button"

    # Search bar: wide and thin, often empty or has placeholder text
    if aspect > 6 and h < 50:
        return "search_bar"

    # Text field: medium width, thin height
    if aspect > 3 and h < 45:
        return "text_field"

    # Link heuristics
    if text_lower.startswith("http") or text_lower.startswith("www"):
        return "link"

    # Default to text
    return "text"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def extract_page_state(img, width, height):
    """
    Convert a decoded screenshot into structured page state.

    Args:
        img:    Pre-decoded OpenCV image (np.ndarray, BGR).
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        dict with keys:
            page_type        — str  (login | search_results | video | shopping | homepage | unknown)
            visible_elements — list of {"text": str, "bbox": [x1, y1, x2, y2], "confidence": float}
            buttons          — subset where type == "button"
            text_elements    — subset where type == "text"
            search_boxes     — subset where type == "search_bar"
    """
    # Run OCR on full image
    ocr_results = _ocr_reader.readtext(img, detail=1)

    visible_elements = []
    buttons = []
    text_elements = []
    search_boxes = []

    for (bbox_pts, text, confidence) in ocr_results:
        if confidence < 0.25 or len(text.strip()) < 2:
            continue

        # bbox_pts: [top-left, top-right, bottom-right, bottom-left]
        tl_x, tl_y = int(bbox_pts[0][0]), int(bbox_pts[0][1])
        br_x, br_y = int(bbox_pts[2][0]), int(bbox_pts[2][1])
        bbox = [tl_x, tl_y, br_x, br_y]

        elem = {"text": text.strip(), "bbox": bbox, "confidence": round(confidence, 2)}
        visible_elements.append(elem)

        etype = _classify_element_type(text.strip(), bbox, img[tl_y:br_y, tl_x:br_x])
        if etype == "button":
            buttons.append(elem)
        elif etype == "search_bar":
            search_boxes.append(elem)
        else:
            text_elements.append(elem)

    texts = [e["text"] for e in visible_elements]
    page_type = _classify_page_type(texts)

    return {
        "page_type": page_type,
        "visible_elements": visible_elements,
        "buttons": buttons,
        "text_elements": text_elements,
        "search_boxes": search_boxes,
    }


def detect_ui_elements(img, width, height):
    """
    Detect UI elements via OpenCV contour analysis + OCR text overlay.

    Returns a list of element dicts:
        [{"id": int, "type": str, "text": str, "bbox": [x1, y1, x2, y2]}, ...]

    Element types: button, search_bar, text_field, link, text, image
    Coordinates are in real screen pixels.

    Args:
        img:    Pre-decoded OpenCV image (np.ndarray, BGR).
        width:  Image width in pixels.
        height: Image height in pixels.
    """
    elements = []
    element_id = 1

    # ── Phase 1: OpenCV contour-based element detection ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=120)
    dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track bboxes to avoid duplicates with OCR pass
    seen_bboxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter out noise (too small) and the full page (too large)
        if area < 600 or area > (width * height * 0.5):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]

        # Classify based on shape
        aspect = w / max(h, 1)
        if aspect > 5 and h < 50:
            etype = "search_bar"
        elif 1.5 < aspect < 6 and h < 60 and w < 300:
            etype = "button"
        elif aspect < 1.5 and area > 5000:
            etype = "image"
        else:
            etype = "text_field"

        elements.append({
            "id": element_id,
            "type": etype,
            "text": "",
            "bbox": bbox,
        })
        seen_bboxes.append(bbox)
        element_id += 1

    # ── Phase 2: Overlay OCR text onto detected elements ──
    ocr_results = _ocr_reader.readtext(img, detail=1)

    for (bbox_pts, text, confidence) in ocr_results:
        if confidence < 0.30 or len(text.strip()) < 2:
            continue

        tl_x, tl_y = int(bbox_pts[0][0]), int(bbox_pts[0][1])
        br_x, br_y = int(bbox_pts[2][0]), int(bbox_pts[2][1])
        ocr_cx, ocr_cy = (tl_x + br_x) // 2, (tl_y + br_y) // 2

        # Try to assign text to an existing contour element
        matched = False
        for elem in elements:
            ex1, ey1, ex2, ey2 = elem["bbox"]
            if ex1 <= ocr_cx <= ex2 and ey1 <= ocr_cy <= ey2:
                # Append text if element has none, else extend
                if elem["text"]:
                    elem["text"] += " " + text.strip()
                else:
                    elem["text"] = text.strip()
                # Reclassify now that we have text
                elem["type"] = _classify_element_type(
                    elem["text"], elem["bbox"],
                    img[ey1:ey2, ex1:ex2]
                )
                matched = True
                break

        # If no contour matched, add as a standalone text element
        if not matched:
            bbox = [tl_x, tl_y, br_x, br_y]
            etype = _classify_element_type(text.strip(), bbox, img[tl_y:br_y, tl_x:br_x])
            elements.append({
                "id": element_id,
                "type": etype,
                "text": text.strip(),
                "bbox": bbox,
            })
            element_id += 1

    return elements
