from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

# Stand-in for a real-time camera. To use a real camera (OpenCV):
#   cap = cv2.VideoCapture(0)
#   ret, frame = cap.read()                    # frame: HxWx3 BGR uint8
#   image = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
IMAGE_PATH = "/home/hea4sgh/rf-detr/000000039769.jpg"


def get_frame():
    """Grab one frame from the camera (here: load a local image)."""
    return Image.open(IMAGE_PATH).convert("RGB")


def detect(model, image):
    """Return [(class_name, (x_center, y_center, width, height)), ...] in pixels."""
    detections = model.predict(image, threshold=0.5)
    results = []
    for (x1, y1, x2, y2), class_id in zip(detections.xyxy, detections.class_id):
        w, h = x2 - x1, y2 - y1
        results.append((COCO_CLASSES[class_id], (x1 + w / 2, y1 + h / 2, w, h)))
    return results


if __name__ == "__main__":
    model = RFDETRNano()

    frame = get_frame()  # 1. capture (preprocessing = ensure RGB; predict() resizes/normalizes)
    for name, (xc, yc, w, h) in detect(model, frame):  # 2. inference
        print(f"{name:12s} cx={xc:7.1f} cy={yc:7.1f} w={w:6.1f} h={h:6.1f}")
