import cv2
import os
from pathlib import Path

# --- CONFIG ---
folder_path = Path(r"D:\programm-files\steam\steamapps\common\Brawlhalla\mapArt\Backgrounds")
preview_image = next(p for p in folder_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp'])
window_name = "Preview Blur"

# --- Preview Function ---
def apply_blur_and_preview():
    img = cv2.imread(str(preview_image))
    if img is None:
        print("❌ Could not load preview image.")
        return

    cv2.namedWindow(window_name)

    def on_trackbar(val):
        k = val * 2 + 1  # make kernel size odd and >= 1
        if k <= 1:
            blurred = img
        else:
            blurred = cv2.GaussianBlur(img, (k, k), 0)
        cv2.imshow(window_name, blurred)

    cv2.createTrackbar('Blur', window_name, 1, 100, on_trackbar)
    on_trackbar(1)

    print("🧪 Adjust the blur level using the slider. Press 'a' to apply to all images, 'q' to quit without applying.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            blur_amount = cv2.getTrackbarPos('Blur', window_name)
            kernel_size = blur_amount * 2 + 1
            cv2.destroyAllWindows()
            return kernel_size
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

# --- Batch Blur Function ---
def apply_blur_to_all_images(kernel_size):
    for file in folder_path.iterdir():
        if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            try:
                img = cv2.imread(str(file))
                if img is None:
                    print(f"⚠️ Skipping unreadable file: {file.name}")
                    continue
                blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                cv2.imwrite(str(file), blurred)
                print(f"✅ Blurred: {file.name}")
            except Exception as e:
                print(f"❌ Error processing {file.name}: {e}")

# --- Main ---
kernel_size = apply_blur_and_preview()
if kernel_size:
    print(f"\n🌀 Applying Gaussian blur with kernel size ({kernel_size}, {kernel_size}) to all images...")
    apply_blur_to_all_images(kernel_size)
    print("🎉 Done.")
else:
    print("❌ No changes applied.")
