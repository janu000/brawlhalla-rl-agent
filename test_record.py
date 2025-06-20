from ScreenRecorder import ScreenRecorder
import time
import numpy as np
import cv2

def test_screen_recorder():
    print("Starting ScreenRecorder test...")
    recorder = ScreenRecorder(fps=20, region=None, buffer_size=60, render=True) # Test with 10 FPS

    try:
        recorder.start()
        print("Recording started. Waiting for 5 seconds...")
        time.sleep(1) # Record for 5 seconds

        frames = recorder.get_latest_frames(50)
        if frames:
            print(f"Successfully captured {len(frames)} frames.")
            # Save the first captured frame
            resized_frame = cv2.resize(frames[0], (256, 144))
            cv2.imwrite("captured_frame.png", resized_frame)
            print("Saved 'captured_frame.png'")
            # The live rendering is handled by the ScreenRecorder class.
            # You can uncomment the following lines if you want to also display the buffered frames after recording stops
            # for i, frame in enumerate(frames):
            #     cv2.imshow(f"Buffered Frame {i}", frame)
            #     cv2.waitKey(100)
            # cv2.destroyAllWindows()
        else:
            print("No frames captured.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        recorder.stop()
        print("Recording stopped. Test finished.")

if __name__ == "__main__":
    test_screen_recorder() 