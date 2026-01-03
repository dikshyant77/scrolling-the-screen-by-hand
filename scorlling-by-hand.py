import time
import cv2
import mediapipe as mp
import pyautogui

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandMarkerResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def result_callback(result: HandLandMarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def process_scroll(result: HandLandMarkerResult, h, w):
    if not result or not result.hand_landmarks:
        return
    landmarks = result.hand_landmarks[0]
    index_tip = landmarks[8]
    y_pos = int(index_tip.y * h)

    if y_pos < h // 3:
        pyautogui.scroll(50)
    elif y_pos > 2 * h // 3:
        pyautogui.scroll(-50)

def main():
    options = HandLandMarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=result_callback
    )


    with HandLandMarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return


        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break


            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(time.time() * 1000)


            landmarker.detect_async(mp_image, ts_ms)


            h, w = frame_bgr.shape[:2]
            process_scroll(latest_result, h, w)


            cv2.imshow("Hand Gesture Scrolling", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




