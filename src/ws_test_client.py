import asyncio
import json
import cv2
import websockets
import numpy as np

WS_URL = "ws://localhost:8000/ws"
VIDEO_PATH = "test_video.mp4"
FRAME_SKIP = 2  # send every Nth frame to reduce load
JPEG_QUALITY = 80


async def send_video():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        print("Connected to server")

        frame_idx = 0
        cam_id = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video finished")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Encode frame as JPEG
            ok, jpg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                continue

            # Send metadata
            meta = {"cam_id": cam_id}
            await ws.send(json.dumps(meta))

            # Send frame bytes
            await ws.send(jpg.tobytes())

            # Receive annotated frame
            annotated_bytes = await ws.recv()

            img = cv2.imdecode(
                np.frombuffer(annotated_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            if img is not None:
                cv2.imshow("Dart Detection", img)
                cv2.waitKey(1)

            # Receive scores
            response = await ws.recv()
            print("scores:", response)

            await asyncio.sleep(0.03)  # ~30 FPS pacing

    cap.release()


if __name__ == "__main__":
    asyncio.run(send_video())

