import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO)


async def capture_and_send(
    url: str,
    prompt: str,
    negative_prompt: str,
    image_size: int = 256,
    rotate: float = 0,
    fullscreen: bool = False,
    jpeg_quality: int = 90,
    target_fps: int = 30
) -> None:
    """
    Captures frames from the webcam, processes them, and sends them to a WebSocket server.

    Args:
        url (str): WebSocket server URL.
        prompt (str): Positive prompt for the server.
        negative_prompt (str): Negative prompt for the server.
        image_size (int): Target image size (square).
        rotate (float): Rotation angle for received frames.
        fullscreen (bool): Display window in fullscreen mode.
        jpeg_quality (int): JPEG compression quality (1-100).
        target_fps (int): Target frames per second.

    Returns:
        None
    """
    async with websockets.connect(uri=url) as websocket:
        t_start = time.time()
        cap = cv2.VideoCapture(1)
        print(f"Camera init time: {time.time() - t_start:.4f}")

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Connected to server...")

        # Send prompt configuration as JSON
        await websocket.send(json.dumps({"prompt": prompt, "negative_prompt": negative_prompt}))

        frame_count = 0
        last_send_time = 0
        min_frame_interval = 1 / target_fps
        last_stats_time = time.time()
        frames_since_stats = 0
        last_displayed_frame = None

        while True:
            loop_start = time.time()

            # Capture frame
            t_start = time.time()
            ret, frame = cap.read()
            capture_time = time.time() - t_start

            if not ret:
                break

            # Check if we should process this frame based on target FPS
            current_time = time.time()
            if current_time - last_send_time < min_frame_interval:
                # If we have a previous frame, display it to maintain smooth output
                if last_displayed_frame is not None:
                    cv2.imshow("image", last_displayed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Crop frame
            t_start = time.time()
            h, w, _ = frame.shape
            half_min_side = min(h, w) // 2
            frame = frame[
                h // 2 - half_min_side : h // 2 + half_min_side,
                w // 2 - half_min_side : w // 2 + half_min_side
            ]
            crop_time = time.time() - t_start

            # Resize and convert
            t_start = time.time()
            frame = cv2.resize(frame, dsize=(image_size, image_size))
            frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
            resize_convert_time = time.time() - t_start

            # Encode frame
            t_start = time.time()
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            result, encimg = cv2.imencode(ext=".jpg", img=frame, params=encode_params)
            encode_time = time.time() - t_start

            if not result:
                print("Failed to encode image")
                continue

            # Network send
            t_start = time.time()
            jpeg_bytes = encimg.tobytes()
            await websocket.send(jpeg_bytes)
            last_send_time = current_time
            send_time = time.time() - t_start

            # Network receive with timeout
            t_start = time.time()
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1 / target_fps)
                receive_time = time.time() - t_start

                # Process and display
                t_start = time.time()
                if isinstance(response, bytes):
                    decimg = np.frombuffer(response, dtype=np.uint8)
                    frame_decoded = cv2.imdecode(decimg, cv2.IMREAD_COLOR)
                    if frame_decoded is None:
                        print("Failed to decode received image")
                        continue

                    frame_decoded = cv2.flip(frame_decoded, flipCode=1)
                    frame_decoded = cv2.resize(frame_decoded, dsize=(1400, 1400), interpolation=cv2.INTER_CUBIC)

                    if rotate != 0:
                        (h_dec, w_dec) = frame_decoded.shape[:2]
                        center = (w_dec / 2, h_dec / 2)
                        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
                        frame_decoded = cv2.warpAffine(frame_decoded, M, (w_dec, h_dec))

                    cv2.imshow("image", frame_decoded)
                    last_displayed_frame = frame_decoded
                else:
                    print("Received non-bytes message from server")
                process_display_time = time.time() - t_start

            except asyncio.TimeoutError:
                receive_time = time.time() - t_start
                print("Server response timeout - skipping frame")
                # Display last frame if available
                if last_displayed_frame is not None:
                    cv2.imshow(winname="image", mat=last_displayed_frame)
                continue

            total_loop_time = time.time() - loop_start

            # Update statistics
            frames_since_stats += 1
            current_time = time.time()
            stats_interval = current_time - last_stats_time

            # Print timing every 30 frames or every second, whichever comes first
            if frames_since_stats >= 30 or stats_interval >= 1.0:
                print("\nTiming breakdown (seconds):")
                print(f"Capture: {capture_time:.4f}")
                print(f"Crop: {crop_time:.4f}")
                print(f"Resize & Convert: {resize_convert_time:.4f}")
                print(f"Encode: {encode_time:.4f}")
                print(f"Send: {send_time:.4f}")
                print(f"Receive: {receive_time:.4f}")
                print(f"Process & Display: {process_display_time:.4f}")
                print(f"Total loop time: {total_loop_time:.4f}")
                print(f"Actual FPS: {frames_since_stats/stats_interval:.2f}")
                print("-" * 40)

                # Reset statistics
                frames_since_stats = 0
                last_stats_time = current_time

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0.001)  # Small sleep to prevent CPU overload

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="URL of server", default="ws://localhost:5678")
    parser.add_argument("--prompt", type=str, help="Prompt to send to server", required=True)
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt to send to server",
                        default="low quality")
    parser.add_argument("--image_size", type=int, help="Image size", default=256)
    parser.add_argument("--rotate", type=float, default=0, help="Rotate the image by specified degrees")
    parser.add_argument("--fullscreen", action="store_true", help="Display window in fullscreen mode")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG compression quality (1-100)")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS for frame capture")
    args = parser.parse_args()

    asyncio.run(
        capture_and_send(
            url=args.url,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image_size=args.image_size,
            rotate=args.rotate,
            fullscreen=args.fullscreen,
            jpeg_quality=args.jpeg_quality,
            target_fps=args.target_fps
        )
    )
