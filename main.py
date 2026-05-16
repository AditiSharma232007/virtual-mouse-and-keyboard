from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class HandState:
    landmarks: list[Point]
    handedness: str

    def point(self, index: int) -> Point:
        return self.landmarks[index]

    def distance(self, first: int, second: int) -> float:
        a = self.point(first)
        b = self.point(second)
        return math.hypot(a.x - b.x, a.y - b.y)

    def is_finger_up(self, tip: int, pip: int) -> bool:
        return self.point(tip).y < self.point(pip).y


class HandTracker:
    def __init__(self, max_hands: int = 1) -> None:
        model_path = "models/hand_landmarker.task"
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.72,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._started_at = time.monotonic()

    def detect(self, frame: np.ndarray) -> HandState | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._started_at) * 1000)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        height, width = frame.shape[:2]
        points = [
            Point(int(lm.x * width), int(lm.y * height))
            for lm in hand_landmarks
        ]
        handedness = "hand"
        if result.handedness:
            handedness = result.handedness[0][0].category_name
        return HandState(points, handedness)

    def draw(self, frame: np.ndarray, hand: HandState | None) -> None:
        if hand is None:
            return
        for point in hand.landmarks:
            cv2.circle(frame, (point.x, point.y), 3, (30, 220, 160), -1)

        bones = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        )
        for first, second in bones:
            a = hand.point(first)
            b = hand.point(second)
            cv2.line(frame, (a.x, a.y), (b.x, b.y), (45, 160, 255), 2)


class GestureGate:
    def __init__(self, cooldown: float) -> None:
        self.cooldown = cooldown
        self._last_at = 0.0

    def ready(self) -> bool:
        return time.monotonic() - self._last_at >= self.cooldown

    def fire(self) -> None:
        self._last_at = time.monotonic()


class VirtualMouse:
    def __init__(self, sensitivity: float = 1.25) -> None:
        self.enabled = True
        self.control_windows = False
        self.sensitivity = sensitivity
        self.left_click = GestureGate(0.35)
        self.right_click = GestureGate(0.45)
        self.scroll_gate = GestureGate(0.08)
        current_x, current_y = pyautogui.position()
        self._smooth_x = float(current_x)
        self._smooth_y = float(current_y)
        self._last_scroll_y: int | None = None

    def update(self, hand: HandState, frame_size: tuple[int, int]) -> str | None:
        if not self.enabled:
            return None

        frame_width, frame_height = frame_size
        screen_width, screen_height = pyautogui.size()
        index = hand.point(8)

        margin_x = int(frame_width * 0.16)
        margin_y = int(frame_height * 0.16)
        usable_width = max(1, frame_width - margin_x * 2)
        usable_height = max(1, frame_height - margin_y * 2)

        normalized_x = np.clip((index.x - margin_x) / usable_width, 0, 1)
        normalized_y = np.clip((index.y - margin_y) / usable_height, 0, 1)
        target_x = screen_width - (normalized_x * screen_width)
        target_y = normalized_y * screen_height

        alpha = min(0.95, 0.18 * self.sensitivity)
        self._smooth_x = self._smooth_x + (target_x - self._smooth_x) * alpha
        self._smooth_y = self._smooth_y + (target_y - self._smooth_y) * alpha
        if self.control_windows:
            pyautogui.moveTo(self._smooth_x, self._smooth_y)

        pinch_index = hand.distance(4, 8)
        pinch_middle = hand.distance(4, 12)
        index_up = hand.is_finger_up(8, 6)
        middle_up = hand.is_finger_up(12, 10)

        if pinch_index < 34 and self.left_click.ready():
            if self.control_windows:
                pyautogui.click()
            self.left_click.fire()
            return "left click"

        if pinch_middle < 38 and self.right_click.ready():
            if self.control_windows:
                pyautogui.click(button="right")
            self.right_click.fire()
            return "right click"

        if index_up and middle_up and self.scroll_gate.ready():
            if self._last_scroll_y is not None:
                dy = self._last_scroll_y - index.y
                if abs(dy) > 8:
                    if self.control_windows:
                        pyautogui.scroll(int(dy * 0.8))
                    self.scroll_gate.fire()
                    return "scroll"
            self._last_scroll_y = index.y
        else:
            self._last_scroll_y = None

        return None


@dataclass
class KeyButton:
    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, point: Point) -> bool:
        return self.x <= point.x <= self.x + self.w and self.y <= point.y <= self.y + self.h


class VirtualKeyboard:
    ROWS = [
        list("QWERTYUIOP"),
        list("ASDFGHJKL"),
        ["Z", "X", "C", "V", "B", "N", "M"],
        ["SPACE", "BACK", "CLEAR", "EXIT"],
    ]

    def __init__(self) -> None:
        self.enabled = True
        self.type_to_windows = False
        self.press_gate = GestureGate(0.42)
        self._buttons: list[KeyButton] = []
        self._last_hover: str | None = None
        self.text = ""

    def layout(self, frame_width: int, frame_height: int) -> list[KeyButton]:
        key_gap = 7
        key_h = max(38, frame_height // 13)
        key_w = max(42, frame_width // 16)
        start_y = frame_height - ((key_h + key_gap) * 4) - 18
        buttons: list[KeyButton] = []

        for row_index, row in enumerate(self.ROWS):
            row_width = 0
            widths: list[int] = []
            for key in row:
                width = key_w if len(key) == 1 else key_w * 2 + key_gap
                if key == "SPACE":
                    width = key_w * 4
                if key == "EXIT":
                    width = key_w * 2 + key_gap
                widths.append(width)
                row_width += width
            row_width += key_gap * (len(row) - 1)

            x = (frame_width - row_width) // 2
            y = start_y + row_index * (key_h + key_gap)
            for key, width in zip(row, widths):
                buttons.append(KeyButton(key, x, y, width, key_h))
                x += width + key_gap

        self._buttons = buttons
        return buttons

    def update(self, frame: np.ndarray, hand: HandState | None) -> str | None:
        if not self.enabled:
            return None

        height, width = frame.shape[:2]
        buttons = self.layout(width, height)
        fingertip = hand.point(8) if hand else None
        hovered: KeyButton | None = None

        if fingertip:
            hovered = next((button for button in buttons if button.contains(fingertip)), None)
            self._last_hover = hovered.label if hovered else None
        else:
            self._last_hover = None

        self.draw(frame, hovered)

        if hand is None or hovered is None:
            return None

        is_press = hand.distance(4, 8) < 34
        if is_press and self.press_gate.ready():
            self.press_gate.fire()
            if hovered.label == "EXIT":
                return "EXIT"
            self._send_key(hovered.label)
            return hovered.label
        return None

    def draw(self, frame: np.ndarray, hovered: KeyButton | None) -> None:
        overlay = frame.copy()
        self._draw_text_box(frame, overlay)
        for button in self._buttons:
            is_hovered = hovered is not None and button.label == hovered.label
            fill = (80, 230, 140) if is_hovered else (32, 34, 42)
            border = (230, 255, 240) if is_hovered else (95, 105, 125)
            cv2.rectangle(
                overlay,
                (button.x, button.y),
                (button.x + button.w, button.y + button.h),
                fill,
                -1,
            )
            cv2.rectangle(
                overlay,
                (button.x, button.y),
                (button.x + button.w, button.y + button.h),
                border,
                1,
            )
            font_scale = 0.55 if len(button.label) == 1 else 0.45
            text_size = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = button.x + (button.w - text_size[0]) // 2
            ty = button.y + (button.h + text_size[1]) // 2
            color = (12, 22, 18) if is_hovered else (235, 238, 245)
            cv2.putText(frame, button.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    def _draw_text_box(self, frame: np.ndarray, overlay: np.ndarray) -> None:
        height, width = frame.shape[:2]
        box_x = max(20, width // 8)
        box_y = 72
        box_w = width - box_x * 2
        box_h = 64
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 20, 26), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (120, 132, 150), 1)

        visible_text = self.text[-42:] if self.text else "Typed text appears here"
        color = (245, 248, 255) if self.text else (150, 156, 170)
        cv2.putText(
            frame,
            visible_text,
            (box_x + 18, box_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def _send_key(self, label: str) -> None:
        if label == "SPACE":
            self.text += " "
            self._send_to_windows("space")
        elif label == "BACK":
            self.text = self.text[:-1]
            self._send_to_windows("backspace")
        elif label == "ENTER":
            self.text += "\n"
            self._send_to_windows("enter")
        elif label == "CLEAR":
            self.text = ""
        else:
            self.text += label.lower()
            self._send_to_windows(label.lower(), write=True)

    def _send_to_windows(self, key: str, write: bool = False) -> None:
        if not self.type_to_windows:
            return
        try:
            if write:
                pyautogui.write(key)
            else:
                pyautogui.press(key)
        except pyautogui.FailSafeException:
            pass


def draw_status(frame: np.ndarray, lines: Iterable[str]) -> None:
    x, y = 16, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam virtual mouse and keyboard")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--mouse-sensitivity", type=float, default=1.25, help="Mouse smoothing sensitivity")
    parser.add_argument("--no-mouse", action="store_true", help="Start with mouse control disabled")
    parser.add_argument("--no-keyboard", action="store_true", help="Start with keyboard hidden")
    parser.add_argument("--control-windows", action="store_true", help="Allow gestures to control real Windows mouse/keyboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Try --camera 1.")

    tracker = HandTracker()
    mouse = VirtualMouse(args.mouse_sensitivity)
    keyboard = VirtualKeyboard()
    mouse.enabled = not args.no_mouse
    keyboard.enabled = not args.no_keyboard
    mouse.control_windows = args.control_windows
    keyboard.type_to_windows = args.control_windows
    last_action = "ready"

    window_name = "Virtual Mouse & Keyboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            hand = tracker.detect(frame)
            tracker.draw(frame, hand)

            if hand:
                key_action = keyboard.update(frame, hand)
                keyboard_hovering = keyboard.enabled and keyboard._last_hover is not None
                mouse_action = None if keyboard_hovering else mouse.update(hand, (frame.shape[1], frame.shape[0]))
                if mouse_action:
                    last_action = mouse_action
                if key_action:
                    if key_action == "EXIT":
                        break
                    last_action = f"key {key_action}"
            else:
                keyboard.update(frame, None)

            draw_status(
                frame,
                [
                    f"Mouse: {'on' if mouse.enabled else 'off'}   Keyboard: {'on' if keyboard.enabled else 'off'}   Windows: {'on' if mouse.control_windows else 'off'}",
                    f"Action: {last_action}",
                    "q/Esc quit   m mouse   k keyboard   w windows control",
                ],
            )

            cv2.imshow(window_name, frame)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord("m"):
                mouse.enabled = not mouse.enabled
            if key == ord("k"):
                keyboard.enabled = not keyboard.enabled
            if key == ord("w"):
                mouse.control_windows = not mouse.control_windows
                keyboard.type_to_windows = mouse.control_windows
                last_action = f"windows control {'on' if mouse.control_windows else 'off'}"
            if key == ord("c"):
                mouse._smooth_x, mouse._smooth_y = pyautogui.position()
                last_action = "recentered"
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
