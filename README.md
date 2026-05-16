# Virtual Mouse & Keyboard

A webcam-based virtual mouse and keyboard project inspired by the demo you shared.

It uses hand landmarks from MediaPipe:

- Move the mouse with your index fingertip.
- Left click by pinching index finger + thumb.
- Right click by pinching middle finger + thumb.
- Scroll by raising index + middle finger and moving up/down.
- Type by hovering over the on-screen keyboard and pinching.
- By default, typed text stays inside the app so random Windows apps do not open.
- Use the virtual `EXIT` key to close the app from the webcam window.

The project includes `models/hand_landmarker.task`, the MediaPipe hand model used by the current MediaPipe Tasks API.

## Setup

Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If MediaPipe does not install on your current Python version, install Python 3.11 or 3.12 and create the virtual environment with that version.

## Run

```powershell
python main.py
```

Useful options:

```powershell
python main.py --camera 0
python main.py --mouse-sensitivity 1.4
python main.py --no-mouse
python main.py --no-keyboard
python main.py --control-windows
```

## Controls

- `q` or `Esc`: quit
- `m`: toggle mouse mode
- `k`: toggle virtual keyboard
- `w`: toggle real Windows control on/off
- `c`: recenter mouse control area
- Virtual `CLEAR`: clear typed text
- Virtual `EXIT`: quit

## Tips

- Use a bright room and keep your hand fully visible.
- Keep your palm facing the camera.
- Start slowly. Pinches intentionally have a short cooldown so one gesture does not spam clicks or letters.
- On Windows, your first run may ask for camera permission.
- Keep `Windows: off` while practicing. Turn it on with `w` only when you actually want gestures to control your real mouse/keyboard.
