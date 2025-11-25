
"""
Continuous keyboard-controlled motor stepping for Raspberry Pi GPIO steppers.

Controls:
  W/S : X axis forward/back
  A/D : Y axis forward/back
  Q/E : End effector up/down
  ESC : Quit (also cleans up GPIO)

Design:
 - Non-blocking main loop that steps each motor independently at a target stepping rate.
 - Uses pynput for keyboard press/release events so holding a key will keep the motor stepping.
 - Safe GPIO cleanup on exit.

Stepping rate:
 - Default 500 Hz (0.002 s between step pulses). Pulse width is short (0.0005 s).
 - Change STEP_INTERVAL to adjust top speed.

Note: Confirm that your stepper drivers accept the chosen STEP_INTERVAL. Start slow if unsure.
"""
import time
from threading import Event
import RPi.GPIO as GPIO
from pynput import keyboard

# ---------------------------
# === USER-ADJUSTABLE VARS ===
# ---------------------------
# Step timing: total time between step pulses (seconds). 0.002 -> 500 Hz
STEP_INTERVAL = 0.002
# Pulse width for the step pulse (seconds). Keep small but long enough for driver detection.
PULSE_WIDTH = 0.0005

# ---------------------------
# === PIN DEFINITIONS (BOARD MODE) ===
# ---------------------------
GPIO.setmode(GPIO.BOARD)

# X axis (pulse, direction)
PU_x = 13
DR_x = 11

# Y axis 1 (pulse, direction)
PU_y1 = 10
DR_y1 = 8

# Y axis 2 (pulse, direction)
PU_y2 = 12
DR_y2 = 16

# End effector (pulse, direction)
PU_eff = 15
DR_eff = 7

# Optional: enable pins (set to None if unused)
EN_x = None
EN_y1 = None
EN_y2 = None
EN_eff = None

ALL_PINS = [p for p in [PU_x, DR_x, PU_y1, DR_y1, PU_y2, DR_y2, PU_eff, DR_eff,
                       EN_x, EN_y1, EN_y2, EN_eff] if p is not None]

# ---------------------------
# === GPIO SETUP ============
# ---------------------------
for p in ALL_PINS:
    GPIO.setup(p, GPIO.OUT)
    # default low
    GPIO.output(p, GPIO.LOW)

# ---------------------------
# === HELPER: STEP ONCE  ====
# ---------------------------
def pulse(pin_pulse):
    """Generate a single pulse on the given pulse pin. Very short blocking sleep for pulse width."""
    GPIO.output(pin_pulse, GPIO.HIGH)
    time.sleep(PULSE_WIDTH)
    GPIO.output(pin_pulse, GPIO.LOW)

def step_x(direction):
    """Step X axis once. direction: +1 for forward, -1 for backward"""
    GPIO.output(DR_x, GPIO.HIGH if direction > 0 else GPIO.LOW)
    pulse(PU_x)

def step_y1(direction):
    GPIO.output(DR_y1, GPIO.HIGH if direction > 0 else GPIO.LOW)
    pulse(PU_y1)

def step_y2(direction):
    GPIO.output(DR_y2, GPIO.HIGH if direction > 0 else GPIO.LOW)
    pulse(PU_y2)

def step_eff(direction):
    GPIO.output(DR_eff, GPIO.HIGH if direction > 0 else GPIO.LOW)
    pulse(PU_eff)

# ---------------------------
# === KEY STATE TRACKING ====
# ---------------------------
key_state = {
    'w': False,  # X forward
    's': False,  # X backward
    'a': False,  # Y backward
    'd': False,  # Y forward
    'q': False,  # effector up
    'e': False,  # effector down
}

running = Event()
running.set()

def on_press(key):
    try:
        k = key.char.lower()
        if k in key_state:
            key_state[k] = True
    except AttributeError:
        # special keys (e.g. ESC)
        if key == keyboard.Key.esc:
            running.clear()  # stop main loop

def on_release(key):
    try:
        k = key.char.lower()
        if k in key_state:
            key_state[k] = False
    except AttributeError:
        pass

# ---------------------------
# === MAIN LOOP ============
# ---------------------------
def main_loop(step_interval=STEP_INTERVAL):
    # Track last step times per motor so motors can have independent timing
    last_x = last_y = last_eff = time.monotonic()
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            print("Keyboard control active. Hold keys to move. ESC to quit.")
            print("W/S: X axis   A/D: Y axis   Q/E: End effector")
            while running.is_set():
                now = time.monotonic()

                # X axis control (W = forward, S = backward). Priority: W over S if both pressed.
                if key_state['w'] or key_state['s']:
                    if now - last_x >= step_interval:
                        if key_state['w'] and not key_state['s']:
                            step_x(+1)
                        elif key_state['s'] and not key_state['w']:
                            step_x(-1)
                        # if both pressed, do nothing (or decide priority)
                        last_x = now

                # Y axis control (D = forward, A = backward)
                if key_state['d'] or key_state['a']:
                    if now - last_y >= step_interval:
                        if key_state['d'] and not key_state['a']:
                            step_y1(+1)
                            step_y2(+1)
                        elif key_state['a'] and not key_state['d']:
                            step_y1(-1)
                            step_y2(-1)
                        last_y = now

                # End effector control (Q = up, E = down)
                if key_state['q'] or key_state['e']:
                    if now - last_eff >= step_interval:
                        if key_state['q'] and not key_state['e']:
                            step_eff(+1)
                        elif key_state['e'] and not key_state['q']:
                            step_eff(-1)
                        last_eff = now

                # small sleep to yield CPU; avoid busy-waiting
                time.sleep(0.0005)
    except Exception as e:
        print("Exception in main loop:", e)
    finally:
        cleanup()

def cleanup():
    print("Cleaning up GPIO and exiting...")
    try:
        # stop all motors
        for p in ALL_PINS:
            GPIO.output(p, GPIO.LOW)
    except Exception:
        pass
    GPIO.cleanup()

# ---------------------------
# === ENTRY POINT ===========
# ---------------------------
if __name__ == '__main__':
    try:
        main_loop()
    except KeyboardInterrupt:
        running.clear()
        cleanup()
