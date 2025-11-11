import time
import RPi.GPIO as GPIO

PUL_PIN = 26
DIR_PIN = 12


# Setup pins
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# Initialize pins
GPIO.output(PUL_PIN, GPIO.LOW)
GPIO.output(DIR_PIN, GPIO.HIGH)

# --- GPIO setup ---
GPIO.setmode(GPIO.BOARD)  # or GPIO.BOARD if you prefer physical pin numbering

# Define your 6 pins here (BCM numbering)
pins = [5, 6, 13, 19, 17, 21]

# Create boolean flags for each pin
pin_flags = {pin: False for pin in pins}

# Callback function for interrupts
def pin_callback(channel):
    pin_flags[channel] = True
    print(f"Falling edge detected on GPIO {channel}")

# Set up each pin as input with pull-up resistor and attach interrupt
for pin in pins:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(pin, GPIO.FALLING, callback=pin_callback, bouncetime=100)

try:
    print("Monitoring pins for falling edges. Press Ctrl+C to exit.")
    while True:
        # (Optional) show flag states for debugging
        print(pin_flags)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting cleanly...")

finally:
    GPIO.cleanup()



def moveMotor(distance, velocity):
    #400 pulses per revolution, 8mm per revolution
    
    num_pulses = (abs(distance)/.008)*400;
    travel_time = abs(distance/velocity);
    STEP_DELAY = travel_time/num_pulses;
    
    print("Moving Gantry...")
    print("Distance:",distance,"m");
    print("Velocity:",velocity,"m/s");
    print("Number of pulses:",num_pulses);
    print("Pulse delay:",STEP_DELAY)

    try:
        for i in num_pulses:
        
            GPIO.output(PUL_PIN, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            
            GPIO.output(PUL_PIN, GPIO.LOW)
            time.sleep(STEP_DELAY)
    except:
        print("Stopping motor...")

    finally:
        GPIO.cleanup()
        print("GPIO cleanup done.")
