import time
import RPi.GPIO as GPIO

PUL_PIN = 26
DIR_PIN = 12

GPIO.setmode(GPIO.BCM)

# Setup pins
GPIO.setup(PUL_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# Initialize pins
GPIO.output(PUL_PIN, GPIO.LOW)
GPIO.output(DIR_PIN, GPIO.HIGH)

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
