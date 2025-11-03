import time
import RPi.GPIO as GPIO

PU_pin = 13
DR_pin = 11

GPIO.setmode(GPIO.BOARD)

# Setup pins
GPIO.setup(PU_pin, GPIO.OUT)
GPIO.setup(DR_pin, GPIO.OUT)

# Initialize PU pin
GPIO.output(PU_pin, GPIO.LOW)

def moveMotor(distance, velocity):
    #400 pulses per revolution, 8mm per revolution
    
    rotations = abs(distance)/.008;
    num_pulses = int(rotations*400);
    travel_time = abs(distance/velocity);
    step_delay = travel_time/num_pulses;
    
    #Initialize DR pin
    if distance > 0:
        GPIO.output(DR_pin, GPIO.LOW)
    else:
        GPIO.output(DR_pin, GPIO.HIGH)
    
    #step_delay = .01
    
    #num_pulses = 50000
    #print(exp_step_delay)
    
    print("Moving Gantry...")
    print("Distance:",distance,"m");
    print("Velocity:",velocity,"m/s");
    print("Rotations:",rotations);
    print("Number of pulses:",num_pulses);
    print("Pulse delay:",step_delay)

    try:
        for i in range(int(num_pulses/2 + 1)):
            GPIO.output(PU_pin, GPIO.HIGH)
            time.sleep(step_delay)
            #print("pulse",i)
            GPIO.output(PU_pin, GPIO.LOW)
            time.sleep(step_delay)
        
    except:
        print("Stopping motor...")

    finally:
        GPIO.cleanup()
        print("GPIO cleanup done.")

#moveMotor(.008, 0.002) #One rotation at .002 m/s (4 second test)
