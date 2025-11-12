import time
import RPi.GPIO as GPIO

#Set code to read board pin numbers (not GPIO numbers)
GPIO.setmode(GPIO.BOARD)

#SET PIN NUMBERS
#X axis pin numbers (PU is pulse, DR is direction)
PU_x = 13
DR_x = 11
#Y axis pin numbers (PU is pulse, DR is direction)
PU_y = 8
DR_y = 10
#X axis limit switches
limit_upper_x = 37
limit_lower_x = 40
#Y axis limit switches (add these)
limit_upper_y = 9999
limit_lower_y = 9999

#SETUP PINS
GPIO.setup(PU_x, GPIO.OUT)
GPIO.setup(DR_x, GPIO.OUT)
GPIO.setup(PU_y, GPIO.OUT)
GPIO.setup(DR_y, GPIO.OUT)
GPIO.setup(limit_upper_x, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_lower_x, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
#This is currently coded to control the X axis.
def moveMotor(distance, velocity):
    #400 pulses per revolution, 8mm per revolution
    
    # Initialize PU pin
    GPIO.output(PU_x, GPIO.LOW)
    
    #Calculations for motor movement
    rotations = abs(distance)/.008;
    num_pulses = int(rotations*400);
    travel_time = abs(distance/velocity);
    step_delay = travel_time/num_pulses;
    
    #Initialize DR pin
    if distance > 0:
        GPIO.output(DR_x, GPIO.LOW)
    else:
        GPIO.output(DR_x, GPIO.HIGH)
    
    print("Moving Gantry...")
    print("Distance:",distance,"m");
    print("Velocity:",velocity,"m/s");
    print("Rotations:",rotations);
    print("Number of pulses:",num_pulses);
    print("Pulse delay:",step_delay)

    try:
        #Each loop is 2 pulses
        for i in range(int(num_pulses/2 + 1)):
            
            #limit switch failsafe
            if GPIO.input(limit_upper_x) == 0 or GPIO.input(limit_lower_x) == 0:
                print("Limit switch collision")
                break
            
            GPIO.output(PU_x, GPIO.HIGH)
            time.sleep(step_delay)
            GPIO.output(PU_x, GPIO.LOW)
            time.sleep(step_delay)
        
    except:
        print("Stopping motor...")

    finally:
        GPIO.cleanup()
        print("GPIO cleanup done.")
        

def moveGantry(x_distance, y_distance, velocity):
    #400 pulses per revolution, 8mm per revolution

    #X AXIS ACTIONS
    
    # Initialize PU pin
    GPIO.output(PU_x, GPIO.LOW)

    #Calculations for motor movement
    rotations_x = abs(distance_x)/.008;
    num_pulses_x = int(rotations_x*400);
    travel_time_x = abs(distance_x/velocity);
    step_delay_x = travel_time_x/num_pulses_x;
    
    #Initialize DR pin
    if distance_x > 0:
        GPIO.output(DR_x, GPIO.LOW)
    else:
        GPIO.output(DR_x, GPIO.HIGH)
    
    #Y AXIS ACTIONS
    
    # Initialize PU pin
    GPIO.output(PU_y, GPIO.LOW)

    rotations_y = abs(distance_y)/.008;
    num_pulses_y = int(rotations_y*400);
    travel_time_y = abs(distance_y/velocity);
    step_delay_y = travel_time_y/num_pulses_y;
    
    #Initialize DR pin
    if distance_y > 0:
        GPIO.output(DR_y, GPIO.LOW)
    else:
        GPIO.output(DR_y, GPIO.HIGH)
    
    #print("Moving Gantry...")
    #print("Distance:",distance,"m");
    #print("Velocity:",velocity,"m/s");
    #print("Rotations:",rotations);
    #print("Number of pulses:",num_pulses);
    #print("Pulse delay:",step_delay)

    try:
        for i in range(int(max(num_pulses_x, num_pulses_y)/2 + 1)):
            
            #x axis limit switch failsafe
            if GPIO.input(limit_upper_x) == 0 or GPIO.input(limit_lower_x) == 0:
                print("Limit switch collision")
                break
                
            #Pulse each motor high until it has reached num_pulses_i
            if i <= num_pulses_x:
                GPIO.output(PU_x, GPIO.HIGH)
            if i <= num_pulses_y:
                GPIO.output(PU_y, GPIO.HIGH)
            time.sleep(step_delay)
            
            #Pulse each motor low until it has reached num_pulses_i          
            if i <= num_pulses_x:
                GPIO.output(PU_x, GPIO.LOW)
            if i <= num_pulses_y:
                GPIO.output(PU_y, GPIO.LOW)
            time.sleep(step_delay)
        
    except:
        print("Stopping motor...")

    finally:
        GPIO.cleanup()
        print("GPIO cleanup done.")

def zeroGantry(x_position, y_position):

    #X AXIS ACTIONS
    
    # Initialize PU and DR pins
    GPIO.output(PU_x, GPIO.LOW)
    GPIO.output(DR_x, GPIO.HIGH)
    
    #Y AXIS ACTIONS
    
    # Initialize PU and DR pins
    GPIO.output(PU_y, GPIO.LOW)
    GPIO.output(DR_x, GPIO.HIGH)
    
    #50,000 pulses per meter.
    #1 m/s = 50,000 / s
    step_delay = .001
    
    limit_x_pressed = 0
    limit_y_pressed = 0
    while limit_x_pressed != 1 and limit_y_pressed != 1:
            
        #Pulse each motor high until it has reached num_pulses_i
        if limit_x_pressed != 1:
            GPIO.output(PU_x, GPIO.HIGH)
        if i <= num_pulses_y:
            GPIO.output(PU_y, GPIO.HIGH)
        time.sleep(step_delay)
        
        #Pulse each motor low until it has reached num_pulses_i
        if limit_x_pressed != 1:
            GPIO.output(PU_x, GPIO.LOW)
        if limit_y_pressed != 1:
            GPIO.output(PU_y, GPIO.LOW)
        time.sleep(step_delay)

        #x axis limit switch failsafe
        if GPIO.input(limit_lower_x) == 0
            limit_x_pressed = 1
        
        #x axis limit switch failsafe
        if GPIO.input(limit_lower_y) == 0
            limit_y_pressed = 1
            
        

moveMotor(.08, 0.002) #One rotation at .002 m/s (4 second test)
