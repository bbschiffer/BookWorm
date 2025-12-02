import time
import RPi.GPIO as GPIO

#Set code to read board pin numbers (not GPIO numbers)
GPIO.setmode(GPIO.BOARD)

#SET PIN NUMBERS
#X axis pin numbers (PU is pulse, DR is direction)
PU_x = 18
DR_x = 16
#Y axis 1 pin numbers (PU is pulse, DR is direction)
PU_y1 = 10
DR_y1 = 8
#Y axis 2 pin numbers (PU is pulse, DR is direction)
PU_y2 = 5
DR_y2 = 3
#End effector pin numbers (PU is pulse, DR is direction)
PU_ee = 21
DR_ee = 19
#X axis limit switches
limit_upper_x = 37
limit_lower_x = 40
#Y axis 1 limit switches (add these)
limit_upper_y1 = 38
limit_lower_y1 = 35
#Y axis 2 limit switches (add these)
limit_upper_y2 = 36
limit_lower_y2 = 33

#SETUP PINS
GPIO.setup(PU_x, GPIO.OUT)
GPIO.setup(DR_x, GPIO.OUT)
GPIO.setup(PU_y1, GPIO.OUT)
GPIO.setup(DR_y1, GPIO.OUT)
GPIO.setup(PU_y2, GPIO.OUT)
GPIO.setup(DR_y2, GPIO.OUT)
GPIO.setup(PU_ee, GPIO.OUT)
GPIO.setup(DR_ee, GPIO.OUT)
GPIO.setup(limit_upper_x, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_lower_x, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_upper_y1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_lower_y1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_upper_y2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(limit_lower_y2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#This is currently coded to control the X axis.
def moveMotor(distance, velocity):

    global position
    
    #400 pulses per revolution, 8mm per revolution
    GPIO.setmode(GPIO.BOARD)
    
    # Initialize PU pin
    GPIO.output(PU_x, GPIO.LOW)
    
    #Calculations for motor movement
    rotations = abs(distance)/.008;
    num_pulses = int(rotations*400);
    travel_time = abs(distance/velocity);
    if num_pulses > 0:
        step_delay = travel_time/num_pulses;
    else:
        step_delay = .01
    
    #Initialize DR pin
    if distance > 0:
        GPIO.output(DR_x, GPIO.LOW)
    else:
        GPIO.output(DR_x, GPIO.HIGH)
    
    print("Moving Gantry...")
    print("Distance:",distance,"m")
    print("Velocity:",velocity,"m/s")
    print("Rotations:",rotations);
    print("Number of pulses:",num_pulses)
    print("Pulse delay:",step_delay)
    print(" ")


    try:
        #Each loop is 2 pulses
        for i in range(int(num_pulses/2)):
            
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
    
    position = [position[0] + distance, position[1]]
        

def moveGantry(distance_x, distance_y, velocity):

    global position

    #400 pulses per revolution, 8mm per revolution
    GPIO.setmode(GPIO.BOARD)
    
    print("Moving Gantry...")
    
    #X AXIS ACTIONS
    
    # Initialize PU pin
    GPIO.output(PU_x, GPIO.LOW)

    #Calculations for motor movement
    rotations_x = abs(distance_x)/.008
    num_pulses_x = int(rotations_x*400)
    travel_time_x = abs(distance_x/velocity)
    #If x motor was not requested to move 0, calculate step delay.
    if num_pulses_x > 0:
        step_delay = travel_time_x/num_pulses_x
    else:
        step_delay = 0
    #step_delay = 2
    
    print("Moving X Axis...")
    print("Distance:",distance_x,"m")
    print("Velocity:",velocity,"m/s")
    print("Rotations:",rotations_x)
    print("Number of pulses:",num_pulses_x)
    print("Pulse delay:",step_delay)
    
    #Initialize DR pin
    if distance_x > 0:
        GPIO.output(DR_x, GPIO.LOW)
    else:
        GPIO.output(DR_x, GPIO.HIGH)
    
    #Y AXIS ACTIONS
    
    # Initialize PU pin
    GPIO.output(PU_y1, GPIO.LOW)
    GPIO.output(PU_y2, GPIO.LOW)

    rotations_y = abs(distance_y)/.008
    num_pulses_y = int(rotations_y*400)
    travel_time_y = abs(distance_y/velocity)
    #If step delay wasnt calculated with x motor and y motor was not called with 0, calculate it here.
    if step_delay == 0:
        step_delay = travel_time_y/num_pulses_y
    
    #Initialize DR pin
    if distance_y < 0:
        GPIO.output(DR_y1, GPIO.LOW)
        GPIO.output(DR_y2, GPIO.LOW)
    else:
        GPIO.output(DR_y1, GPIO.HIGH)
        GPIO.output(DR_y2, GPIO.HIGH)
    
    print("Moving Y Axis...")
    print("Distance:",distance_y,"m")
    print("Velocity:",velocity,"m/s")
    print("Rotations:",rotations_y)
    print("Number of pulses:",num_pulses_y)
    print("Pulse delay:",step_delay)
    print(" ")

    k = 1
    for i in range(int(max(num_pulses_x, num_pulses_y)/2 + 1)):
        
        #x axis limit switch failsafe
        #if GPIO.input(limit_upper_x) == 0 or GPIO.input(limit_lower_x) == 0:
         #   print("Limit switch collision")
         #   break
        
        #ADD CODE FOR Y AXIS LIMIT SWITCHES
            
        #Pulse each motor high until it has reached num_pulses_i
        if i < (num_pulses_x/2):
            GPIO.output(PU_x, GPIO.HIGH)
            
        if i < (num_pulses_y/2):
            GPIO.output(PU_y1, GPIO.HIGH)
            GPIO.output(PU_y2, GPIO.HIGH)
        time.sleep(step_delay)
        
        #Pulse each motor low until it has reached num_pulses_i
        if i < (num_pulses_x/2):
            GPIO.output(PU_x, GPIO.LOW)
            
        if i < (num_pulses_y/2):
            GPIO.output(PU_y1, GPIO.LOW)
            GPIO.output(PU_y2, GPIO.LOW)
        time.sleep(step_delay)
        
    #except:
    #    print("Stopping motor...")
    
    position = [position[0] + distance_x, position[1] + distance_y]

def zeroGantry():
    
    global position
    
    #X AXIS ACTIONS
    
    
    # Initialize PU and DR pins
    GPIO.output(PU_x, GPIO.LOW)
    GPIO.output(DR_x, GPIO.HIGH)
    
    #Y AXIS ACTIONS
    
    # Initialize PU and DR pins
    GPIO.output(PU_y1, GPIO.LOW)
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
            GPIO.output(PU_y1, GPIO.HIGH)
        time.sleep(step_delay)
        
        #Pulse each motor low until it has reached num_pulses_i
        if limit_x_pressed != 1:
            GPIO.output(PU_x, GPIO.LOW)
        if limit_y_pressed != 1:
            GPIO.output(PU_y1, GPIO.LOW)
        time.sleep(step_delay)

        #x axis limit switch failsafe
        if GPIO.input(limit_lower_x) == 0:
            limit_x_pressed = 1
        
        #x axis limit switch failsafe
        if GPIO.input(limit_lower_y1) == 0:
            limit_y_pressed = 1
            
    position = [0,0]
    

def moveEndEffector(distance, velocity):
    #400 pulses per revolution, 8mm per revolution
    GPIO.setmode(GPIO.BOARD)
    
    # Initialize PU pin
    GPIO.output(PU_ee, GPIO.LOW)
    
    #Calculations for motor movement
    rotations = abs(distance)/.006283;
    num_pulses = int(rotations*400);
    travel_time = abs(distance/velocity);
    if num_pulses > 0:
        step_delay = travel_time/num_pulses;
    else:
        step_delay = .01
    
    #Initialize DR pin
    if distance > 0:
        GPIO.output(DR_ee, GPIO.LOW)
    else:
        GPIO.output(DR_ee, GPIO.HIGH)
    
    print("Moving Gantry...")
    print("Distance:",distance,"m")
    print("Velocity:",velocity,"m/s")
    print("Rotations:",rotations);
    print("Number of pulses:",num_pulses)
    print("Pulse delay:",step_delay)
    print(" ")


    try:
        #Each loop is 2 pulses
        for i in range(int(num_pulses/2)):
            
            #limit switch failsafe
            if GPIO.input(limit_upper_x) == 0 or GPIO.input(limit_lower_x) == 0:
                print("Limit switch collision")
                break
            
            GPIO.output(PU_ee, GPIO.HIGH)
            time.sleep(step_delay)
            
            
            GPIO.output(PU_ee, GPIO.LOW)
            time.sleep(step_delay)
        
    except:
        print("Stopping motor...")
        
def getPosition():
    return position


def depositCubby():
    moveEndEffector(0.04,0.002) # move forward
    moveGantry(0,0.03,0.02) #move down
    moveEndEffector(0.01,0.002) # move until trigger

position = [0,0]
#moveMotor(.008, 0.002) #One rotation at .002 m/s (4 second test)
#moveGantry(0.1, 0, 0.002) #One rotation at .002 m/s (4 second test)
#moveGantry(0, -0.05, 0.05) #One rotation at .002 m/s (4 second test)

#time.sleep(2)
#moveEndEffector(-.1,.002)
moveGantry(0,0.01, 0.03) #One rotation at .002 m/s (4 second test)
#depositCubby()
#print(position)
#moveEndEffector(.005, .002)
#depositCubby()
