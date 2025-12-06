from motorControl import *

global vertDisplacment

ee_extension = .2
ee_extension_loaded = .25
ee_velocity = .05

def pickUpCubby(slotted=False):
    
    #Dont ever call this
    if slotted: # cubby is T-shape slotted
        moveEndEffector(.10,.01) # distance in meters, velocity in m/s, move forward
        moveGantry(0,-.05,.01) # x distance in meters, y distance in meters, velocity in m/s, move down
        moveEndEffector(-.05,.01) # distance in meters, velocity in m/s, move backward halfway
        moveEndEffector(0.01,.01) # distance in meters, velocity in m/s, move forward slowly to align
        moveGantry(0,.05,.01) # x distance in meters, y distance in meters, velocity in m/s, move up
        moveEndEffector(-.05,.01) # distance in meters, velocity in m/s, move backward to original position
    else: # cubby is non-slotted
        moveEndEffector(ee_extension,ee_velocity) # distance in meters, velocity in m/s, move forward
        moveGantry(0,.02,.04) # x distance in meters, y distance in meters, velocity in m/s, move up
        
        #Slow retraction at end to prevent cubby wobbling
        moveEndEffector(-(ee_extension_loaded-.02),ee_velocity) # distance in meters, velocity in m/s, move backward to original position
        moveEndEffector(-.02,ee_velocity/2) # distance in meters, velocity in m/s, move backward to original position
    
    return "Cubby picked up"

def releaseCubby():
    # moveArmToPosition("down")
    # openGripper()
    # moveArmToPosition("up")
    moveEndEffector(ee_extension_loaded,ee_velocity) # distance in meters, velocity in m/s, move forward
    moveGantry(0,-.02,.04) # x distance in meters, y distance in meters, velocity in m/s, move up
    moveEndEffector(-ee_extension,ee_velocity) # distance in meters, velocity in m/s, move backward to original position
    return "Cubby released"

if __name__ == "__main__": 
    #moveGantry(0.1,0.1,0.06)
    moveGantry(.337,.573,0.06)
    pickUpCubby()
    time.sleep(2)
    releaseCubby()
    zeroGantry()
    
