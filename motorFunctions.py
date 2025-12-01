from motorControl import *

global vertDisplacment


def pickUpCubby(slotted=True):
    if slotted: # cubby is T-shape slotted
        moveEndEffector(.10,.01) # distance in meters, velocity in m/s, move forward
        moveGantry(0,-.05,.01) # x distance in meters, y distance in meters, velocity in m/s, move down
        moveEndEffector(-.05,.01) # distance in meters, velocity in m/s, move backward halfway
        moveEndEffector(0.01,.01) # distance in meters, velocity in m/s, move forward slowly to align
        moveGantry(0,.05,.01) # x distance in meters, y distance in meters, velocity in m/s, move up
        moveEndEffector(-.05,.01) # distance in meters, velocity in m/s, move backward to original position
    else: # cubby is non-slotted
        moveEndEffector(.10,.01) # distance in meters, velocity in m/s, move forward
        moveGantry(0,.05,.01) # x distance in meters, y distance in meters, velocity in m/s, move up
        moveEndEffector(-.10,.01) # distance in meters, velocity in m/s, move backward to original position
    
    
    return "Cubby picked up"

def releaseCubby():
    # moveArmToPosition("down")
    # openGripper()
    # moveArmToPosition("up")
    return "Cubby released"