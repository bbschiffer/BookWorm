import threading
import functionsToCall
import motorControl
import sqlite3

aruco_dict_name, marker_length, camera, calib, db_path, presence_timeout, basket_size, min_dt = functionsToCall.init()
conn = sqlite3.connect(db_path, check_same_thread=False)

def start_camera_thread():
    # create a stop event
    stop_event = threading.Event()
    # initialize and start camera thread
    cam_thread = threading.Thread(
        target=functionsToCall.begin_camera_detection,
        args=(aruco_dict_name, marker_length, camera, calib, db_path, presence_timeout, basket_size, min_dt, stop_event),
        daemon=True
    )
    cam_thread.start()
    return stop_event, cam_thread

def perform_scan(start_pos, end_pos, step_size, speed, width, height, conn):
    """
    Perform a scan based on the provided parameters.

    Args:
        scan_parameters (dict): A dictionary containing scan parameters such as
                                start position, end position, step size, and speed.
    """
    motorControl.zeroGantry()
    distance_x = start_pos[0] 
    distance_y = start_pos[1]
    velocity = speed
    basket_location = []

    stop_event, cam_thread = start_camera_thread()

    try:
        motorControl.moveGantry(distance_x, distance_y, velocity)
        
        position = motorControl.getPosition()
        next_basket_detected = False
        while 0 < position[0] < width and 0 < position[1] < height:
            while next_basket_detected == False:
                motorControl.moveGantry(step_size, 0, velocity)
                if functionsToCall.basket_detected(conn):
                    info = functionsToCall.most_recent_basket_detection(conn)
                    basket_location.append([info[0], motorControl.getPosition()])
                    next_basket_detected = True
            motorControl.moveGantry(to one side of basket, to where books are below, velocity)
            motorControl.moveGantry(across a basket, 0, velocity)
            next_basket_detected = False
            position = motorControl.getPosition()
        motorControl.moveGantry(to_Next_row_begining, to Next row below, velocity)
        return basket_location
    finally:
        stop_event.set()
        cam_thread.join()
