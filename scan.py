import threading
import functionsToCall
import motorControl
import sqlite3

aruco_dict_name, marker_length, camera, calib, db_path, presence_timeout, basket_size, min_dt = functionsToCall.init()
conn = sqlite3.connect(db_path, check_same_thread=False)

import time

def find_centered_basket_x(conn, x_tol=0.01):
    """
    Check if there is any currently present *basket* whose x (camera frame) is close to 0.
    Returns the basket marker id if found, otherwise None.
    """
    cur = conn.cursor()
    # Only consider baskets: id >= BASKET_ID_MIN
    cur.execute(
        "SELECT id, x FROM presence WHERE present = 1 AND id >= ?",
        (functionsToCall.BASKET_ID_MIN,)
    )
    for mid, x in cur.fetchall():
        if x is None:
            continue
        if abs(float(x)) <= x_tol:
            return int(mid)
    return None


def find_centered_basket_y(conn, y_tol=0.01):
    """
    Check if there is any currently present *basket* whose y (camera frame) is close to 0.
    Returns the basket marker id if found, otherwise None.
    """
    cur = conn.cursor()
    # Only consider baskets: id >= BASKET_ID_MIN
    cur.execute(
        "SELECT id, y FROM presence WHERE present = 1 AND id >= ?",
        (functionsToCall.BASKET_ID_MIN,)
    )
    for mid, y in cur.fetchall():
        if y is None:
            continue
        if abs(float(y)) <= y_tol:
            return int(mid)
    return None


def move_until_marker_centered_x(conn, step_x, velocity, x_tol=0.01,
                                 max_steps=1000, sleep_dt=0.05):
    """
    Move the gantry in +x direction in small steps until some marker's x is ~0.

    Returns:
        marker_id if a centered marker was found, otherwise None.
    """
    for _ in range(max_steps):
        # Check DB for a centered marker in x
        mid = find_centered_basket_x(conn, x_tol=x_tol)
        if mid is not None:
            return mid

        # If not centered yet, move a small step in x
        motorControl.moveGantry(step_x, 0, velocity)
        # Give camera thread some time to grab a new frame and update DB
        time.sleep(sleep_dt)

    return None  # could not find centered marker within max_steps


def move_until_marker_centered_y(conn, step_y, velocity, y_tol=0.01,
                                 max_steps=1000, sleep_dt=0.05):
    """
    Move the gantry in +y direction in small steps until some marker's y is ~0.

    Returns:
        marker_id if a centered marker was found, otherwise None.
    """
    for _ in range(max_steps):
        mid = find_centered_basket_y(conn, y_tol=y_tol)
        if mid is not None:
            return mid

        motorControl.moveGantry(0, step_y, velocity)
        time.sleep(sleep_dt)

    return None


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

    Corrected behavior:
      - Basket search happens:
          * at the beginning (first row),
          * while scanning horizontally within a row,
          * when moving to the next row.
      - Whenever a basket marker is found and used for centering (x or y),
        its current gantry position is written into basket_locations.
      - The 0.25 m up/down motion is NOT used to search for baskets.
        It is only executed AFTER a basket has been centered in x, as a
        mechanical scan of the region below that basket.

    All distances are in meters.
    """
    # 0) Home the gantry and move to the initial starting position.
    motorControl.zeroGantry()
    start_x, start_y = start_pos
    velocity = speed
    basket_location = []

    # Start camera detection thread so the presence DB stays updated.
    stop_event, cam_thread = start_camera_thread()

    try:
        # Move to user-defined starting position (rough).
        motorControl.moveGantry(start_x, start_y, velocity)

        # 1) Find and center the first row in Y using basket markers only.
        #    This is the "initial" basket search in y-direction.
        centered_row_id = move_until_marker_centered_y(
            conn,
            step_y=-0.01,        # 1 cm per step downward, adjust as needed
            velocity=0.15,
            y_tol=0.01,          # ±1 cm tolerance in camera coordinates
            max_steps=1000,
            sleep_dt=0.05
        )
        if centered_row_id is None:
            print("[perform_scan] Could not center any basket marker in y for the first row.")
            return basket_location

        # We have a basket-centered row in y -> record its gantry position.
        pos = motorControl.getPosition()
        basket_location.append([centered_row_id, pos])
        functionsToCall.upsert_basket_location(conn, centered_row_id, pos)

        # 2) Row-by-row scanning loop.
        x, y = motorControl.getPosition()
        while 0 < y < height:
            # 2.1) For each row, start horizontally from the left boundary (start_x).
            cur_x, cur_y = motorControl.getPosition()
            dx_to_start = start_x - cur_x
            motorControl.moveGantry(dx_to_start, 0.0, velocity)
            x, y = motorControl.getPosition()

            # 2.2) Within this row, repeatedly:
            #      - move until a BASKET marker is centered in x,
            #      - record that basket's gantry position,
            #      - then perform a 0.25 m down/up motion (NOT for searching),
            #      - continue to look for the next x-centered basket.
            while 0 < x < width:
                # Horizontal basket search in x-direction.
                centered_id_x = move_until_marker_centered_x(
                    conn,
                    step_x=step_size,   # step_size in meters (e.g. 0.05)
                    velocity=velocity,
                    x_tol=0.01,         # ±1 cm in camera x
                    max_steps=1000,
                    sleep_dt=0.05
                )
                # Refresh current gantry position after the movement.
                x, y = motorControl.getPosition()

                # If no basket marker could be centered in x, this row is done.
                if centered_id_x is None or not (0 < x < width):
                    break

                # We have a basket marker centered in x -> record its gantry position.
                pos = motorControl.getPosition()
                basket_location.append([centered_id_x, pos])
                functionsToCall.upsert_basket_location(conn, centered_id_x, pos)

                # --- 0.25 m up/down motion (NOT used for basket search) ---
                # This is purely mechanical motion once the basket is already centered.
                motorControl.moveGantry(0.0, -0.25, velocity=0.1)
                time.sleep(0.1)
                motorControl.moveGantry(0.0,  0.25, velocity=0.1)
                time.sleep(0.1)
                # After the vertical motion we simply continue scanning in x.
                x, y = motorControl.getPosition()

            # ---- End of one row in x. Move to the next row. ----

            # 2.3) Move in y to the next row: basket search in y-direction.
            centered_row_id = move_until_marker_centered_y(
                conn,
                step_y=-0.01,      # step down 1 cm each iteration
                velocity=0.15,
                y_tol=0.01,
                max_steps=1000,
                sleep_dt=0.05
            )
            if centered_row_id is None:
                print("[perform_scan] No further basket-centered row in y; stopping scan.")
                break

            # Record the new row's basket-centered gantry position.
            pos = motorControl.getPosition()
            basket_location.append([centered_row_id, pos])
            functionsToCall.upsert_basket_location(conn, centered_row_id, pos)

            # Update position for the next row loop.
            x, y = motorControl.getPosition()

        return basket_location

    finally:
        # Always stop the camera thread when scan finishes.
        stop_event.set()
        cam_thread.join()



def retract_basket(conn, book_name, speed=50):
    """
    This version:
      1) looks up the basket_id for the given book name in the DB, and
      2) looks up the basket's gantry position (gx, gy) in the basket_locations table.
      3) moves the gantry to that position.

    Args:
        conn            : sqlite3 connection to the presence.db database.
        basket_location : kept for backward compatibility, not used anymore.
        book_name       : name of the book, e.g. "book3".
        speed           : gantry moving speed (passed to motorControl.moveGantry).
    Returns:
        True if a matching basket location was found and the gantry was moved,
        False otherwise.
    """
    # Reset gantry to a known reference position (e.g. home/origin).
    motorControl.zeroGantry()

    # 1) Find which basket this book belongs to.
    basket_id = functionsToCall.get_basket_id_by_book_name(conn, book_name)
    if basket_id is None:
        print(f"[retract_basket] No basket recorded for book '{book_name}'.")
        return False

    # 2) Look up the gantry position of this basket in the DB.
    basket_pos = functionsToCall.get_basket_location_by_id(conn, basket_id)
    if basket_pos is None:
        print(
            f"[retract_basket] Basket {basket_id} (for book '{book_name}') "
            f"has no recorded gantry position in basket_locations."
        )
        return False

    target_x, target_y = basket_pos

    # 3) Compute relative motion from current gantry position to the basket position.
    cur_x, cur_y = motorControl.getPosition()
    dx = target_x - cur_x
    dy = target_y - cur_y

    # 4) Move the gantry to that basket position.
    motorControl.moveGantry(dx, dy, speed)

    print(
        f"[retract_basket] Moved gantry to basket {basket_id} "
        f"for book '{book_name}' at ({target_x}, {target_y})."
    )
    return True

if __name__ == "__main__": 
    start_pos = (0.0, 0.0)  # starting gantry position (x, y) in m
    end_pos = (2.0, 1.5)    # ending gantry position (x, y) in m
    step_size = 0.05  # 5 cm
    speed = 0.2       # m/s
    width = 2.0       # total scan width in m
    height = 1.5      # total scan height in m 
    perform_scan(start_pos, end_pos, step_size, speed, width, height, conn)


