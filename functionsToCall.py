#helper functions for gui
import argparse, time, sys, os, json
import sqlite3
from pathlib import Path
import yaml
import numpy as np
import cv2

def init():
    aruco_dict_name = "DICT_6X6_250"
    marker_length = 0.05  # in meters
    camera = 0
    calib = "calib.yaml"
    db_path = "presence.db"
    presence_timeout = 2.0  # seconds
    basket_size = 0.1  # in meters
    min_dt = 0.8  # seconds
    return aruco_dict_name, marker_length, camera, calib,db_path, presence_timeout, basket_size, min_dt

# ---------- SQLite presence syncronizing tools ----------
def db_init(db_path: str):
    """make/open presence.db and ensure tables exist"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    # Markers table
    cur.execute("""CREATE TABLE IF NOT EXISTS markers(
        id INTEGER PRIMARY KEY, name TEXT NOT NULL)""")
    
    # Presence table
    cur.execute("""CREATE TABLE IF NOT EXISTS presence(
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, present INTEGER NOT NULL,
        last_seen REAL NOT NULL, x REAL, y REAL, z REAL,
        FOREIGN KEY(id) REFERENCES markers(id))""")
    
    # History table
    cur.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER, name TEXT NOT NULL, t REAL NOT NULL, 
        x REAL, y REAL, z REAL,
        FOREIGN KEY(id) REFERENCES markers(id))""")
    
    # Basket / Item relation table
    cur.execute("""CREATE TABLE IF NOT EXISTS basket_items(
        basket_id INTEGER,
        item_id   INTEGER,
        last_seen REAL NOT NULL,
        PRIMARY KEY (basket_id, item_id),
        FOREIGN KEY(basket_id) REFERENCES markers(id),
        FOREIGN KEY(item_id)   REFERENCES markers(id)
    )""")

    # Basket location table
    cur.execute("""CREATE TABLE IF NOT EXISTS basket_locations(
        basket_id  INTEGER PRIMARY KEY,
        gx         REAL NOT NULL,
        gy         REAL NOT NULL,
        last_seen  REAL NOT NULL,
        FOREIGN KEY(basket_id) REFERENCES markers(id)
    )""")

    # optonal indexes
    cur.execute("""CREATE INDEX IF NOT EXISTS ix_basket_items_basket ON basket_items(basket_id)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS ix_basket_items_item   ON basket_items(item_id)""")

    conn.commit()

    cur.executemany("INSERT OR IGNORE INTO markers(id,name) VALUES(?,?)",
                [(0,'book0'),(1,'book1'),(2,'book2'),(3,'book3'),(4,'book4'),(5,'book5'),
                 (6,'book6'),(7,'book7'),(8,'book8'),(9,'book9'),(10,'book10'), 
                 (11,'book11'),(12,'book12'),(13,'book13'),(14,'book14'),(15,'book15'),
                 (16,'book16'),(17,'book17'),(18,'book18'),(19,'book19'),(20,'book20'),
                 (21,'book21'),(22,'book22'),(23,'book23'),(24,'book24')
                ])
    conn.commit()
    return conn

def db_insert_history(conn, aruco_id: int, tvec, name: str|None=None):
    """add collected info to history table"""
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    now_ts = time.time()
    cur = conn.cursor()
    if name is None:
        cur.execute("INSERT OR IGNORE INTO markers(id,name) VALUES(?,?)",
                    (aruco_id, f"ID {aruco_id}"))
    cur.execute("""
        INSERT INTO history(id,name,t,x,y,z)
        VALUES(?, COALESCE(?, (SELECT name FROM markers WHERE id=?)), ?, ?, ?, ?)
        """, (aruco_id, name, aruco_id, now_ts, x, y, z))
    cur.execute(
        """
        DELETE FROM history
        WHERE id = ?
          AND t NOT IN (
              SELECT t FROM history
              WHERE id = ?
              ORDER BY t DESC
              LIMIT 2
          )
        """,
        (aruco_id, aruco_id),
    )
    conn.commit()

def db_upsert_presence(conn, aruco_id: int, tvec, name: str|None=None):
    """update id to present, record location"""
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    now_ts = time.time()
    cur = conn.cursor()
    if name is None:
        cur.execute("INSERT OR IGNORE INTO markers(id,name) VALUES(?,?)",
                    (aruco_id, f"ID {aruco_id}"))
    cur.execute("""
    INSERT INTO presence(id,name,present,last_seen,x,y,z)
    VALUES(?, COALESCE(?, (SELECT name FROM markers WHERE id=?)), 1, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
      name=excluded.name,
      present=1,
      last_seen=excluded.last_seen,
      x=excluded.x, y=excluded.y, z=excluded.z
    """, (aruco_id, name, aruco_id, now_ts, x, y, z))
    conn.commit()

def upsert_basket_location(conn, basket_id: int, gantry_pos):
    """write basket location in gantry coordinate system / update to basket_locations"""
    gx, gy = gantry_pos   # motorControl.getPosition() 返回的 (x, y)
    gx = float(gx)
    gy = float(gy)
    now_ts = time.time()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO basket_locations(basket_id, gx, gy, last_seen)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(basket_id) DO UPDATE SET
            gx        = excluded.gx,
            gy        = excluded.gy,
            last_seen = excluded.last_seen
    """, (int(basket_id), gx, gy, now_ts))
    conn.commit()


def db_timeout_sweep(conn, timeout_s: float):
    """if not seen after timeout_s, id is not present"""
    cur = conn.cursor()
    cur.execute("UPDATE presence SET present=0 WHERE present=1 AND last_seen < ?",
                (time.time() - float(timeout_s),))
    conn.commit()


# ========= Basket / Item relation=========
BASKET_ID_MIN = 100  #

def is_basket_id(aruco_id:int)->bool:
    return int(aruco_id) >= BASKET_ID_MIN

_last_relation_update = {}

def assign_items_to_baskets(conn, items, baskets, side=0.1, min_dt=0.8):
    global _last_relation_update
    if not items:
        return
    cur = conn.cursor()
    now = time.time()
    last = most_recent_basket_detection(conn)
    for iid, (ix,iy,iz) in items:
        # Find the basket that contains the item
        best = None
        '''logic 1'''
        #The item belongs to a basket if it is within the basket boundaries
        # if baskets:
        #     for bid, (bx,by,bz) in baskets:
        #         # Calculate basket boundaries
        #         dl = bx - side/2  # left boundary
        #         dr = bx + side/2  # right boundary
        #         db = by - side  # bottom boundary 
        #         dt = by   # top boundary
                
        #         # Check if item is inside basket boundaries
        #         if ix >= dl and ix <= dr and iy >= db and iy <= dt:
        #             best = bid
        #             break  # Found containing basket, no need to check others
        '''logic 2'''
        #The items belongs to the most recently seen basket
        
        if best is None:
            
            if last is None:
                continue  # no basket in history yet; skip
            mid, name, t, x, y, z, t_iso = last
            #mid, name, t, x, y, z, t_iso = most_recent_basket_detection(conn)
            print(f"Most recent basket detection: ID={mid}, name={name}, time={t_iso}")
            best = mid

        key = (int(best), int(iid))
        last_ts = _last_relation_update.get(key, 0.0)
        if now - last_ts < min_dt:
            continue

        if best is not None:
            cur.execute("""
                INSERT INTO basket_items(basket_id, item_id, last_seen)
                VALUES(?,?,?)
                ON CONFLICT(basket_id, item_id)
                DO UPDATE SET last_seen = excluded.last_seen
            """, (int(best), int(iid), now))
            _last_relation_update[key] = now
    conn.commit()
import sqlite3

def get_xyz_by_id(conn, aruco_id:int):
    cur = conn.cursor()
    cur.execute("""
        SELECT x, y, z FROM presence
        WHERE id = ? AND present = 1
        ORDER BY last_seen DESC LIMIT 1
    """, (aruco_id,))
    row = cur.fetchone()
    if not row:
        return None
    x, y, z = map(float, row)
    return x, y, z

from datetime import datetime, timezone

def most_recent_basket_detection(conn):
    """
    Returns the single most recent history row that belongs to a basket.
    Output: (id, name, t, x, y, z, t_iso)
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT h.id, COALESCE(m.name, CAST(h.id AS TEXT)) AS name, h.t, h.x, h.y, h.z
        FROM history AS h
        LEFT JOIN markers AS m ON m.id = h.id
        WHERE h.id >= ?
        ORDER BY h.t DESC
        LIMIT 1
    """, (BASKET_ID_MIN,))
    row = cur.fetchone()
    if not row:
        return None
    mid, name, t, x, y, z = row
    t_iso = datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
    return int(mid), name, float(t), float(x), float(y), float(z), t_iso

def most_recent_book_detection(conn):
    """
    Returns the single most recent history row that belongs to a book.
    Output: (id, name, t, x, y, z, t_iso)
    """
    cur = conn.cursor()
    cur.execute(f"""
        SELECT h.id, COALESCE(m.name, CAST(h.id AS TEXT)) AS name, h.t, h.x, h.y, h.z
        FROM history AS h
        LEFT JOIN markers AS m ON m.id = h.id
        WHERE h.id < ?
        ORDER BY h.t DESC
        LIMIT 1
    """, (BASKET_ID_MIN,))
    row = cur.fetchone()
    if not row:
        return None
    mid, name, t, x, y, z = row
    t_iso = datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
    return int(mid), name, float(t), float(x), float(y), float(z), t_iso

def get_basket_xyz_for_item(conn, basketname, baskets):
    cur = conn.cursor()
    cur.execute("SELECT id FROM markers WHERE name = ?", (basketname,))
    row = cur.fetchone()
    if row is None:
        return None 
    basket_id = row[0]
    for bid, (bx, by, bz) in baskets:
        if bid == basket_id:
            return (bx, by, bz)
    return None

def get_basket_id_by_book_name(conn, book_name: str):
    cur = conn.cursor()
    cur.execute("""
        SELECT bi.basket_id
        FROM basket_items AS bi
        JOIN markers AS m ON m.id = bi.item_id
        WHERE m.name = ?
        ORDER BY bi.last_seen DESC
        LIMIT 1
    """, (book_name,))
    row = cur.fetchone()
    if row is None:
        return None
    return int(row[0])

def get_basket_location_by_id(conn, basket_id: int):
    """
    Return the gantry position (gx, gy) for the given basket_id,
    or None if no location has been recorded yet.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT gx, gy FROM basket_locations
        WHERE basket_id = ?
    """, (int(basket_id),))
    row = cur.fetchone()
    if row is None:
        return None
    gx, gy = row
    return float(gx), float(gy)


def get_detector(aruco_dict_name: str):
    '''get aruco marker detector function, dictionary and parameters'''
    name = aruco_dict_name.upper()
    table = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    if name not in table:
        raise ValueError(f"Unknown Dictionary {aruco_dict_name}， can choose from：{', '.join(table)}")

    dictionary = cv2.aruco.getPredefinedDictionary(table[name])

    # 4.7+ new interface
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        def detect(gray):
            return detector.detectMarkers(gray)
        return detect, dictionary, params
    else:
        # old interface
        params = cv2.aruco.DetectorParameters_create()
        def detect(gray):
            return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return detect #, dictionary, params

def load_cam_params(path):
    '''load camera calibration parameters from file'''
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calib file not found: {p}")
    if p.suffix.lower()==".npz":
        data = np.load(p)
        return data["K"], data["dist"]
    # yaml/json
    with open(p, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    K = np.array(obj["K"], dtype=np.float64)
    dist = np.array(obj["dist"], dtype=np.float64).reshape(-1)
    return K, dist

def draw_axis_if_possible(frame, corners, ids, K, dist, marker_length_m):
    if K is None or marker_length_m is None or ids is None:
        return
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, float(marker_length_m), K, dist)
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        #draw axis
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, float(marker_length_m) * 0.5)
        X, Y, Z = tvec.reshape(3)
        dist_m = float(np.linalg.norm([X, Y, Z]))  
        Z_m = float(Z)                             
        # text position
        c = corners[i][0].mean(axis=0).astype(int)
        x, y = int(c[0]), int(c[1])
        cv2.putText(frame, f"Z={Z_m:.3f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"|t|={dist_m:.3f} m", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

def begin_camera_detection(aruco_dict_name, marker_length, camera, yaml, db_path, presence_timeout, basket_size, min_dt):
    detect, dictionary, params = get_detector(aruco_dict_name)
    K, dist = load_cam_params(yaml)
    conn = db_init(db_path)
    print(f"[INFO] presence DB: {db_path}")
    
    
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERR] cannot turn on camera {camera}")
        sys.exit(1)
    

    prev = time.time()
    while True:
        items_list = []
        baskets_list = []
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect(gray)

        if ids is not None:
            # draw frame, axis and id text
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            draw_axis_if_possible(frame, corners, ids, K, dist, marker_length)
            for i, cid in enumerate(ids.flatten()):
                c = corners[i][0].mean(axis=0).astype(int)
                cv2.putText(frame, f"id={int(cid)}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            # --- synchronize with presence.db ---
            if conn is not None and (K is not None) and (marker_length is not None):
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, float(marker_length), K, dist
                    )
                    # 1) write presence / history
                    for i, cid in enumerate(ids.flatten()):
                        t = tvecs[i, 0]   # (X, Y, Z) in meters
                        db_upsert_presence(conn, int(cid), t)
                        db_insert_history(conn, int(cid), t)

                        # distinguish baskets and items
                        triple = (float(t[0]), float(t[1]), float(t[2]))
                        if is_basket_id(int(cid)):
                            baskets_list.append((int(cid), triple))
                        else:
                            items_list.append((int(cid), triple))
                    # 2) assign items to baskets
                    assign_items_to_baskets(conn, items_list, baskets_list, basket_size, min_dt)

                except Exception as e:
                    cv2.putText(frame, f"DB sync err: {str(e)[:40]}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
        # ---- display FPS / preview/ exit with ESC----
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        if conn is not None and (now - prev) > 0.5:
            db_timeout_sweep(conn, presence_timeout)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow("ArUco", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC / q
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":  
    begin_camera_detection(*init())


