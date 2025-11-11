# -*- coding: utf-8 -*-
import argparse, time, sys, os, json
import sqlite3
from pathlib import Path
import numpy as np
import cv2

# ==== 位姿工具 ====
def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = tvec.reshape(3)
    return T

def T_inv(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

# ==== 机械手驱动占位（把这里替换为你的SDK调用即可）====
class RobotDriver:
    def __init__(self, workspace_box=np.array([[-0.5,0.5],[-0.5,0.5],[0.0,0.8]])):
        self.workspace_box = workspace_box  # x,y,z范围

    def in_workspace(self, p):
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = self.workspace_box
        return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax) and (zmin <= p[2] <= zmax)

    def move_pose(self, x, y, z):
        # TODO: 这里改为你的机械手移动指令。当前仅打印。
        print("[SAFE] target out of workspace:", p)
        return False

import yaml
from pathlib import Path

def load_cam_params(path):
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

def load_T_base_cam(path):
    """从 yaml/json 读取 4x4 手眼矩阵（键名 T_base_cam，列表形式 4x4）"""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    T = np.array(obj["T_base_cam"], dtype=np.float64)
    assert T.shape==(4,4)
    return T

# ---------- SQLite presence 同步工具 ----------
def db_init(db_path: str):
    """创建/打开 presence.db 并建表"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS markers(
        id INTEGER PRIMARY KEY, name TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS presence(
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, present INTEGER NOT NULL,
        last_seen REAL NOT NULL, x REAL, y REAL, z REAL,
        FOREIGN KEY(id) REFERENCES markers(id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER, name TEXT NOT NULL, t REAL NOT NULL, 
        x REAL, y REAL, z REAL,
        FOREIGN KEY(id) REFERENCES markers(id))""")
    
    # 关系表：哪个筐子里有哪些物品（去重）
    cur.execute("""CREATE TABLE IF NOT EXISTS basket_items(
        basket_id INTEGER,
        item_id   INTEGER,
        last_seen REAL NOT NULL,
        PRIMARY KEY (basket_id, item_id),
        FOREIGN KEY(basket_id) REFERENCES markers(id),
        FOREIGN KEY(item_id)   REFERENCES markers(id)
    )""")
    # 可选索引（查询更快）
    cur.execute("""CREATE INDEX IF NOT EXISTS ix_basket_items_basket ON basket_items(basket_id)""")
    cur.execute("""CREATE INDEX IF NOT EXISTS ix_basket_items_item   ON basket_items(item_id)""")

    conn.commit()

    # 写入映射
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
    """把检测结果追加到历史轨迹表"""
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
    conn.commit()

def db_upsert_presence(conn, aruco_id: int, tvec, name: str|None=None):
    """把某个ID更新为在场,并记录最近位置"""
    x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
    now_ts = time.time()
    cur = conn.cursor()
    if name is None:
        # 如果 markers 里没有名字，就默认记为 "ID {id}"
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

def db_timeout_sweep(conn, timeout_s: float):
    """把超过 timeout_s 未出现的标记置为不在场"""
    cur = conn.cursor()
    cur.execute("UPDATE presence SET present=0 WHERE present=1 AND last_seen < ?",
                (time.time() - float(timeout_s),))
    conn.commit()


# ========= Basket / Item 关系维护 =========
BASKET_ID_MIN = 100  # 例：ID >= 100 视为筐子，其它视为物品；按你的实际规则调整

def is_basket_id(aruco_id:int)->bool:
    return int(aruco_id) >= BASKET_ID_MIN

def assign_items_to_baskets(conn, items, baskets, side=0.1):
    """
    items   = [(item_id, (x,y,z))]
    baskets = [(basket_id, (x,y,z))]
    """
    if not items or not baskets:
        return
    cur = conn.cursor()
    now = time.time()
    for iid, (ix,iy,iz) in items:
        # Find the basket that contains the item
        best = None
        #The item belongs to a basket if it is within the basket boundaries
        for bid, (bx,by,bz) in baskets:
            # Calculate basket boundaries
            dl = bx - side/2  # left boundary
            dr = bx + side/2  # right boundary
            db = by - side  # bottom boundary 
            dt = by   # top boundary
            
            # Check if item is inside basket boundaries
            if ix >= dl and ix <= dr and iy >= db and iy <= dt:
                best = bid
                break  # Found containing basket, no need to check others

        #The items belongs to the most recently seen basket
        mid, name, t, x, y, z = most_recent_basket_detection(conn)
        best = mid
        if best is not None:
            cur.execute("""
                INSERT INTO basket_items(basket_id, item_id, last_seen)
                VALUES(?,?,?)
                ON CONFLICT(basket_id, item_id)
                DO UPDATE SET last_seen = excluded.last_seen
            """, (int(best), int(iid), now))
    conn.commit()
import sqlite3

def get_basket_xyz_for_item(conn, basketname, baskets):
    """根据篮子名称获取篮子的XYZ坐标"""
    cur = conn.cursor()
    cur.execute("SELECT id FROM markers WHERE name = ?", (basketname,))
    row = cur.fetchone()
    if row is None:
        return None  # 篮子名称不存在
    basket_id = row[0]
    for bid, (bx, by, bz) in baskets:
        if bid == basket_id:
            return (bx, by, bz)
    return None  # 未找到对应的篮子坐标

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

def draw_axis_if_possible(frame, corners, ids, K, dist, marker_length_m):
    if K is None or marker_length_m is None or ids is None:
        return

    # 估计位姿
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, float(marker_length_m), K, dist)

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # 画坐标轴（轴长可按需放大/缩小）
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, float(marker_length_m) * 0.5)

        # ----- 文本显示距离 -----
        X, Y, Z = tvec.reshape(3)
        dist_m = float(np.linalg.norm([X, Y, Z]))  # 欧氏距离（相机光心到marker中心）
        Z_m = float(Z)                             # 轴向距离（沿相机Z）

        # 文本位置：用该标记角点的平均值
        c = corners[i][0].mean(axis=0).astype(int)
        x, y = int(c[0]), int(c[1])

        cv2.putText(frame, f"Z={Z_m:.3f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"|t|={dist_m:.3f} m", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

def setup(cam_id, aruco_const, cam_yaml, handeye_yaml):
    # 字典与检测器
    ar_dict = cv2.aruco.getPredefinedDictionary(aruco_const)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ar_dict, params)
        detect = lambda gray: detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        detect = lambda gray: cv2.aruco.detectMarkers(gray, ar_dict, parameters=params)

    # 相机、手眼
    K, dist = load_cam_params(cam_yaml)
    T_base_cam = load_T_base_cam(handeye_yaml)

    # 摄像头
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    robot = RobotDriver()
    return detect, K, dist, T_base_cam, cap, robot


def run_servo(detect, K, dist, T_base_cam, cap, robot, marker_len_m, 
              id_filter=None, z_offset=0.04, smoothing=0.4):
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        if ids is not None and len(ids) > 0:
            # 选择目标 ID
            pick = 0
            if id_filter is not None:
                idx = np.where(ids.flatten()==id_filter)[0]
                pick = int(idx[0]) if len(idx)>0 else None

            if pick is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len_m, K, dist)
                rvec, tvec = rvecs[pick], tvecs[pick]
                T_cam_marker = rvec_tvec_to_T(rvec, tvec)

                # 末端期望：Z 轴对齐标记 Z，沿 Z 留 z_offset
                T_ee_marker_des = np.eye(4)
                T_ee_marker_des[:3,:3] = np.eye(3)  # 这里等价于“末端Z对齐标记Z”
                T_ee_marker_des[:3,3]  = np.array([0,0, z_offset])

                # 坐标链：Base->Cam->Marker->EE
                T_base_marker    = T_base_cam @ T_cam_marker
                T_base_ee_target = T_base_marker @ T_inv(T_ee_marker_des)

                # ---- 位置 EMA 平滑 ----
                p = T_base_ee_target[:3,3]
                if p_smooth is None: p_smooth = p.copy()
                p_smooth = float(smoothing)*p + (1.0-float(smoothing))*p_smooth
                T_base_ee_target[:3,3] = p_smooth

                # 下发
                robot.move_pose(T_base_ee_target)

                # 可视化
                cv2.aruco.drawDetectedMarkers(frame, [corners[pick]], ids[pick:pick+1])
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_len_m*0.5)

        cv2.imshow("servo", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()


def scanbookshelf(detect, cap, robot, conn, origin_x, origin_y, origin_z):
    ''' Scan the bookshelf by moving the robot and detecting ArUco markers. 
    The items seen will be assigned to the most recently detected basket.'''
    x, y, z = origin_x, origin_y, origin_z
    robot.move_pose(x, y, z)

    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)
        if ids[0] >= BASKET_ID_MIN: 
            basketid, basketname, bt, bx, by, bz, bt_iso = most_recent_basket_detection(conn)
            current_basket_id = ids[0]
            while current_basket_id == basketid:
                bookid, bookname, it, ix, iy, iz, it_iso = most_recent_book_detection(conn)
                # Perform scanning logic here
                item = bookid, (ix,iy,iz)
                basket = basketid, (bx,by,bz)
                assign_items_to_baskets(conn, item, basket, side=0.1)


def main():
    ap = argparse.ArgumentParser(description="ArUco 识别（图片/摄像头）")

    ap.add_argument('--assign-radius', type=float, default=0.25,
                    help='物品归属筐子的距离阈值（与 tvec 单位一致,默认0.25m)')
    ap.add_argument('--basket-id-min', type=int, default=BASKET_ID_MIN,
                    help='>=此值视为筐子ID,其他为物品ID')
    ap.add_argument("--image", type=str, help="待检测图片路径（不填则用摄像头）")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号,默认0")
    ap.add_argument("--dict", type=str, default="DICT_4X4_50", help="ArUco 字典")
    ap.add_argument("--calib", type=str, help="相机标定文件（含 K、dist)")
    ap.add_argument("--marker-length", type=float, help="标记实际边长(米)，提供则启用位姿估计")
    ap.add_argument("--save", type=str, help="将结果保存到图片/视频文件（后缀.avi/.mp4 则保存视频）")
    ap.add_argument("--servo", action="store_true", help="启用伺服模式：根据位姿估计控制机械手")
    ap.add_argument("--scan", action="store_true", help="Make a scan of the bookshelf by moving" \
                    " the robot and detecting ArUco markers. books will be assigned to the most " \
                    "recently detected basket.")
    ap.add_argument("--handeye", type=str, help="手眼标定文件(yaml/json,含 4x4 T_base_cam)")
    ap.add_argument("--id", type=int, dest="follow_id", help="仅跟随指定 id")
    ap.add_argument("--smoothing", type=float, default=0.4, help="位置平滑系数(0-1),默认0.4")
    ap.add_argument("--db", type=str, help="SQLite 数据库路径（例如 presence.db)。提供则启用在场同步")
    ap.add_argument("--presence-timeout", type=float, default=2.0, help="超时下线秒数,默认2s")
    args = ap.parse_args()
    globals()['BASKET_ID_MIN'] = int(getattr(args, 'basket_id_min', BASKET_ID_MIN))
    
    # 启动设置
    origin_x, origin_y, origin_z = 0.0, 0.0, 0.0 # 书架扫描起始位置
    const = getattr(cv2.aruco, args.dict.upper())
    detect, K, dist, T_base_cam, cap, robot = setup(cam_id=args.camera,aruco_const=const,cam_yaml=args.calib,handeye_yaml=args.handeye)
    
    # 伺服模式
    if args.servo:
        if not (args.calib and args.handeye and args.marker_length):
            raise SystemExit("--servo 需要同时提供 --calib、--handeye、--marker-length")
        run_servo(detect, K, dist, T_base_cam, cap, robot,marker_len_m=args.marker_length,id_filter=args.follow_id,z_offset=0.04, smoothing=args.smoothing)
        return
    
    # 扫描书架模式
    if args.scan:
        if not (args.calib and args.handeye and args.marker_length):
            raise SystemExit("--servo 需要同时提供 --calib、--handeye、--marker-length")
        scanbookshelf(detect, cap, robot, conn, origin_x, origin_y, origin_z)
        return

    # 允许无标定文件，仅做检测；只有在提供 --calib 时才加载参数
    K, dist = (None, None)
    if args.calib:
        K, dist = load_cam_params(args.calib)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"[ERR] 读图失败：{args.image}")
            sys.exit(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            draw_axis_if_possible(img, corners, ids, K, dist, args.marker_length)
            for i, cid in enumerate(ids.flatten()):
                c = corners[i][0].mean(axis=0).astype(int)
                cv2.putText(img, f"id={cid}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            print(f"检测到 {len(ids)} 个标记：", ids.flatten().tolist())
        else:
            print("未检测到标记")
        if args.save:
            cv2.imwrite(args.save, img)
            print("[OK] 已保存到", args.save)
        cv2.imshow("ArUco", img)
        cv2.waitKey(0)
        return

    #数据库同步
    conn = None
    if args.db:
        conn = db_init(args.db)
        print(f"[INFO] presence DB: {args.db}")

    # 可选视频保存
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if args.save.lower().endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (w, h))
        print("[INFO] 保存到", args.save)

    items_list = []
    baskets_list = []

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detect(gray)

        # 每帧重置：只使用本帧检测到的点做归属
        items_list = []
        baskets_list = []

        if ids is not None:
            # 先把框、坐标轴和 id 文本画出来（与 DB 无关）
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            draw_axis_if_possible(frame, corners, ids, K, dist, args.marker_length)
            for i, cid in enumerate(ids.flatten()):
                c = corners[i][0].mean(axis=0).astype(int)
                cv2.putText(frame, f"id={int(cid)}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # --- 同步到 presence.db（需要位姿条件：K/dist 和 marker_length 都已提供） ---
            if conn is not None and (K is not None) and (args.marker_length is not None):
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, float(args.marker_length), K, dist
                    )
                    # 1) 逐个写 presence / history，并按规则分类
                    for i, cid in enumerate(ids.flatten()):
                        t = tvecs[i, 0]   # (X, Y, Z) in meters
                        db_upsert_presence(conn, int(cid), t)
                        db_insert_history(conn, int(cid), t)

                        # 分类：筐子 or 物品
                        triple = (float(t[0]), float(t[1]), float(t[2]))
                        if is_basket_id(int(cid)):
                            baskets_list.append((int(cid), triple))
                        else:
                            items_list.append((int(cid), triple))

                    # 2) 本帧只调用一次归属计算
                    assign_items_to_baskets(
                        conn, items_list, baskets_list,
                        radius=getattr(args, 'assign_radius', 0.25)
                    )
                except Exception as e:
                    cv2.putText(frame, f"DB sync err: {str(e)[:40]}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # 每帧都做一次“超时下线”（与是否检测到/是否异常无关）
        if conn is not None:
            db_timeout_sweep(conn, args.presence_timeout)

        # ---- 显示 FPS / 预览 / 键盘 ----
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        cv2.imshow("ArUco", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC / q
            break

    # 循环结束后再释放
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

   

if __name__ == "__main__":
    main()