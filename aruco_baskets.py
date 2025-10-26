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

def R_from_z_aligned(z_axis, x_hint=np.array([1,0,0.0])):
    z = z_axis / (np.linalg.norm(z_axis) + 1e-9)
    xh = x_hint / (np.linalg.norm(x_hint) + 1e-9)
    y = np.cross(z, xh); n = np.linalg.norm(y)
    if n < 1e-6:
        xh = np.array([0,1,0.0]); y = np.cross(z, xh); n = np.linalg.norm(y)
    y /= n
    x = np.cross(y, z)
    return np.stack([x,y,z], axis=1)  # 列为基

# ==== 四元数工具 ====
def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> (w,x,y,z)，数值稳定"""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = np.argmax([m[0,0], m[1,1], m[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + m[0,0]-m[1,1]-m[2,2]) * 2.0
            w = (m[2,1]-m[1,2]) / s; x = 0.25*s
            y = (m[0,1]+m[1,0]) / s; z = (m[0,2]+m[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1,1]-m[0,0]-m[2,2]) * 2.0
            w = (m[0,2]-m[2,0]) / s; x = (m[0,1]+m[1,0]) / s
            y = 0.25*s; z = (m[1,2]+m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2]-m[0,0]-m[1,1]) * 2.0
            w = (m[1,0]-m[0,1]) / s; x = (m[0,2]+m[2,0]) / s
            y = (m[1,2]+m[2,1]) / s; z = 0.25*s
    q = np.array([w,x,y,z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3"""
    w,x,y,z = q / (np.linalg.norm(q) + 1e-12)
    xx,yy,zz = x*x,y*y,z*z
    xy,xz,yz = x*y,x*z,y*z
    wx,wy,wz = w*x,w*y,w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """球面插值，alpha∈[0,1]"""
    q0 = q0 / (np.linalg.norm(q0)+1e-12)
    q1 = q1 / (np.linalg.norm(q1)+1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha*(q1 - q0)
        return q / (np.linalg.norm(q)+1e-12)
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1-alpha)*theta0) / (sin0 + 1e-12)
    s1 = np.sin(alpha*theta0)     / (sin0 + 1e-12)
    return s0*q0 + s1*q1

# ==== 机械手驱动占位（把这里替换为你的SDK调用即可）====
class RobotDriver:
    def __init__(self, workspace_box=np.array([[-0.5,0.5],[-0.5,0.5],[0.0,0.8]])):
        self.workspace_box = workspace_box  # x,y,z范围

    def in_workspace(self, p):
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = self.workspace_box
        return (xmin <= p[0] <= xmax) and (ymin <= p[1] <= ymax) and (zmin <= p[2] <= zmax)

    def move_pose(self, T_base_ee):
        # TODO: 这里改为你的机械手移动指令。当前仅打印。
        p = T_base_ee[:3,3]; R = T_base_ee[:3,:3]
        if not self.in_workspace(p):
            print("[SAFE] target out of workspace:", p)
            return False
        # 简单转欧拉ZYX打印
        sy = (R[0,0]**2 + R[1,0]**2)**0.5
        if sy < 1e-6:
            rx = np.arctan2(-R[1,2], R[1,1]); ry = np.arctan2(-R[2,0], sy); rz = 0.0
        else:
            rx = np.arctan2(R[2,1], R[2,2]); ry = np.arctan2(-R[2,0], sy); rz = np.arctan2(R[1,0], R[0,0])
        print(f"[CMD] xyz={p.round(3)} rpy={[rx,ry,rz]}")
        return True

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

def assign_items_to_baskets(conn, items, baskets, radius=0.1):
    """将物品分配到最近的筐子（XY 平面，单位与 tvec 一致；默认 0.25m 半径）。
    items   = [(item_id, (x,y,z))]
    baskets = [(basket_id, (x,y,z))]
    """
    if not items or not baskets:
        return
    cur = conn.cursor()
    now = time.time()
    for iid, (ix,iy,iz) in items:
        # 选最近的筐子
        best = None
        best_d2 = None
        for bid, (bx,by,bz) in baskets:
            dx, dy = ix-bx, iy-by
            d2 = dx*dx + dy*dy
            if (best is None) or (d2 < best_d2):
                best = bid; best_d2 = d2
        if best is None:
            continue
        # 半径过滤
        if (best_d2 is not None) and (best_d2 <= (radius*radius)):
            cur.execute("""
                INSERT INTO basket_items(basket_id, item_id, last_seen)
                VALUES(?,?,?)
                ON CONFLICT(basket_id, item_id)
                DO UPDATE SET last_seen = excluded.last_seen
            """, (int(best), int(iid), now))
    conn.commit()

# --------- 兼容 OpenCV 4.7+ / 旧版 API ----------
def get_detector(aruco_dict_name: str):
    # 字典名到 OpenCV 常量
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
        raise ValueError(f"未知字典 {aruco_dict_name}，可选：{', '.join(table)}")

    dictionary = cv2.aruco.getPredefinedDictionary(table[name])

    # 4.7+ 新接口
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        def detect(gray):
            return detector.detectMarkers(gray)
        return detect, dictionary, params
    else:
        # 旧接口
        params = cv2.aruco.DetectorParameters_create()
        def detect(gray):
            return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        return detect, dictionary, params

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

        # 你也可以改成厘米显示：dist_cm = dist_m*100
        cv2.putText(frame, f"Z={Z_m:.3f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"|t|={dist_m:.3f} m", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


def generate_marker(out_path: str, dict_name: str, marker_id: int, side_px: int = 600):
    # 将字典名映射为 OpenCV 常量（整数）
    const = getattr(cv2.aruco, dict_name.upper(), None)
    if const is None:
        raise ValueError(f"未知字典 {dict_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(const)

    # 生成并保存
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"[OK] 已生成 {dict_name} id={marker_id} 到 {out.resolve()}")

# ------------------ 伺服主流程 ------------------
def run_servo(cam_id, aruco_const, marker_len_m, cam_yaml, handeye_yaml,
              id_filter=None, z_offset=0.04, smoothing=0.4, rot_smoothing=None):
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

    # 平滑状态
    p_smooth = None
    q_smooth = None
    if rot_smoothing is None:
        rot_smoothing = smoothing

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

                # ---- 旋转 SLERP 平滑 ----
                R_target = T_base_ee_target[:3,:3]
                q_target = mat_to_quat(R_target)
                if q_smooth is None:
                    q_smooth = q_target
                else:
                    q_smooth = quat_slerp(q_smooth, q_target, float(rot_smoothing))
                T_base_ee_target[:3,:3] = quat_to_mat(q_smooth)

                # 下发
                robot.move_pose(T_base_ee_target)

                # 可视化
                cv2.aruco.drawDetectedMarkers(frame, [corners[pick]], ids[pick:pick+1])
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_len_m*0.5)

        cv2.imshow("servo", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): break

    cap.release(); cv2.destroyAllWindows()

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
    ap.add_argument("--make", type=str, help="生成一个标记图片到该路径(PNG/JPG),与 --make-id 搭配")
    ap.add_argument("--make-id", type=int, default=0, help="生成标记的 id")
    ap.add_argument("--servo", action="store_true", help="启用伺服模式：根据位姿估计控制机械手")
    ap.add_argument("--handeye", type=str, help="手眼标定文件(yaml/json,含 4x4 T_base_cam)")
    ap.add_argument("--id", type=int, dest="follow_id", help="仅跟随指定 id")
    ap.add_argument("--smoothing", type=float, default=0.4, help="位置平滑系数(0-1)，默认0.4")
    ap.add_argument("--rot-smoothing", type=float, help="旋转平滑系数(0-1)，默认与 --smoothing 相同")
    ap.add_argument("--db", type=str, help="SQLite 数据库路径（例如 presence.db）。提供则启用在场同步")
    ap.add_argument("--presence-timeout", type=float, default=2.0, help="超时下线秒数，默认2s")
    args = ap.parse_args()
    globals()['BASKET_ID_MIN'] = int(getattr(args, 'basket_id_min', BASKET_ID_MIN))


    # 伺服模式
    if args.servo:
        if not (args.calib and args.handeye and args.marker_length):
            raise SystemExit("--servo 需要同时提供 --calib、--handeye、--marker-length")
        # 将字典名转为 OpenCV 常量
        const = getattr(cv2.aruco, args.dict.upper())
        run_servo(cam_id=args.camera,
            aruco_const=const,
            marker_len_m=args.marker_length,
            cam_yaml=args.calib,
            handeye_yaml=args.handeye,
            id_filter=args.follow_id,
            z_offset=0.04,
            smoothing=args.smoothing if hasattr(args, "smoothing") and args.smoothing is not None else 0.4,
            rot_smoothing=args.rot_smoothing)
        return

    # 仅生成标记
    if args.make:
        generate_marker(args.make, args.dict, args.make_id)
        return

    detect, dictionary, params = get_detector(args.dict)

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

    # 可选：数据库同步
    conn = None
    if args.db:
        conn = db_init(args.db)
        print(f"[INFO] presence DB: {args.db}")

    # 摄像头模式
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERR] 打不开摄像头 {args.camera}")
        sys.exit(1)

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