# README – `aruco_basket.py` Usage

> This document is specifically for **aruco_basket.py**:  
> On top of the original ArUco detection flow (`markers / presence / history`), it adds maintenance of the **Basket–Item** relationship, so you can always know which basket contains which ArUco-tagged items.

---

## 1) Overview
- **Classification**: Divide detected markers into *baskets* and *items* based on ID rules.  
- **Assignment**: Each frame, assign items to the **nearest basket** within a 2D (XY-plane) radius threshold.  
- **Persistence**: Write basket–item relationships into `basket_items(basket_id, item_id, last_seen)`.  
- **Compatibility**: Does not break existing `markers / presence / history` tables.

Data flow: Camera → ArUco detection/pose → `presence`/`history` update → **classification** → **basket_items update**.

---

## 2) Dependencies and Startup
- Python 3.8+
- OpenCV (with `cv2.aruco` and calibration), NumPy, SQLite (standard library)
- FFmpeg recommended if you want to save video

**Example command**:
```bash
python aruco_baskets.py --camera 1 --dict DICT_6X6_250 --calib calib.yaml     --marker-length 0.03 --db presence.db     --assign-radius 0.25 --basket-id-min 100 --presence-timeout 2
```

Parameters:
- `--camera`: Camera index or video file path  
- `--marker-length`: Real-world ArUco marker side length (meters)  
- `--db`: SQLite database file path  
- `--assign-radius`: XY-plane distance threshold (meters, default 0.25) for assigning items to baskets  
- `--basket-id-min`: IDs ≥ this value are considered baskets; smaller IDs are items (default 100)  

If using a calibration file, provide it as in the original script (`--calib`).

---

## 3) New Database Objects (auto-created)
```sql
CREATE TABLE IF NOT EXISTS basket_items(
  basket_id INTEGER,
  item_id   INTEGER,
  last_seen REAL NOT NULL,
  PRIMARY KEY (basket_id, item_id),
  FOREIGN KEY(basket_id) REFERENCES markers(id),
  FOREIGN KEY(item_id)   REFERENCES markers(id)
);
CREATE INDEX IF NOT EXISTS ix_basket_items_basket ON basket_items(basket_id);
CREATE INDEX IF NOT EXISTS ix_basket_items_item   ON basket_items(item_id);
```

- `basket_id`: basket ArUco ID  
- `item_id`: item ArUco ID  
- `last_seen`: last time (epoch seconds) the item was observed inside that basket  

The original `markers / presence / history` remain untouched.

---

## 4) Classification and Assignment
### 4.1 Classification (ID → basket/item)
- Default: IDs `>= basket-id-min` → **basket**; otherwise → **item**.  
- Can override with `--basket-id-min`; for custom rules, edit `is_basket_id(...)`.

### 4.2 Assignment (nearest + radius filter)
- Only assign when XY-plane distance ≤ `assign-radius`.  
- Each item is assigned to the **nearest single basket**.  
- If you want “one item in multiple baskets,” modify the code to insert all within threshold.

---

## 5) Common SQL Queries
### 5.1 Show all baskets and their items
```sql
SELECT b.id   AS basket_id,
       b.name AS basket_name,
       m.id   AS item_id,
       m.name AS item_name,
       datetime(bi.last_seen,'unixepoch','localtime') AS last_seen
FROM basket_items bi
JOIN markers b ON bi.basket_id = b.id
JOIN markers m ON bi.item_id   = m.id
ORDER BY basket_id, item_id;
```

### 5.2 Current contents of a basket (last N seconds)
```sql
SELECT m.id, m.name
FROM basket_items bi
JOIN markers m ON m.id = bi.item_id
WHERE bi.basket_id = :basket
  AND bi.last_seen >= (strftime('%s','now') - :seconds)
ORDER BY m.id;
```
Example: `:basket=101, :seconds=30`.

### 5.3 Where did an item last appear?
```sql
SELECT basket_id,
       datetime(last_seen,'unixepoch','localtime') AS ts
FROM basket_items
WHERE item_id = :item
ORDER BY last_seen DESC
LIMIT 1;
```

---

## 6) Tuning and Recommendations
- **Radius threshold**: in meters (same unit as `tvec`). Start with 0.25; adjust to 0.15–0.35 based on setup and jitter.  
- **Smoothing**: Apply moving average / Kalman filter to (x,y,z), or enlarge threshold.  
- **Advanced geometry**: If baskets have 4 corner markers, replace circle check with “point inside quadrilateral.”  
- **Indexes**: Already indexed by `basket_id`/`item_id`. Add `last_seen` index if filtering by time is frequent.  
- **Integration with `presence`**: `presence` shows in-scene state; `basket_items` shows historical assignment.

---

## 7) FAQ
**Q1: Why no basket–item data?**  
- Ensure ArUco detection works (borders/ID/axes visible)  
- Ensure calibration + `marker-length` provided (otherwise no `tvec`)  
- Increase `--assign-radius` (e.g. 0.30)  
- Verify `--basket-id-min` matches your label planning  

**Q2: Assignment unstable?**  
- Adjust radius; smooth positions; or switch to polygon containment.

**Q3: More visual GUI?**  
- Add `/baskets` page in Flask showing baskets and current items  
- Plot X/Y/Z curves and XY trajectories under `/history/<id>`  
- Add CSV export, threshold sliders, presence-only filters

---

## 8) Version Highlights (`aruco_basket.py`)
- New args: `--assign-radius` (default 0.25m), `--basket-id-min` (default 100)  
- New functions: `is_basket_id(aruco_id)`, `assign_items_to_baskets(...)`  
- Main loop: update `presence/history + classification` → at end of frame, update `basket_items`.

---

## 9) Debugging
- If you see `DB sync err: ...` in overlay, DB write failed (check path/tables/permissions).  
- SQLite integrity check:
```sql
PRAGMA integrity_check;
SELECT name FROM sqlite_master WHERE type IN ('table','index') ORDER BY 1;
```
- Reclaim space (after many deletes):
```sql
PRAGMA page_count;
PRAGMA freelist_count;
VACUUM;
```

---

## 10) License
Follows the same license as the original project (or internal use by default).
