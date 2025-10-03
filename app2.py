# app2.py — Unified GUI/API with "recent history" fallback
# - /api/items: item 下拉
# - /api/track_item: 先按 window 查最近归属；若无结果且 fallback=1，退回到最近一次归属（不受窗口限制）
# - /health: DB 健康检查与最近一条 basket_items
#
# 环境变量：
#   PRESENCE_DB       默认 presence.db
#   BASKET_ID_MIN     默认 100（仅用于 /api/items 的兜底：markers < BASKET_ID_MIN）

import os
import sqlite3
from flask import Flask, request, jsonify, render_template_string

DB = os.getenv("PRESENCE_DB", "presence.db")
BASKET_ID_MIN = int(os.getenv("BASKET_ID_MIN", "100"))

app = Flask(__name__)

# ---------- SQLite helpers ----------
def get_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def q_all(sql, params=()):
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def q_one(sql, params=()):
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

# 统一把 last_seen / t 归一化为“秒”（兼容毫秒）
# 在 SQL 里用 CASE WHEN last_seen>1e12 THEN last_seen/1000 ELSE last_seen END
NORM = "CASE WHEN {col} > 1000000000000 THEN {col}/1000 ELSE {col} END"

# ---------- Pages (简洁首页，便于测试) ----------
INDEX_HTML = """
<!doctype html>
<html lang="zh-CN"><head>
  <meta charset="utf-8">
  <title>统一面板（含历史回退）</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>body{padding:20px}</style>
</head><body>
  <h1 class="h4">统一面板（含历史回退）</h1>
  <div class="mb-2">
    <a class="btn btn-sm btn-outline-secondary" href="/health">健康检查</a>
  </div>

  <div class="card mb-3"><div class="card-body">
    <h5 class="card-title">跟踪：Item → Basket</h5>
    <div class="row g-2 align-items-end">
      <div class="col-4">
        <label class="form-label">Item ID</label>
        <input id="itemId" class="form-control" placeholder="例如 5">
      </div>
      <div class="col-3">
        <label class="form-label">时间窗口（秒）</label>
        <input id="win" type="number" class="form-control" value="30" min="1">
      </div>
      <div class="col-3">
        <label class="form-label">允许历史回退</label>
        <select id="fallback" class="form-select">
          <option value="1" selected>是（建议）</option>
          <option value="0">否</option>
        </select>
      </div>
      <div class="col-2">
        <button class="btn btn-primary w-100" onclick="track()">查询</button>
      </div>
    </div>
  </div></div>

  <pre id="out" class="bg-light p-2 border"></pre>

  <script>
  async function track(){
    const id = document.getElementById('itemId').value.trim();
    const win = document.getElementById('win').value || 30;
    const fb  = document.getElementById('fallback').value || 1;
    const url = `/api/track_item?item_id=${encodeURIComponent(id)}&window=${encodeURIComponent(win)}&fallback=${encodeURIComponent(fb)}`;
    const r = await fetch(url);
    const j = await r.json();
    document.getElementById('out').textContent = JSON.stringify(j, null, 2);
  }
  </script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

# ---------- API: item 列表 ----------
@app.route("/api/items")
def api_items():
    try:
        # 优先从 basket_items 中拿“出现过的 item”
        items = q_all("""
            SELECT DISTINCT m.id, COALESCE(m.name,'') AS name
            FROM basket_items bi
            JOIN markers m ON m.id = bi.item_id
            ORDER BY m.id
        """)
        if not items:
            # 兜底：markers 中 id < BASKET_ID_MIN 的都认为是 item 候选
            items = q_all("""
                SELECT id, COALESCE(name,'') AS name
                FROM markers
                WHERE id < ?
                ORDER BY id
            """, (BASKET_ID_MIN,))
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---------- API: track item -> 当前 bins（含历史回退） ----------
@app.route("/api/track_item")
def api_track_item():
    try:
        item_id = int(request.args.get("item_id", "0"))
        window  = int(request.args.get("window", "30"))
        # fallback=1：若窗口内查不到，就回退到“最近一次归属”
        fallback = request.args.get("fallback", "1") in ("1", "true", "True")

        # 1) 先查：窗口内的 basket 归属（按归一化秒比较）
        norm_bi = NORM.format(col="bi.last_seen")
        rows = q_all(f"""
            SELECT bi.basket_id,
                   b.name AS basket_name,
                   bi.last_seen AS last_seen_raw,
                   {norm_bi} AS last_seen_sec,
                   datetime({norm_bi},'unixepoch','localtime') AS last_seen
            FROM basket_items bi
            JOIN markers b ON b.id = bi.basket_id
            WHERE bi.item_id = ?
              AND {norm_bi} >= (strftime('%s','now') - ?)
            ORDER BY {norm_bi} DESC, bi.basket_id
        """, (item_id, window))

        # 2) 如窗口内无数据且 fallback=1，回退到“最近一次归属”
        if not rows and fallback:
            rows = q_all(f"""
                SELECT bi.basket_id,
                       b.name AS basket_name,
                       bi.last_seen AS last_seen_raw,
                       {norm_bi} AS last_seen_sec,
                       datetime({norm_bi},'unixepoch','localtime') AS last_seen
                FROM basket_items bi
                JOIN markers b ON b.id = bi.basket_id
                WHERE bi.item_id = ?
                ORDER BY {norm_bi} DESC
                LIMIT 1
            """, (item_id,))

        # 3) 合并 presence（若没有，再回退到 history 的最近一条位姿）
        out = []
        for r in rows:
            bid = r["basket_id"]

            # presence 兼容毫秒/秒时间戳
            pres = q_one(f"""
                SELECT id, name, present,
                       last_seen AS present_last_seen_raw,
                       {NORM.format(col="last_seen")} AS present_last_seen_sec,
                       datetime({NORM.format(col="last_seen")},'unixepoch','localtime') AS present_last_seen_local,
                       x, y, z
                FROM presence
                WHERE id = ?
            """, (bid,))

            if not pres:
                pres = {"id": bid, "name": r.get("basket_name"), "present": None,
                        "present_last_seen_raw": None, "present_last_seen_sec": None,
                        "present_last_seen_local": None, "x": None, "y": None, "z": None}

            # 如果 presence 没有位姿（x/y/z 为空），尝试从 history 拿最近一条补上
            if pres.get("x") is None or pres.get("y") is None or pres.get("z") is None:
                h = q_one(f"""
                    SELECT id, name, x, y, z, t AS hist_t_raw,
                           {NORM.format(col="t")} AS hist_t_sec,
                           datetime({NORM.format(col="t")},'unixepoch','localtime') AS hist_ts
                    FROM history
                    WHERE id = ?
                    ORDER BY {NORM.format(col="t")} DESC
                    LIMIT 1
                """, (bid,))
                if h:
                    # 只在 x/y/z 为空时用 history 的
                    for k in ("x", "y", "z"):
                        if pres.get(k) is None and h.get(k) is not None:
                            pres[k] = h[k]
                    # 如果 presence_last_seen 也为空，则补上 hist 时间
                    if pres.get("present_last_seen_local") is None:
                        pres["present_last_seen_local"] = h.get("hist_ts")
                        pres["present_last_seen_sec"] = h.get("hist_t_sec")
                        pres["present_last_seen_raw"] = h.get("hist_t_raw")

            merged = dict(r)
            merged.update({
                "present": pres.get("present"),
                "present_last_seen": pres.get("present_last_seen_sec"),
                "present_last_seen_local": pres.get("present_last_seen_local"),
                "x": pres.get("x"), "y": pres.get("y"), "z": pres.get("z"),
            })
            out.append(merged)

        return jsonify({"ok": True, "bins": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---------- health ----------
@app.route("/health")
def health():
    try:
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()

        # 显示 basket_items 最新 1 条，便于确认是否有数据 & 时间单位
        latest = q_one(f"""
            SELECT bi.item_id, bi.basket_id, bi.last_seen AS raw,
                   {NORM.format(col="bi.last_seen")} AS sec,
                   datetime({NORM.format(col="bi.last_seen")},'unixepoch','localtime') AS ts
            FROM basket_items bi
            ORDER BY {NORM.format(col="bi.last_seen")} DESC
            LIMIT 1
        """) or {}

        return {
            "ok": True,
            "db": os.path.abspath(DB),
            "basket_id_min": BASKET_ID_MIN,
            "latest_basket_item": latest
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "db": os.path.abspath(DB)}, 500

if __name__ == "__main__":
    # 建议：开发时用 debug=True；生产请改为 False 并放到 WSGI
    app.run(debug=True)
