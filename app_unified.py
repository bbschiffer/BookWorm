# app_unified.py — Unified Presence Dashboard + Item→Basket Tracking
# Pages:
#   /                 Presence table (click ID -> /history/<id>)
#   /history/<id>     Recent history table for an ID
#   /track            Pick an item, see its basket(s) (supports historical fallback)
# APIs:
#   /api/presence?only_present=0|1
#   /api/history/<id>?limit=100
#   /api/items
#   /api/track_item?item_id=<id>&window=<sec>&fallback=0|1&hist_window=<sec>
#   /health
#
# Env:
#   PRESENCE_DB   default: presence.db
#   BASKET_ID_MIN default: 100 (fallback list from markers where id < this)
#
# Notes:
# - present_then = whether the basket was IN at the time of that record (based on history near last_seen)
# - present_now  = presence.present as of now
# - Timestamps are normalized: if >1e12 treat as milliseconds (divide by 1000)

import os
import sqlite3
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

DB = os.getenv("PRESENCE_DB", "presence.db")
BASKET_ID_MIN = int(os.getenv("BASKET_ID_MIN", "100"))

app = Flask(__name__)

# ------------- SQLite helpers -------------
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

def query_df(sql, params=()):
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df

# Normalize seconds/milliseconds
NORM = "CASE WHEN {col} > 1000000000000 THEN {col}/1000 ELSE {col} END"

# ------------- Templates -------------
HTML_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Unified Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body { padding: 20px; }
    .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; }
    .dot-on { background:#28a745; }
    .dot-off { background:#6c757d; }
  </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h4 mb-0">Unified Dashboard</h1>
    <div class="d-flex gap-2">
      <a class="btn btn-sm {{ 'btn-primary active' if not only_present else 'btn-outline-primary' }}"
         href="/" aria-current="{{ 'page' if not only_present else 'false' }}">
         All
      </a>
      <a class="btn btn-sm {{ 'btn-primary active' if only_present else 'btn-outline-primary' }}"
         href="/?only_present=1" aria-current="{{ 'page' if only_present else 'false' }}">
         Only Present
      </a>
      <a class="btn btn-sm btn-success" href="/track">Track Item → Basket</a>
      <a class="btn btn-sm btn-outline-secondary" href="/health">Health</a>
    </div> 
  </div>

  {% if error %}
    <div class="alert alert-danger" role="alert"><pre style="white-space:pre-wrap">{{ error }}</pre></div>
  {% else %}
    {{ table|safe }}
    <small class="text-muted d-block mt-2">Tip: click the ID to view its recent history.</small>
  {% endif %}

  <script>
    // First column becomes a link to /history/<id>
    document.querySelectorAll("table tbody tr").forEach(tr => {
      const td = tr.querySelector("td");
      if (!td) return;
      const id = td.innerText.trim();
      if (id && !isNaN(Number(id))) td.innerHTML = `<a href="/history/${id}">${id}</a>`;
    });
  </script>
</body>
</html>
"""

HTML_HISTORY = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>History for {{id}}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style> body { padding: 20px; } </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h4 mb-0">History for ID {{ id }}</h1>
    <a class="btn btn-sm btn-secondary" href="/">← Back</a>
  </div>
  {% if error %}
    <div class="alert alert-danger" role="alert"><pre style="white-space:pre-wrap">{{ error }}</pre></div>
  {% else %}
    {{ table|safe }}
  {% endif %}
</body>
</html>
"""

HTML_TRACK = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>追踪：Item → Basket</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body { padding: 20px; }
    .badge-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }
    .on { background:#28a745; } .off { background:#6c757d; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h4 mb-0">追踪：Item → Basket</h1>
    <div class="d-flex gap-2">
      <a class="btn btn-sm btn-secondary" href="/">← Back</a>
      <a class="btn btn-sm btn-outline-secondary" href="/health">Health</a>
    </div>
  </div>

  <div class="card shadow-sm mb-3">
    <div class="card-body">
      <div class="row g-3 align-items-end">
        <div class="col-12 col-md-4">
          <label class="form-label">选择物品（item）</label>
          <select id="itemSelect" class="form-select"></select>
        </div>
        <div class="col-6 col-md-3">
          <label class="form-label">时间窗口（秒）</label>
          <input id="winInput" type="number" min="1" value="30" class="form-control" />
        </div>
        <div class="col-6 col-md-2">
          <label class="form-label">历史回退</label>
          <select id="fallback" class="form-select">
            <option value="1" selected>开</option>
            <option value="0">关</option>
          </select>
        </div>
        <div class="col-6 col-md-3">
          <label class="form-label">历史近邻窗（秒）</label>
          <input id="histWin" type="number" min="0" value="2" class="form-control" />
        </div>
        <div class="col-6 col-md-2">
          <label class="form-label d-block">操作</label>
          <button class="btn btn-primary w-100" id="btnTrack">开始/刷新</button>
        </div>
      </div>
    </div>
  </div>

  <div id="errorBox" class="alert alert-danger d-none"></div>

  <div class="card shadow-sm">
    <div class="card-body">
      <h5 class="card-title">包含该物品的筐子（当前/最近）</h5>
      <div class="table-responsive">
        <table class="table table-striped table-sm">
          <thead>
            <tr>
              <th>篮ID</th>
              <th>名称</th>
              <th>物品最后出现</th>
              <th>当时在场</th>
              <th>现在在场</th>
              <th>篮最后在场时间</th>
              <th>X</th><th>Y</th><th>Z</th>
            </tr>
          </thead>
          <tbody id="binsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const err = document.getElementById('errorBox');
    function setErr(msg){ err.textContent = msg; err.classList.remove('d-none'); }
    function clrErr(){ err.textContent=''; err.classList.add('d-none'); }

    async function loadItems(){
      try{
        const res = await fetch('/api/items');
        const data = await res.json();
        if(!data.ok){ setErr(data.error||'加载物品失败'); return; }
        const sel = document.getElementById('itemSelect');
        sel.innerHTML = data.items.map(it => {
          const name = (it.name && it.name.trim()) ? ` (${it.name})` : '';
          return `<option value="${it.id}">${it.id}${name}</option>`;
        }).join('');
      }catch(e){ setErr(String(e)); }
    }

    function badgeYesNo(valTrue){
      return (valTrue ? '<span class="badge bg-success">在</span>' : '<span class="badge bg-secondary">不在</span>');
    }

    function renderBins(list){
      const body = document.getElementById('binsBody');
      body.innerHTML = list.map(row => {
        const thenBadge = badgeYesNo(row.present_then === 1);
        const nowBadge  = badgeYesNo(row.present_now  === 1);
        const x = (row.x==null? '': Number(row.x).toFixed(3));
        const y = (row.y==null? '': Number(row.y).toFixed(3));
        const z = (row.z==null? '': Number(row.z).toFixed(3));
        return `<tr>
          <td class="mono">${row.basket_id}</td>
          <td>${row.basket_name||''}</td>
          <td>${row.last_seen||''}</td>
          <td>${thenBadge}</td>
          <td>${nowBadge}</td>
          <td>${row.present_last_seen_local||''}</td>
          <td>${x}</td><td>${y}</td><td>${z}</td>
        </tr>`;
      }).join('');
    }

    async function trackOnce(){
      try{
        clrErr();
        const itemId = document.getElementById('itemSelect').value;
        const win = document.getElementById('winInput').value || 30;
        const fb  = document.getElementById('fallback').value || 1;
        const hw  = document.getElementById('histWin').value || 2;
        const res = await fetch(`/api/track_item?item_id=${encodeURIComponent(itemId)}&window=${encodeURIComponent(win)}&fallback=${encodeURIComponent(fb)}&hist_window=${encodeURIComponent(hw)}`);
        const data = await res.json();
        if(!data.ok){ setErr(data.error||'跟踪失败'); return; }
        renderBins(data.bins||[]);
      }catch(e){ setErr(String(e)); }
    }

    document.getElementById('btnTrack').addEventListener('click', trackOnce);
    loadItems().then(trackOnce);
    setInterval(trackOnce, 3000);
  </script>
</body>
</html>
"""

# ------------- Pages -------------
@app.route("/")
def index():
    only_present = request.args.get("only_present") == "1"
    try:
        base_sql = f"""
            SELECT id, name, present,
                   datetime({NORM.format(col="last_seen")},'unixepoch','localtime') AS last_seen,
                   x, y, z
            FROM presence
        """
        if only_present:
            base_sql += " WHERE present = 1"
        base_sql += " ORDER BY id"
        df = query_df(base_sql)
        if df.empty:
            df = df.reindex(columns=["id","name","present","last_seen","x","y","z"])
        table_html = df.to_html(classes='table table-striped table-sm', index=False, border=0, escape=False)
        return render_template_string(HTML_INDEX, table=table_html, error=None, only_present=only_present)
    except Exception as e:
        return render_template_string(HTML_INDEX, table="", error=str(e), only_present=only_present), 500

@app.route("/history/<int:id>")
def history(id: int):
    try:
        sql = f"""
            SELECT id, name,
                   datetime({NORM.format(col="t")},'unixepoch','localtime') AS ts,
                   x, y, z
            FROM history
            WHERE id = ?
            ORDER BY {NORM.format(col="t")} DESC
            LIMIT 200
        """
        df = query_df(sql, (id,))
        if df.empty:
            df = df.reindex(columns=["id","name","ts","x","y","z"])
        table_html = df.to_html(classes='table table-bordered table-sm', index=False, border=0, escape=False)
        return render_template_string(HTML_HISTORY, table=table_html, error=None, id=id)
    except Exception as e:
        return render_template_string(HTML_HISTORY, table="", error=str(e), id=id), 500

@app.route("/track")
def track():
    return render_template_string(HTML_TRACK)

# ------------- APIs -------------
@app.route("/api/presence")
def api_presence():
    only_present = request.args.get("only_present") == "1"
    sql = f"""
        SELECT id, name, present,
               datetime({NORM.format(col="last_seen")},'unixepoch','localtime') AS last_seen,
               {NORM.format(col="last_seen")} AS last_seen_unix,
               x, y, z
        FROM presence
    """
    if only_present:
        sql += " WHERE present = 1"
    sql += " ORDER BY id"
    try:
        df = query_df(sql)
        present_count = int((df['present'] == 1).sum()) if not df.empty else 0
        total = int(len(df)) if not df.empty else 0
        data = df.to_dict(orient="records") if not df.empty else []
        return jsonify({"ok": True, "items": data, "present_count": present_count, "total": total})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/history/<int:id>")
def api_history(id: int):
    limit = int(request.args.get("limit", "100"))
    try:
        df = query_df(f"""
            SELECT id, name, t,
                   datetime({NORM.format(col="t")},'unixepoch','localtime') AS ts,
                   x, y, z
            FROM history
            WHERE id = ?
            ORDER BY {NORM.format(col="t")} DESC
            LIMIT ?
        """, (id, limit))
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)  # chronological
            items = df.to_dict(orient="records")
        else:
            items = []
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/items")
def api_items():
    try:
        items = q_all("""
            SELECT DISTINCT m.id, COALESCE(m.name,'') AS name
            FROM basket_items bi
            JOIN markers m ON m.id = bi.item_id
            ORDER BY m.id
        """)
        if not items:
            items = q_all("""
                SELECT id, COALESCE(name,'') AS name
                FROM markers
                WHERE id < ?
                ORDER BY id
            """, (BASKET_ID_MIN,))
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/api/track_item")
def api_track_item():
    """
    Returns for each candidate basket:
    - present_then: was it present around last_seen time (history within ±hist_window sec, or presence.last_seen >= record time)
    - present_now:  current presence.present
    - present_last_seen_local: last time present (localtime)
    - x,y,z: pose (from presence; if missing, fallback to latest history)
    """
    try:
        item_id   = int(request.args.get("item_id", "0"))
        window    = int(request.args.get("window", "30"))
        fallback  = request.args.get("fallback", "1") in ("1", "true", "True")
        hist_win  = float(request.args.get("hist_window", "2"))

        norm_bi = NORM.format(col="bi.last_seen")

        # 1) Windowed assignments
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

        # 2) Fallback: latest assignment if none in window
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

        out = []
        for r in rows:
            bid = r["basket_id"]
            last_seen_sec = float(r["last_seen_sec"])

            # A) presence now
            pres = q_one(f"""
                SELECT id, name, present,
                       last_seen AS present_last_seen_raw,
                       {NORM.format(col="last_seen")} AS present_last_seen_sec,
                       datetime({NORM.format(col="last_seen")},'unixepoch','localtime') AS present_last_seen_local,
                       x, y, z
                FROM presence
                WHERE id = ?
            """, (bid,))

            present_now = None
            if pres and pres.get("present") is not None:
                present_now = 1 if int(pres["present"]) == 1 else 0

            if not pres:
                pres = {"id": bid, "name": r.get("basket_name"), "present": None,
                        "present_last_seen_raw": None, "present_last_seen_sec": None,
                        "present_last_seen_local": None, "x": None, "y": None, "z": None}

            # Fill pose from latest history if missing
            if pres.get("x") is None or pres.get("y") is None or pres.get("z") is None:
                h_pose = q_one(f"""
                    SELECT id, name, x, y, z, t AS hist_t_raw,
                           {NORM.format(col="t")} AS hist_t_sec,
                           datetime({NORM.format(col="t")},'unixepoch','localtime') AS hist_ts
                    FROM history
                    WHERE id = ?
                    ORDER BY {NORM.format(col="t")} DESC
                    LIMIT 1
                """, (bid,))
                if h_pose:
                    for k in ("x","y","z"):
                        if pres.get(k) is None and h_pose.get(k) is not None:
                            pres[k] = h_pose[k]
                    if pres.get("present_last_seen_local") is None:
                        pres["present_last_seen_local"] = h_pose.get("hist_ts")
                        pres["present_last_seen_sec"]   = h_pose.get("hist_t_sec")
                        pres["present_last_seen_raw"]   = h_pose.get("hist_t_raw")

            # B) present THEN (near last_seen)
            h_then = q_one(f"""
                SELECT 1 AS ok
                FROM history
                WHERE id = ?
                  AND {NORM.format(col="t")} BETWEEN ? AND ?
                LIMIT 1
            """, (bid, last_seen_sec - hist_win, last_seen_sec + hist_win))

            if h_then:
                present_then = 1
            else:
                pls = pres.get("present_last_seen_sec")
                present_then = 1 if (pls is not None and float(pls) >= last_seen_sec) else 0

            merged = dict(r)
            merged.update({
                "present_then": present_then,
                "present_now":  present_now,
                "present":      present_now,  # kept for compatibility with older frontends
                "present_last_seen": pres.get("present_last_seen_sec"),
                "present_last_seen_local": pres.get("present_last_seen_local"),
                "x": pres.get("x"), "y": pres.get("y"), "z": pres.get("z"),
            })
            out.append(merged)

        return jsonify({"ok": True, "bins": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ------------- Health -------------
@app.route("/health")
def health():
    try:
        # DB reachable?
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()

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
    app.run(debug=True)
