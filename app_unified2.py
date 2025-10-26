# app_unified.py — Unified Presence + History + Item→Basket + Run Detector + Rename Markers
# Pages:
#   /                  Presence table (click ID -> /history/<id>)
#   /history/<id>      Recent history table for an ID
#   /track             Pick an item, see its basket(s) (supports historical fallback)
#   /detector          Start/stop aruco_baskets.py, see live logs
# APIs:
#   /api/presence?only_present=0|1
#   /api/history/<id>?limit=100
#   /api/items
#   /api/track_item?item_id=<id>&window=<sec>&fallback=0|1&hist_window=<sec>
#   /api/detector/status
#   /api/detector/log?n=200
#   /detector/start (POST), /detector/stop (POST)
#   /api/markers/rename (POST)  <-- rename ArUco marker name (item/basket)
#   /health
#
# Env:
#   PRESENCE_DB       default: presence.db
#   BASKET_ID_MIN     default: 100
#   ARUCO_BASKETS     default: aruco_baskets.py
#   PYTHON_EXEC       default: current python executable
#
# Notes:
# - present_then = presence at the time of a record (via history near last_seen or presence.last_seen >= record time)
# - present_now  = current presence.present
# - Timestamps normalized: if >1e12 treat as milliseconds
# - Pandas to_html(..., justify="left") + CSS make headers and cells left-aligned
# - /track page supports rename selected item (and optional rename basket name inline)

import os
import sys
import time
import threading
import subprocess
import signal
from collections import deque

import sqlite3
import pandas as pd
from flask import (
    Flask, render_template_string, request, jsonify
)

DB = os.getenv("PRESENCE_DB", "presence.db")
BASKET_ID_MIN = int(os.getenv("BASKET_ID_MIN", "100"))
ARUCO_BASKETS = os.getenv("ARUCO_BASKETS", "aruco_baskets.py")
PYTHON_EXEC = os.getenv("PYTHON_EXEC", sys.executable or "python")

app = Flask(__name__)

# Normalize seconds/milliseconds
NORM = "CASE WHEN {col} > 1000000000000 THEN {col}/1000 ELSE {col} END"

# ---------------- SQLite helpers ----------------
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

def exec_sql(sql, params=()):
    conn = get_conn()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()

def query_df(sql, params=()):
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df

# ---------------- Detector manager ----------------
class DetectorManager:
    def __init__(self):
        self.proc = None
        self.thread = None
        self.log = deque(maxlen=5000)
        self.lock = threading.RLock()

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def _reader(self, pipe, prefix=""):
        for line in iter(pipe.readline, ''):
            if not line:
                break
            text = f"{prefix}{line.rstrip()}"
            with self.lock:
                self.log.append(text)
        try:
            pipe.close()
        except Exception:
            pass

    def start(self, args_dict):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return False, "Detector already running"

            # --- 构造命令：无缓冲输出 (-u) ---
            cmd = [PYTHON_EXEC, "-u", ARUCO_BASKETS]

            # 常用参数拼接
            camera = args_dict.get("camera", "").strip()
            if camera:
                cmd += ["--camera", camera]

            calib = args_dict.get("calib", "").strip()
            if calib:
                cmd += ["--calib", calib]

            marker_length = args_dict.get("marker_length", "").strip()
            if marker_length:
                cmd += ["--marker-length", marker_length]

            dbfile = args_dict.get("db", "").strip() or DB
            cmd += ["--db", dbfile]

            assign_radius = args_dict.get("assign_radius", "").strip()
            if assign_radius:
                cmd += ["--assign-radius", assign_radius]

            presence_timeout = args_dict.get("presence_timeout", "").strip()
            if presence_timeout:
                cmd += ["--presence-timeout", presence_timeout]

            basket_id_min = args_dict.get("basket_id_min", "").strip()
            if basket_id_min:
                cmd += ["--basket-id-min", basket_id_min]

            aruco_dict = args_dict.get("dict", "").strip()
            if aruco_dict:
                cmd += ["--dict", aruco_dict]

            save_path = args_dict.get("save", "").strip()
            if save_path:
                cmd += ["--save", save_path]

            if args_dict.get("debug_overlay", "") == "1":
                cmd += ["--debug-overlay"]


            # --- 日志：显示启动命令 ---
            self.log.clear()
            self.log.append(f"[launcher] starting: {' '.join(cmd)}")

            # --- 关键：工作目录 = 脚本所在目录（相对路径更稳） ---
            script_dir = os.path.dirname(os.path.abspath(ARUCO_BASKETS)) or None

            # --- 关键：无缓冲环境 ---
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            try:
                # 注意：移除 CREATE_NO_WINDOW，允许 GUI/弹窗正常显示
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                    cwd=script_dir,
                    env=env,
                    shell=False  # 更安全
                )
            except Exception as e:
                self.log.append(f"[launcher] start error: {e}")
                return False, f"Failed to start: {e}"

            # stdout/stderr 读线程
            if self.proc.stdout:
                threading.Thread(target=self._reader, args=(self.proc.stdout, "[out] "), daemon=True).start()
            if self.proc.stderr:
                threading.Thread(target=self._reader, args=(self.proc.stderr, "[err] "), daemon=True).start()

            threading.Thread(target=self._waiter, daemon=True).start()
            return True, "Detector started"


    def _waiter(self):
        p = self.proc
        if not p:
            return
        rc = p.wait()
        with self.lock:
            self.log.append(f"[launcher] detector exited with code {rc}")
            self.proc = None

    def stop(self):
        with self.lock:
            if self.proc is not None or self.proc.poll() is None:
                return False, "Detector not running"
            p = self.proc

        try:
            if os.name != "nt":
                p.send_signal(signal.SIGTERM)
            else:
                p.terminate()
        except Exception as e:
            with self.lock:
                self.log.append(f"[launcher] terminate error: {e}")

        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception as e2:
                with self.lock:
                    self.log.append(f"[launcher] kill error: {e2}")

        with self.lock:
            self.log.append("[launcher] detector stopped")
            self.proc = None
        return True, "Detector stopped"

    def tail(self, n=200):
        with self.lock:
            return list(self.log)[-n:]

detector = DetectorManager()

# ---------------- Templates ----------------
HTML_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Unified Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style> body { padding: 20px; } </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h4 mb-0">Unified Dashboard</h1>
    <div class="d-flex gap-2">
      <a class="btn btn-sm {{ 'btn-primary active' if not only_present else 'btn-outline-primary' }}" href="/" aria-current="{{ 'page' if not only_present else 'false' }}">All</a>
      <a class="btn btn-sm {{ 'btn-primary active' if only_present else 'btn-outline-primary' }}" href="/?only_present=1" aria-current="{{ 'page' if only_present else 'false' }}">Only Present</a>
      <a class="btn btn-sm btn-success" href="/track">Track Item → Basket</a>
      <a class="btn btn-sm btn-warning" href="/detector">Run Detector</a>
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
  <style>
    body { padding: 20px; }
    table thead th { text-align: left !important; }
    table tbody td { text-align: left !important; }
  </style>
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
  <title>Track：Item → Basket</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body { padding: 20px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .small { font-size: 12px; color: #6c757d; }
    .help-fixed { min-height: 2.25rem; }
  </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h4 mb-0">Track：Item → Basket</h1>
    <div class="d-flex gap-2">
      <a class="btn btn-sm btn-secondary" href="/">← Back</a>
      <a class="btn btn-sm btn-warning" href="/detector">Run Detector</a>
      <a class="btn btn-sm btn-outline-secondary" href="/health">Health</a>
    </div>
  </div>

  <div class="card shadow-sm mb-3">
    <div class="card-body">
      <div class="row g-3 align-items-end">
        <div class="col-12 col-md-5">
          <label class="form-label">Selected item</label>
          <div class="input-group">
            <select id="itemSelect" class="form-select"></select>
            <button class="btn btn-outline-secondary" type="button" id="btnRename">✏️ Rename</button>
          </div>
          <div class="form-text help-fixed">&nbsp;</div> 
        </div>
        <div class="col-6 col-md-3">
          <label class="form-label">Refresh time window（second）</label>
          <input id="winInput" type="number" min="1" value="30" class="form-control" />
          <div class="form-text help-fixed">&nbsp;</div> 
        </div>
        <div class="col-6 col-md-2">
          <label class="form-label">History fallback</label>
          <select id="fallback" class="form-select">
            <option value="1" selected>On</option>
            <option value="0">Off</option>
          </select>
          <div class="small mt-1">Fall back to the last recorded state when no new reading</div>
        </div>
        <div class="col-6 col-md-2">
          <label class="form-label">Show history with in（second）</label>
          <input id="histWin" type="number" min="0" value="2" class="form-control" />
          <div class="form-text help-fixed">&nbsp;</div> 
        </div>
        <div class="col-6 col-md-2">
          <label class="form-label d-block">Operation</label>
          <button class="btn btn-primary w-100" id="btnTrack">Start/Refresh</button>
        </div>
      </div>
    </div>
  </div>

  <div id="errorBox" class="alert alert-danger d-none"></div>

  <div class="card shadow-sm">
    <div class="card-body">
      <h5 class="card-title">Basket containing the item（current/latest）</h5>
      <div class="table-responsive">
        <table class="table table-striped table-sm">
          <thead>
            <tr>
              <th>Basket ID</th>
              <th>Name</th>
              <th>Item last appeared</th>
              <th>Present at the time</th>
              <th>Present right now</th>
              <th>Time when basket was last present</th>
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
        if(!data.ok){ setErr(data.error||'Failed to load item'); return; }
        const sel = document.getElementById('itemSelect');
        sel.innerHTML = data.items.map(it => {
          const name = (it.name && it.name.trim()) ? ` (${it.name})` : '';
          return `<option value="${it.id}">${it.id}${name}</option>`;
        }).join('');
      }catch(e){ setErr(String(e)); }
    }

    function badge(valTrue){ return (valTrue ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>'); }

    function renderBins(list){
      const body = document.getElementById('binsBody');
      body.innerHTML = list.map(row => {
        const nm = (row.basket_name||'');
        const nameCell = `${nm} <button class="btn btn-sm btn-link p-0 ms-1" onclick="renameMarker(${row.basket_id}, ${JSON.stringify(nm)})">✏️</button>`;
        const thenBadge = badge(row.present_then === 1);
        const nowBadge  = badge(row.present_now  === 1);
        const x = (row.x==null? '': Number(row.x).toFixed(3));
        const y = (row.y==null? '': Number(row.y).toFixed(3));
        const z = (row.z==null? '': Number(row.z).toFixed(3));
        return `<tr>
          <td class="mono">${row.basket_id}</td>
          <td>${nameCell}</td>
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
        if(!data.ok){ setErr(data.error||'Failed to track'); return; }
        renderBins(data.bins||[]);
      }catch(e){ setErr(String(e)); }
    }

    async function renameSelectedItem(){
      try{
        const sel = document.getElementById('itemSelect');
        const idStr = sel.value;
        if (idStr === "" || Number.isNaN(Number(idStr))) {alert('Select an item');return;}
        const id = Number(idStr);      // id 可以是 0

        const currentText = sel.options[sel.selectedIndex]?.text || '';
        const curName = (currentText.includes('(') && currentText.includes(')'))
          ? currentText.slice(currentText.indexOf('(')+1, currentText.indexOf(')')).trim()
          : '';

        const newName = prompt(`Rename item #${id}\nCurrent: ${curName || '(empty)'}\nType in the new name:`, curName);
        if(newName === null) return;
        const name = newName.trim();
        if(!name){ alert('Name can not be empty'); return; }

        const res = await fetch('/api/markers/rename', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id, name })
        });
        const j = await res.json();
        if(!j.ok){ alert('Failed to change name: ' + (j.error||'')); return; }

        await loadItems();
        await trackOnce();
        alert(` The name of #${id} has been changed to "${name}"`);
      }catch(e){
        alert('Name changing exception: ' + e);
      }
    }

    async function renameMarker(id, oldName=''){
      const newName = prompt(`Rename marker #${id}\nCurrent: ${oldName || '(empty)'}\nType in the new name:`, oldName||'');
      if(newName === null) return;
      const name = newName.trim();
      if(!name){ alert('Name can not be empty'); return; }
      const res = await fetch('/api/markers/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, name })
      });
      const j = await res.json();
      if(!j.ok){ alert('Failed to change name: ' + (j.error||'')); return; }
      await loadItems();
      await trackOnce();
      alert(` The name of #${id} has been changed to "${name}"`);
    }

    document.getElementById('btnTrack').addEventListener('click', trackOnce);
    document.getElementById('btnRename').addEventListener('click', renameSelectedItem);
    loadItems().then(trackOnce);
    setInterval(trackOnce, 3000);
  </script>
</body>
</html>
"""

HTML_DETECTOR = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Run Detector (aruco_baskets.py)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <style>
    body { padding: 20px; }
    textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; height: 360px; }
  </style>
</head>
<body>
  <div class="d-flex align-items-center justify-content-between mb-3">
    <h1 class="h5 mb-0">Run Detector (aruco_baskets.py)</h1>
    <div class="d-flex gap-2">
      <a class="btn btn-sm btn-secondary" href="/">← Back</a>
      <a class="btn btn-sm btn-outline-secondary" href="/health">Health</a>
    </div>
  </div>

  <form id="startForm" class="card mb-3 shadow-sm" onsubmit="return false;">
    <div class="card-body">
      <div class="row g-3">
        <div class="col-md-2">
          <label class="form-label">camera</label>
          <input name="camera" class="form-control" placeholder="0 or video path">
        </div>
        <div class="col-md-3">
          <label class="form-label">calib</label>
          <input name="calib" class="form-control" placeholder="calib.yaml">
        </div>
        <div class="col-md-2">
          <label class="form-label">marker-length (m)</label>
          <input name="marker_length" class="form-control" placeholder="0.03">
        </div>
        <div class="col-md-3">
          <label class="form-label">db (sqlite)</label>
          <input name="db" class="form-control" placeholder="presence.db" value="{{ db }}">
        </div>
        <div class="col-md-2">
          <label class="form-label">assign-radius (m)</label>
          <input name="assign_radius" class="form-control" placeholder="0.25">
        </div>
        <div class="col-md-2">
          <label class="form-label">basket-id-min (comma)</label>
          <input name="basket_id_min" class="form-control" placeholder="100">
        </div>
        <div class="col-md-2">
          <label class="form-label">presence-timeout (s)</label>
          <input name="presence_timeout" class="form-control" placeholder="2">
        </div>
        <div class="col-md-2">
          <label class="form-label">dict</label>
          <input name="dict" class="form-control" placeholder="DICT_6X6_250">
        </div>
        <div class="col-md-3">
          <label class="form-label">save (video)</label>
          <input name="save" class="form-control" placeholder="out.mp4">
        </div>
        <div class="col-md-2">
          <label class="form-label">debug overlay</label>
          <select name="debug_overlay" class="form-select">
            <option value="">off</option>
            <option value="1">on</option>
          </select>
        </div>
      </div>

      <div class="mt-3 d-flex gap-2">
        <button class="btn btn-primary" onclick="startDetector()">Start</button>
        <button class="btn btn-danger" onclick="stopDetector()">Stop</button>
        <span id="stat" class="ms-2 small text-muted">status: ...</span>
      </div>
    </div>
  </form>

  <div class="card shadow-sm">
    <div class="card-body">
      <h6 class="card-title">Live Log</h6>
      <textarea id="logbox" class="form-control" readonly></textarea>
    </div>
  </div>

  <script>
    async function startDetector(){
      const form = document.getElementById('startForm');
      const fd = new FormData(form);
      const res = await fetch('/detector/start', { method: 'POST', body: fd });
      const j = await res.json();
      alert(j.ok ? 'Started' : ('Failed: ' + j.error));
      await refreshStatus();
    }
    async function stopDetector(){
      const res = await fetch('/detector/stop', { method: 'POST' });
      const j = await res.json();
      alert(j.ok ? 'Stopped' : ('Failed: ' + j.error));
      await refreshStatus();
    }
    async function refreshStatus(){
      const res = await fetch('/api/detector/status');
      const j = await res.json();
      document.getElementById('stat').textContent = 'status: ' + (j.running ? 'running' : 'stopped');
    }
    async function refreshLog(){
      const res = await fetch('/api/detector/log?n=400');
      const j = await res.json();
      if(j.ok){
        const ta = document.getElementById('logbox');
        ta.value = (j.lines||[]).join('\\n');
        ta.scrollTop = ta.scrollHeight;
      }
    }
    refreshStatus();
    refreshLog();
    setInterval(refreshStatus, 2000);
    setInterval(refreshLog, 1000);
  </script>
</body>
</html>
"""

# ---------------- Pages ----------------
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
        table_html = df.to_html(classes='table table-striped table-sm', index=False, border=0, escape=False, justify="left")
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
        table_html = df.to_html(classes='table table-bordered table-sm', index=False, border=0, escape=False, justify="left")
        return render_template_string(HTML_HISTORY, table=table_html, error=None, id=id)
    except Exception as e:
        return render_template_string(HTML_HISTORY, table="", error=str(e), id=id), 500

@app.route("/track")
def track():
    return render_template_string(HTML_TRACK)

@app.route("/detector")
def detector_page():
    return render_template_string(HTML_DETECTOR, db=DB, basket_id_min=BASKET_ID_MIN)

# ---------------- APIs ----------------
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
    try:
        item_id   = int(request.args.get("item_id", "0"))
        window    = int(request.args.get("window", "30"))
        fallback  = request.args.get("fallback", "1") in ("1", "true", "True")
        hist_win  = float(request.args.get("hist_window", "2"))

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
                "present":      present_now,  # compatibility
                "present_last_seen": pres.get("present_last_seen_sec"),
                "present_last_seen_local": pres.get("present_last_seen_local"),
                "x": pres.get("x"), "y": pres.get("y"), "z": pres.get("z"),
            })
            out.append(merged)

        return jsonify({"ok": True, "bins": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---------- Rename marker (item/basket) ----------
def ensure_markers_table():
    exec_sql("""
        CREATE TABLE IF NOT EXISTS markers(
          id    INTEGER PRIMARY KEY,
          name  TEXT
        )
    """)

@app.route("/api/markers/rename", methods=["POST"])
def api_markers_rename():
    """
    JSON body: { "id": <int>, "name": "<new name>" }
    Upsert into markers: INSERT ... ON CONFLICT(id) DO UPDATE
    """
    try:
        payload = request.get_json(silent=True) or {}
        mid_raw = payload.get("id", None)
        try:
            mid = int(mid_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid id"})

        new_name = (payload.get("name") or "").strip()
        if not new_name:
            return jsonify({"ok": False, "error": "name cannot be empty"})


        ensure_markers_table()
        exec_sql("""
            INSERT INTO markers(id, name) VALUES(?, ?)
            ON CONFLICT(id) DO UPDATE SET name=excluded.name
        """, (mid, new_name))

        return jsonify({"ok": True, "id": mid, "name": new_name})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---------- Detector endpoints ----------
@app.route("/api/detector/status")
def api_detector_status():
    return jsonify({"ok": True, "running": detector.is_running(), "cmd": ARUCO_BASKETS, "python": PYTHON_EXEC})

@app.route("/api/detector/log")
def api_detector_log():
    try:
        n = int(request.args.get("n", "200"))
    except Exception:
        n = 200
    return jsonify({"ok": True, "lines": detector.tail(n)})

@app.route("/detector/start", methods=["POST"])
def detector_start():
    args_dict = {k: (v if isinstance(v, str) else v[0]) for k, v in request.form.items()}
    ok, msg = detector.start(args_dict)
    return jsonify({"ok": ok, "error": None if ok else msg, "args": args_dict})

@app.route("/detector/stop", methods=["POST"])
def detector_stop():
    ok, msg = detector.stop()
    return jsonify({"ok": ok, "error": None if ok else msg})

# ---------------- Health ----------------
@app.route("/health")
def health():
    try:
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
            "detector_running": detector.is_running(),
            "detector_script": os.path.abspath(ARUCO_BASKETS),
            "python_exec": PYTHON_EXEC,
            "latest_basket_item": latest
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "db": os.path.abspath(DB)}, 500

if __name__ == "__main__":
    app.run(debug=True)

