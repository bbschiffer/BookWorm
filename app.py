# app.py
from flask import Flask, render_template_string, request
import sqlite3, pandas as pd, os, traceback

DB = os.getenv("PRESENCE_DB", "presence.db")
app = Flask(__name__)

def query_df(sql, params=()):
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()
    return df

HTML_INDEX = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Presence Dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
</head>
<body class="p-4">
  <h1 class="mb-3">Presence Dashboard</h1>
  <div class="mb-2">
    <a class="btn btn-sm btn-primary" href=" ">全部</a >
    <a class="btn btn-sm btn-outline-success" href="/?only_present=1">只看在场</a >
    <a class="btn btn-sm btn-outline-secondary" href="/health">健康检查</a >
  </div>
  {% if error %}
    <div class="alert alert-danger" role="alert"><pre style="white-space:pre-wrap">{{ error }}</pre></div>
  {% else %}
    {{ table|safe }}
    <small class="text-muted d-block mt-2">提示：点击 ID 查看最近 50 条历史</small>
  {% endif %}
  <script>
    // 给第一列ID加链接
    document.querySelectorAll("table tbody tr").forEach(tr => {
      const td = tr.querySelector("td");
      if (!td) return;
      const id = td.innerText.trim();
      if (id && !isNaN(Number(id))) td.innerHTML = `<a href="/history/${id}">${id}</a >`;
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
</head>
<body class="p-4">
  <h1 class="mb-3">History for ID {{ id }}</h1>
  <a class="btn btn-sm btn-secondary mb-3" href="/">← 返回主页</a >
  {% if error %}
    <div class="alert alert-danger" role="alert"><pre style="white-space:pre-wrap">{{ error }}</pre></div>
  {% else %}
    {{ table|safe }}
  {% endif %}
</body>
</html>
"""

@app.route("/health")
def health():
    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close(); conn.close()
        return {"ok": True, "db": os.path.abspath(DB)}
    except Exception as e:
        return {"ok": False, "error": str(e), "db": os.path.abspath(DB)}, 500

@app.route("/")
def index():
    only_present = request.args.get("only_present") == "1"
    try:
        base_sql = """
            SELECT id, name, present,
                   datetime(last_seen,'unixepoch','localtime') as last_seen,
                   x, y, z
            FROM presence
        """
        if only_present:
            base_sql += " WHERE present = 1"
        base_sql += " ORDER BY id"
        df = query_df(base_sql)
        if df.empty:
            # 确保有表头可见
            df = df.reindex(columns=["id","name","present","last_seen","x","y","z"])
        table_html = df.to_html(classes='table table-striped table-sm', index=False, border=0, escape=False)
        return render_template_string(HTML_INDEX, table=table_html, error=None)
    except Exception:
        err = traceback.format_exc()
        print(err)
        return render_template_string(HTML_INDEX, table="", error=err), 500

@app.route("/history/<int:id>")
def history(id: int):
    try:
        sql = """
            SELECT id, name,
                   datetime(t,'unixepoch','localtime') AS ts,
                   x, y, z
            FROM history
            WHERE id = ?
            ORDER BY t DESC
            LIMIT 50
        """
        df = query_df(sql, (id,))
        if df.empty:
            df = df.reindex(columns=["id","name","ts","x","y","z"])
        table_html = df.to_html(classes='table table-bordered table-sm', index=False, border=0, escape=False)
        return render_template_string(HTML_HISTORY, table=table_html, error=None, id=id)
    except Exception:
        err = traceback.format_exc()
        print(err)
        return render_template_string(HTML_HISTORY, table="", error=err, id=id), 500

if __name__ == "__main__":
    # 打开调试模式，控制台能看到详细错误
    app.run(debug=True)

#http://127.0.0.1:5000/