"""
阶段 7：把网球 Agent 做成服务。

- POST /api/analyze  提交任务，立即返回 thread_id（不阻塞）
- WS   /ws/{id}      订阅该任务的进度事件
- GET  /             最小演示页

关键点：
1. 编排是同步阻塞的，必须丢到线程里跑，否则卡死事件循环
2. 线程 → 事件循环用 call_soon_threadsafe + asyncio.Queue 桥接
3. 后台任务要保存引用，否则可能被 GC（3.6 学过的坑）
"""

import asyncio
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.agent.runner import stream_run   # noqa: E402

app = FastAPI(title="Tennis Coach Agent")

UPLOAD_ROOT = (PROJECT_ROOT / "data" / "uploads").resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 200
ALLOWED_SUFFIX = {".mp4", ".mov", ".avi", ".mkv"}

# 进程内连接池：thread_id -> {websocket}
CONNECTIONS: dict[str, set[WebSocket]] = {}
# 后台任务引用，防止被 GC
TASKS: set[asyncio.Task] = set()


class AnalyzeRequest(BaseModel):
    query: str
    thread_id: str | None = None
    user_id: str | None = None


async def broadcast(thread_id: str, event: dict):
    for ws in list(CONNECTIONS.get(thread_id, ())):
        try:
            await ws.send_json(event)
        except Exception:
            CONNECTIONS.get(thread_id, set()).discard(ws)


async def run_task(query: str, thread_id: str, user_id: str):
    """在线程里跑同步编排，把事件搬回事件循环广播出去。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for ev in stream_run(query, thread_id, user_id):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, name=f"agent-{thread_id[:8]}", daemon=True).start()

    while True:
        event = await queue.get()
        if event is None:
            break
        await broadcast(thread_id, event)


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    task = asyncio.create_task(run_task(req.query, thread_id,
                                        req.user_id or "anonymous"))
    TASKS.add(task)
    task.add_done_callback(TASKS.discard)      # 必须保存引用并清理
    return {"status": "started", "thread_id": thread_id}


@app.websocket("/ws/{thread_id}")
async def ws_endpoint(websocket: WebSocket, thread_id: str):
    await websocket.accept()
    CONNECTIONS.setdefault(thread_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()      # 心跳
            await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        CONNECTIONS.get(thread_id, set()).discard(websocket)


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    thread_id: str = Form(...),
):
    """接收击球视频。

    安全要点：
    1. 文件名只取 basename，杜绝 ../ 穿越
    2. 目录名做了 UUID 校验，防止跳出 UPLOAD_ROOT
    3. 分块写入 + 大小上限，超限立即中断并删除残文件
    4. 上传后立刻探测能否解码，坏文件当场拒绝（失败要早）
    """
    try:
        uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法 thread_id")

    target_dir = (UPLOAD_ROOT / f"session_{thread_id}").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        name = Path((file.filename or "").replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise HTTPException(status_code=400, detail=f"非法文件名: {file.filename}")
        if Path(name).suffix.lower() not in ALLOWED_SUFFIX:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {name}")

        dest = (target_dir / name).resolve()
        if not dest.is_relative_to(UPLOAD_ROOT):
            raise HTTPException(status_code=400, detail="路径越界")

        written = 0
        with dest.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_MB * 1024 * 1024:
                    buffer.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413,
                                        detail=f"{name} 超过 {MAX_UPLOAD_MB}MB 限制")
                buffer.write(chunk)

        meta = _probe_video(dest)
        if not meta["decodable"]:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400,
                                detail=f"{name} 无法解码，请确认是有效的视频文件")

        saved.append({
            "name": name,
            "rel_path": str(dest.relative_to(PROJECT_ROOT)),
            "size_mb": round(written / 1024 / 1024, 2),
            **meta,
        })

    return {"files": saved}


def _probe_video(path: Path) -> dict:
    """读一帧就知道能不能用——比等 Agent 跑 40 秒后失败要好。"""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok, _frame = cap.read()
        meta = {
            "decodable": bool(ok),
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return meta
    except Exception:
        return {"decodable": False, "fps": 0, "frames": 0, "width": 0, "height": 0}


@app.get("/api/video")
async def get_video(path: str):
    """预览用：只允许读取 UPLOAD_ROOT 内的文件。"""
    target = (PROJECT_ROOT / path).resolve()
    if not target.is_relative_to(UPLOAD_ROOT):
        raise HTTPException(status_code=403, detail="拒绝访问")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, media_type="video/mp4")


@app.get("/api/health")
async def health():
    return {"ok": True, "connections": {k: len(v) for k, v in CONNECTIONS.items()}}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>网球教练 Agent</title>
<style>
:root{
  --bg:#fff;--fg:#1f2329;--muted:#6b7280;--line:#e3e5e8;--card:#f7f8f9;
  --brand:#0a7d4b;--brand-soft:#eaf7f0;--userbg:#eef4ff;--err:#c62828;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#15171a;--fg:#e8eaed;--muted:#9aa3ad;--line:#2f343b;--card:#1c1f23;
        --brand:#3ddc84;--brand-soft:#16301f;--userbg:#1b2433;--err:#ef6b6b;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:20px 16px 30px}
h1{font-size:20px;margin:0 0 4px}
.hint{color:var(--muted);font-size:13px;margin:0 0 16px}
#drop{border:1.5px dashed var(--line);border-radius:12px;padding:18px;text-align:center;
      color:var(--muted);cursor:pointer;font-size:14px;transition:.15s}
#drop.over{border-color:var(--brand);background:var(--brand-soft);color:var(--brand)}
#bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:8px;display:none}
#bar>i{display:block;height:100%;width:0;background:var(--brand);transition:width .2s}
#meta{font-size:13px;color:var(--muted);margin:8px 0 0}
video{max-width:100%;border-radius:10px;margin-top:10px;display:none;background:#000}
#log{border:1px solid var(--line);border-radius:12px;padding:14px;height:46vh;
     min-height:260px;overflow:auto;margin-top:16px;background:var(--card)}
.msg{margin:0 0 14px}
.who{font-size:12px;color:var(--muted);margin-bottom:4px}
.bubble{padding:10px 12px;border-radius:10px;font-size:14px;line-height:1.7;
        word-break:break-word;background:var(--bg);border:1px solid var(--line)}
.u .bubble{background:var(--userbg);white-space:pre-wrap;border-color:transparent}
.trace{font-size:12px;color:var(--muted);margin-top:6px}
.err .bubble{color:var(--err);border-color:var(--err)}
.bubble table{border-collapse:collapse;margin:8px 0;font-size:13px;width:100%}
.bubble th,.bubble td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
.bubble th{background:var(--card);font-weight:600}
.bubble code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
.bubble pre{background:var(--card);padding:10px;border-radius:8px;overflow:auto;font-size:13px}
.bubble h4{margin:10px 0 6px;font-size:15px}
.bubble ul,.bubble ol{margin:6px 0;padding-left:20px}
.bubble p{margin:6px 0}
#chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.chip{border:1px solid var(--line);background:var(--bg);color:var(--fg);
      border-radius:999px;padding:6px 12px;font-size:13px;cursor:pointer}
.chip:hover{border-color:var(--brand);color:var(--brand)}
#composer{display:flex;gap:8px;margin-top:10px;align-items:flex-end}
#q{flex:1;min-height:44px;max-height:150px;resize:vertical;padding:10px 12px;
   border:1px solid var(--line);border-radius:10px;font:inherit;font-size:14px;
   background:var(--bg);color:var(--fg)}
#q:focus{outline:none;border-color:var(--brand)}
#send{padding:11px 18px;border:0;border-radius:10px;background:var(--brand);
      color:#fff;font-size:14px;cursor:pointer;font-weight:600}
#send:disabled{opacity:.45;cursor:not-allowed}
.foot{color:var(--muted);font-size:12px;margin-top:10px;display:flex;justify-content:space-between}
.foot a{color:var(--muted)}
</style>

<div class="wrap">
  <h1>网球教练 Agent</h1>
  <p class="hint">先上传一段击球视频再提问；也可以直接问技术问题、看进步、要训练计划。</p>

  <div id="drop">拖拽击球视频到这里，或点击选择（mp4 / mov / avi / mkv，≤200MB）</div>
  <input id="file" type="file" accept="video/*" style="display:none">
  <div id="bar"><i></i></div>
  <div id="meta">还没有上传视频</div>
  <video id="preview" controls></video>

  <div id="log"></div>

  <div id="chips">
    <button class="chip" data-q="这段正手帮我看看">分析动作</button>
    <button class="chip" data-q="我最近一个月有没有进步？">看进步</button>
    <button class="chip" data-q="给我一个四周训练计划，先练什么？">训练计划</button>
    <button class="chip" data-q="正手击球点应该在身体的什么位置？">技术问答</button>
    <button class="chip" data-q="预算1000元，初学者，想要控制好的球拍">选球拍</button>
  </div>

  <form id="composer">
    <textarea id="q" placeholder="输入问题…（Enter 发送，Shift + Enter 换行）"></textarea>
    <button id="send" type="submit">发送</button>
  </form>

  <div class="foot">
    <span>会话 <code id="sidLabel">-</code> · 用户 <code id="uidLabel">-</code></span>
    <a href="#" id="reset">开始新会话</a>
  </div>
</div>

<script>
const NL = String.fromCharCode(10);
const $ = function(id){ return document.getElementById(id); };
const logEl = $('log'), qEl = $('q'), sendBtn = $('send'), metaEl = $('meta');
const dropEl = $('drop'), fileInput = $('file'), bar = $('bar'), barIn = bar.querySelector('i');
const preview = $('preview'), sidLabel = $('sidLabel');

let sid = crypto.randomUUID();
// 用户标识：跨会话保持不变，用来把不同人的训练记录分开。
// 没有它，纵向对比会把两个人的数据当成同一个人的两次训练。
let uid = localStorage.getItem('tennis_uid');
if (!uid){ uid = crypto.randomUUID().slice(0, 8); localStorage.setItem('tennis_uid', uid); }
let ws = null, busy = false, current = null;
sidLabel.textContent = sid.slice(0, 8);
$('uidLabel').textContent = uid;

/* ---------------- 上传 ---------------- */
dropEl.onclick = function(){ fileInput.click(); };
dropEl.ondragover = function(e){ e.preventDefault(); dropEl.classList.add('over'); };
dropEl.ondragleave = function(){ dropEl.classList.remove('over'); };
dropEl.ondrop = function(e){
  e.preventDefault(); dropEl.classList.remove('over');
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
};
fileInput.onchange = function(){ if (fileInput.files[0]) upload(fileInput.files[0]); };

function upload(f){
  if (!f.type || f.type.indexOf('video/') !== 0){ metaEl.textContent = '请选择视频文件'; return; }
  const fd = new FormData();
  fd.append('files', f);
  fd.append('thread_id', sid);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  bar.style.display = 'block'; metaEl.textContent = '上传中…';
  xhr.upload.onprogress = function(e){
    if (e.lengthComputable) barIn.style.width = (e.loaded / e.total * 100).toFixed(0) + '%';
  };
  xhr.onload = function(){
    bar.style.display = 'none'; barIn.style.width = '0';
    if (xhr.status !== 200){
      let detail = xhr.status;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (e) {}
      metaEl.textContent = '上传失败：' + detail;
      return;
    }
    const f0 = JSON.parse(xhr.responseText).files[0];
    metaEl.textContent = f0.name + ' · ' + f0.size_mb + 'MB · ' + f0.width + 'x' +
                         f0.height + ' · ' + f0.fps + 'fps · ' + f0.frames + '帧';
    preview.src = '/api/video?path=' + encodeURIComponent(f0.rel_path);
    preview.style.display = 'block';
    if (!qEl.value.trim()) qEl.value = f0.rel_path + ' 这段正手帮我看看';
    qEl.focus();
  };
  xhr.onerror = function(){ metaEl.textContent = '上传失败：网络错误'; bar.style.display = 'none'; };
  xhr.send(fd);
}

/* ---------------- 对话 ---------------- */
function addMsg(role){
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = role === 'u' ? '你' : '教练';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const trace = document.createElement('div');
  trace.className = 'trace';
  wrap.appendChild(who); wrap.appendChild(bubble); wrap.appendChild(trace);
  logEl.appendChild(wrap);
  logEl.scrollTop = logEl.scrollHeight;
  return { wrap: wrap, bubble: bubble, trace: trace };
}

function addUser(text){
  const m = addMsg('u');
  m.bubble.textContent = text;
  return m;
}

function addAssistant(){
  const m = addMsg('a');
  m.bubble.textContent = '思考中…';
  return m;
}

function addTrace(text){
  if (!current) return;
  const line = document.createElement('div');
  line.textContent = text;
  current.trace.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function finish(){
  busy = false; sendBtn.disabled = false; current = null;
  logEl.scrollTop = logEl.scrollHeight;
}

function ensureWs(){
  if (ws && ws.readyState <= 1) return Promise.resolve();
  return new Promise(function(res, rej){
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    ws = new WebSocket(proto + location.host + '/ws/' + sid);
    ws.onopen = function(){ res(); };
    ws.onerror = function(){ rej(new Error('WebSocket 连接失败')); };
    ws.onmessage = onEvent;
  });
}

function onEvent(e){
  let d;
  try { d = JSON.parse(e.data); } catch (err) { return; }
  if (d.type === 'pong' || !current) return;
  if (d.type === 'dispatch'){ addTrace('→ 调度 ' + d.subagent); }
  else if (d.type === 'subagent_done'){ addTrace('← 子智能体返回 ' + d.chars + ' 字符'); }
  else if (d.type === 'final'){
    current.bubble.innerHTML = render(d.content || '');
    if (!current.trace.childElementCount) current.trace.style.display = 'none';
    finish();
  } else if (d.type === 'error'){
    current.wrap.classList.add('err');
    current.bubble.textContent = d.message;
    finish();
  }
}

async function send(text){
  const q = (text !== undefined ? text : qEl.value).trim();
  if (!q || busy) return;
  busy = true; sendBtn.disabled = true;
  addUser(q);
  current = addAssistant();
  qEl.value = '';
  try {
    await ensureWs();
    await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, thread_id: sid, user_id: uid })
    });
  } catch (err) {
    current.wrap.classList.add('err');
    current.bubble.textContent = '请求失败：' + err.message;
    finish();
  }
}

$('composer').onsubmit = function(e){ e.preventDefault(); send(); };
qEl.onkeydown = function(e){
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); }
};
document.querySelectorAll('.chip').forEach(function(b){
  b.onclick = function(){ send(b.getAttribute('data-q')); };
});
$('reset').onclick = function(e){
  e.preventDefault();
  if (ws){ ws.onclose = null; ws.close(); ws = null; }
  sid = crypto.randomUUID();
  sidLabel.textContent = sid.slice(0, 8);
  logEl.innerHTML = '';
  metaEl.textContent = '还没有上传视频';
  preview.style.display = 'none'; preview.removeAttribute('src');
  qEl.value = '';
};

/* ---------------- 极简 Markdown 渲染 ---------------- */
function esc(s){
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function boldify(s){
  const parts = s.split('**');
  let out = '';
  for (let k = 0; k < parts.length; k++){
    if (k === 0){ out += parts[k]; continue; }
    out += (k % 2 === 1 ? '<strong>' : '</strong>') + parts[k];
  }
  if (parts.length % 2 === 0) out += '</strong>';
  return out;
}

function inline(s){
  const parts = esc(s).split('`');
  let out = '';
  for (let k = 0; k < parts.length; k++){
    out += (k % 2 === 1) ? '<code>' + parts[k] + '</code>' : boldify(parts[k]);
  }
  return out;
}

function isSepRow(s){
  s = s.trim();
  if (s.indexOf('|') < 0) return false;
  for (let k = 0; k < s.length; k++){
    if ('|-: '.indexOf(s.charAt(k)) < 0) return false;
  }
  return true;
}

function cells(s){
  s = s.trim();
  while (s.charAt(0) === '|') s = s.slice(1);
  while (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
  return s.split('|');
}

function isNumItem(t){
  const dot = t.indexOf('. ');
  if (dot < 1 || dot > 3) return false;
  const head = t.slice(0, dot);
  for (let k = 0; k < head.length; k++){
    if ('0123456789'.indexOf(head.charAt(k)) < 0) return false;
  }
  return true;
}

function render(md){
  const lines = String(md).split(NL);
  const out = [];
  let i = 0;
  while (i < lines.length){
    const t = lines[i].trim();
    if (!t){ i++; continue; }

    if (t.indexOf('```') === 0){
      i++;
      const buf = [];
      while (i < lines.length && lines[i].trim().indexOf('```') !== 0){ buf.push(lines[i]); i++; }
      i++;
      out.push('<pre>' + esc(buf.join(NL)) + '</pre>');
      continue;
    }

    if (t.charAt(0) === '|' && i + 1 < lines.length && isSepRow(lines[i + 1])){
      const head = cells(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().charAt(0) === '|'){ rows.push(cells(lines[i])); i++; }
      let html = '<table><thead><tr>';
      head.forEach(function(c){ html += '<th>' + inline(c.trim()) + '</th>'; });
      html += '</tr></thead><tbody>';
      rows.forEach(function(r){
        html += '<tr>';
        r.forEach(function(c){ html += '<td>' + inline(c.trim()) + '</td>'; });
        html += '</tr>';
      });
      out.push(html + '</tbody></table>');
      continue;
    }

    if (t.indexOf('- ') === 0 || t.indexOf('* ') === 0){
      const items = [];
      while (i < lines.length){
        const s2 = lines[i].trim();
        if (s2.indexOf('- ') === 0 || s2.indexOf('* ') === 0){
          items.push('<li>' + inline(s2.slice(2)) + '</li>'); i++;
        } else break;
      }
      out.push('<ul>' + items.join('') + '</ul>');
      continue;
    }

    if (isNumItem(t)){
      const items = [];
      while (i < lines.length && isNumItem(lines[i].trim())){
        const s2 = lines[i].trim();
        items.push('<li>' + inline(s2.slice(s2.indexOf('. ') + 2)) + '</li>'); i++;
      }
      out.push('<ol>' + items.join('') + '</ol>');
      continue;
    }

    if (t.indexOf('###') === 0){
      out.push('<h4>' + inline(t.replace('###', '').trim()) + '</h4>'); i++; continue;
    }
    if (t === '---'){ out.push('<hr>'); i++; continue; }
    if (t.charAt(0) === '>'){ out.push('<p>' + inline(t.slice(1).trim()) + '</p>'); i++; continue; }

    out.push('<p>' + inline(t) + '</p>');
    i++;
  }
  return out.join('');
}
</script>
"""


if __name__ == "__main__":
    import os

    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8010")), reload=False)
