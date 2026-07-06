"""零依赖 Web 演示服务：输入产品 → 可视化 双路召回 + LLM 精排 → 命中节点。

运行：  python -m product_mapper.server
然后浏览器打开：  http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from .agent import ProductMapper

MAPPER = None  # 启动时构建一次（建索引约 8s）

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>产品 - 标准体系映射智能体</title>
<style>
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif;
   background:#0f172a;color:#e2e8f0}
 .wrap{max-width:960px;margin:0 auto;padding:32px 20px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#94a3b8;font-size:13px;margin-bottom:24px}
 .bar{display:flex;gap:10px} input{flex:1;padding:12px 14px;border-radius:10px;border:1px solid #334155;
   background:#1e293b;color:#e2e8f0;font-size:15px} input:focus{outline:none;border-color:#6366f1}
 button{padding:12px 22px;border:0;border-radius:10px;background:#6366f1;color:#fff;font-size:15px;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
 .chips{margin:12px 0 0;display:flex;flex-wrap:wrap;gap:8px}
 .chip{padding:5px 12px;background:#1e293b;border:1px solid #334155;border-radius:999px;font-size:13px;cursor:pointer;color:#cbd5e1}
 .chip:hover{border-color:#6366f1}
 .card{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:20px;margin-top:22px}
 .hit{border-color:#22c55e} .miss{border-color:#f59e0b}
 .path{font-size:18px;font-weight:600;line-height:1.5} .path .arrow{color:#64748b;margin:0 6px}
 .path .leaf{color:#4ade80}
 .meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:13px;color:#94a3b8}
 .badge{padding:2px 9px;border-radius:6px;font-size:12px}
 .b-llm{background:#3730a3;color:#c7d2fe} .b-fusion{background:#78350f;color:#fde68a}
 .conf{height:8px;background:#334155;border-radius:5px;overflow:hidden;margin-top:6px;width:200px}
 .conf>i{display:block;height:100%;background:linear-gradient(90deg,#6366f1,#22c55e)}
 .reason{margin-top:12px;color:#cbd5e1;font-size:14px;line-height:1.6}
 table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #334155}
 th{color:#94a3b8;font-weight:500} tr.sel{background:#14321f}
 td.num{font-variant-numeric:tabular-nums;color:#cbd5e1} .g{color:#64748b}
 .load{display:none;color:#94a3b8;margin-top:20px} h3{font-size:14px;color:#cbd5e1;margin:0 0 4px}
 .note{font-size:12px;color:#64748b;margin-top:6px}
</style></head><body><div class="wrap">
 <h1>产品 - 标准产品体系映射智能体</h1>
 <div class="sub">双路召回（pg_trgm 字面 ∪ 向量语义）→ 多策略融合 + DeepSeek 精排 → 唯一标准节点</div>
 <div class="bar">
   <input id="q" placeholder="输入一个产品名，如：苞米、独头蒜、Vigna radiata、红富士苹果" autofocus>
   <button id="go" onclick="run()">映射</button>
 </div>
 <div class="chips" id="chips"></div>
 <div class="load" id="load">⏳ 正在召回并请求 DeepSeek 精排…（约 2~3 秒）</div>
 <div id="out"></div>
</div>
<script>
const SAMPLES=["苞米","独头蒜","Vigna radiata","红富士苹果","五常大米","笔记本电脑","金针菇"];
const chips=document.getElementById('chips');
SAMPLES.forEach(s=>{const c=document.createElement('span');c.className='chip';c.textContent=s;
  c.onclick=()=>{document.getElementById('q').value=s;run()};chips.appendChild(c)});
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});

async function run(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out'), load=document.getElementById('load'), go=document.getElementById('go');
  out.innerHTML=''; load.style.display='block'; go.disabled=true;
  try{
    const r=await fetch('/api/map',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product:q})});
    const d=await r.json(); render(d);
  }catch(e){ out.innerHTML='<div class="card miss">请求失败：'+e+'</div>'; }
  load.style.display='none'; go.disabled=false;
}
function pathHtml(p){ if(!p)return''; const a=p.split(' > ');
  return a.map((x,i)=>i===a.length-1?'<span class="leaf">'+x+'</span>':x+'<span class="arrow">›</span>').join(''); }
function render(d){
  const R=d.result, C=d.candidates, hit=R.node_id!==null;
  const badge=R.source==='llm'?'<span class="badge b-llm">DeepSeek 精排</span>':'<span class="badge b-fusion">融合分兜底</span>';
  let head = hit
    ? '<h3>命中标准节点</h3><div class="path">'+pathHtml(R.path)+'</div>'
    : '<h3>未找到合适节点</h3><div class="path g">建议触发【体系扩展】：为该产品新增节点</div>';
  let conf = hit? '<div>置信度 '+R.confidence+'<div class="conf"><i style="width:'+Math.round(R.confidence*100)+'%"></i></div></div>':'';
  let card='<div class="card '+(hit?'hit':'miss')+'">'+head+
    '<div class="meta">'+badge+'<span>node_id: '+(R.node_id??'—')+'</span>'+
    '<span>候选 '+R.n_candidates+' 个</span><span>耗时 '+R.latency_ms+' ms</span></div>'+
    conf + (R.reason?'<div class="reason">💡 '+R.reason+'</div>':'')+'</div>';
  let rows=C.map(c=>'<tr class="'+(c.chosen?'sel':'')+'"><td>'+(c.chosen?'✅':'')+' '+c.name+
    '<div class="note">'+c.path+(c.synonyms.length?'　·　同义词: '+c.synonyms.join('、'):'')+'</div></td>'+
    '<td class="num">'+c.trgm+'</td><td class="num">'+c.vec+'</td><td class="num">'+c.fused+'</td></tr>').join('');
  let table='<div class="card"><h3>召回候选与融合打分（Top '+C.length+'）</h3>'+
    '<table><thead><tr><th>候选标准节点</th><th>trgm 字面</th><th>向量 语义</th><th>融合分</th></tr></thead>'+
    '<tbody>'+rows+'</tbody></table></div>';
  document.getElementById('out').innerHTML=card+table;
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/map":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            product = (payload.get("product") or "").strip()
            if not product:
                return self._send(400, json.dumps({"error": "empty product"}))
            result, candidates = MAPPER.explain(product)
            body = json.dumps({"result": result, "candidates": candidates},
                              ensure_ascii=False)
            self._send(200, body)
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))

    def log_message(self, *args):
        pass  # 静默


def main(port: int = 8000):
    global MAPPER
    print("正在构建索引（约 8 秒）…")
    MAPPER = ProductMapper()
    print(f"索引就绪：{len(MAPPER.nodes)} 节点，"
          f"LLM={'已启用' if config.has_llm() else '未启用(降级)'}")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n✅ 打开浏览器访问：http://localhost:{port}\n   （Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
