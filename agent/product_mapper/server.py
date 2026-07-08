"""零依赖 Web 演示服务：输入产品 → 可视化映射结果。
支持三种方案切换：Route A (RAG+LLM) / Route B (PageIndex) / Hybrid (A→B fallback)
支持 Embedder 切换：Hash / ST

运行：  python -m product_mapper.server
然后浏览器打开：  http://localhost:8000
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from .agent import ProductMapper
from .embedder import st_available
from .extension import append_extension_record, route_a_reliable, route_b_reliable, suggest_extension
from .pageindex_mapper import PageIndexMapper

MAPPER = None         # Route A
PI_MAPPER = None      # Route B
CURRENT_EMBEDDER = None
CURRENT_METHOD = "raga"  # "raga" | "pageindex" | "hybrid"

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>产品 - 标准体系映射智能体</title>
<style>
 *{box-sizing:border-box} body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif;
   background:#0f172a;color:#e2e8f0}
 .wrap{max-width:960px;margin:0 auto;padding:32px 20px}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#94a3b8;font-size:13px;margin-bottom:16px}
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
 .badge{padding:2px 9px;border-radius:6px;font-size:12px;white-space:nowrap}
 .b-llm{background:#3730a3;color:#c7d2fe} .b-fusion{background:#78350f;color:#fde68a}
 .b-exact{background:#14532d;color:#86efac} .b-pageindex{background:#1e3a5f;color:#93c5fd}
 .b-hybrid{background:#4a1d6b;color:#d8b4fe}
 .conf{height:8px;background:#334155;border-radius:5px;overflow:hidden;margin-top:6px;width:200px}
 .conf>i{display:block;height:100%;background:linear-gradient(90deg,#6366f1,#22c55e)}
 .reason{margin-top:12px;color:#cbd5e1;font-size:14px;line-height:1.6}
 table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #334155}
 th{color:#94a3b8;font-weight:500} tr.sel{background:#14321f}
 td.num{font-variant-numeric:tabular-nums;color:#cbd5e1} .g{color:#64748b}
 .load{display:none;color:#94a3b8;margin-top:20px} h3{font-size:14px;color:#cbd5e1;margin:0 0 4px}
 .note{font-size:12px;color:#64748b;margin-top:6px}
 .toggle-row{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}
 .toggle-label{font-size:13px;color:#94a3b8;min-width:60px}
 .toggle-group{display:flex;border-radius:8px;overflow:hidden;border:1px solid #334155}
 .toggle-btn{padding:6px 16px;font-size:13px;cursor:pointer;border:0;background:#1e293b;color:#94a3b8;transition:.2s}
 .toggle-btn.active{background:#6366f1;color:#fff}
 .toggle-btn:disabled{opacity:.5;cursor:wait}
 .toggle-note{font-size:12px;color:#f59e0b}
 .desc{font-size:12px;color:#64748b;margin-top:2px;line-height:1.5}
 .trace-card{background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:16px;margin-top:14px}
 .trace-step{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #1e293b;align-items:flex-start}
 .trace-step:last-child{border-bottom:0}
 .trace-num{background:#1e3a5f;color:#93c5fd;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
 .trace-info{flex:1} .trace-name{font-weight:600;color:#e2e8f0} .trace-reason{font-size:12px;color:#94a3b8;margin-top:2px}
 .trace-conf{font-size:11px;color:#64748b}
 .flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}
 .flow-step{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;min-height:76px}
 .flow-step.done{border-color:#22c55e}.flow-step.warn{border-color:#f59e0b}.flow-step.stop{border-color:#ef4444}
 .flow-title{font-size:13px;color:#e2e8f0;font-weight:600}.flow-desc{font-size:12px;color:#94a3b8;margin-top:6px;line-height:1.5}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
 .extension{border-color:#f59e0b;background:#221a10}
 .kv{display:grid;grid-template-columns:120px 1fr;gap:8px 14px;margin-top:14px;font-size:13px}
 .kv .k{color:#94a3b8}.kv .v{color:#e2e8f0}
 @media(max-width:760px){.flow{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.bar{flex-direction:column}.kv{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
 <h1>产品 - 标准产品体系映射智能体</h1>
 <div class="sub" id="subtitle">双路召回（trigram + 向量）→ 融合 + LLM 精排 → 唯一标准节点</div>

 <!-- 方案切换 -->
 <div class="toggle-row">
   <span class="toggle-label">匹配方案：</span>
   <div class="toggle-group" id="methodToggle">
     <button class="toggle-btn active" data-method="raga" onclick="switchMethod('raga')">Route A · RAG</button>
     <button class="toggle-btn" data-method="pageindex" onclick="switchMethod('pageindex')">Route B · PageIndex</button>
     <button class="toggle-btn" data-method="hybrid" onclick="switchMethod('hybrid')">Hybrid · A+B</button>
   </div>
   <span class="toggle-note" id="methodNote"></span>
 </div>

 <!-- Embedder 切换 (仅 Route A 相关) -->
 <div class="toggle-row" id="embedderRow">
   <span class="toggle-label">向量模型：</span>
   <div class="toggle-group" id="toggle">
     <button class="toggle-btn active" data-emb="hash" onclick="switchEmb('hash')">哈希 (Hash)</button>
     <button class="toggle-btn" data-emb="st" onclick="switchEmb('st')">语义 (ST)</button>
   </div>
   <span class="toggle-note" id="embNote"></span>
 </div>

 <div class="bar">
   <input id="q" placeholder="输入一个产品名，如：苞米、华为Matebook X Pro、隆基绿能光伏板" autofocus>
   <button id="go" onclick="run()">映射</button>
 </div>
 <div class="chips" id="chips"></div>
 <div class="load" id="load">[加载中] 正在匹配…</div>
 <div id="out"></div>
</div>
<script>
const SAMPLES=["苞米","独头蒜","Vigna radiata","红富士苹果","笔记本电脑","工业机器人","华为Matebook X Pro","大疆无人机Mavic 3","特斯拉Model 3电池包","XYZ999999","火星地产会员卡"];
const chips=document.getElementById('chips');
SAMPLES.forEach(s=>{const c=document.createElement('span');c.className='chip';c.textContent=s;
  c.onclick=()=>{document.getElementById('q').value=s;run()};chips.appendChild(c)});
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')run()});

let currentEmb='hash';
let currentMethod='raga';

const methodDescs = {
  'raga': 'Route A：双路召回（trigram 字面 + 向量语义）→ 多策略融合 + DeepSeek 精排 → 唯一标准节点',
  'pageindex': 'Route B：LLM 在标准体系树上逐层推理搜索（PageIndex 式），无向量依赖，完整可追溯路径',
  'hybrid': 'Hybrid：Route A 召回先试 → 低置信度或无结果时自动 fallback 到 Route B 树搜索',
};

async function getState(){try{const r=await fetch('/api/state');const d=await r.json();currentEmb=d.embedder;currentMethod=d.method;updateMethodBtns();updateEmbBtns();updateEmbedderRow()}catch(e){}}

function updateMethodBtns(){
  document.querySelectorAll('#methodToggle .toggle-btn').forEach(b=>{
    b.classList.toggle('active',b.dataset.method===currentMethod);
  });
  document.getElementById('subtitle').textContent = methodDescs[currentMethod] || '';
}

function updateEmbBtns(){
  document.querySelectorAll('#toggle .toggle-btn').forEach(b=>{
    b.classList.toggle('active',b.dataset.emb===currentEmb);
  });
}

function updateEmbedderRow(){
  document.getElementById('embedderRow').style.display = currentMethod==='pageindex' ? 'none' : '';
}

async function switchMethod(method){
  if(method===currentMethod)return;
  const btns=document.querySelectorAll('#methodToggle .toggle-btn');
  const note=document.getElementById('methodNote');
  btns.forEach(b=>b.disabled=true);
  note.textContent='切换中…';
  try{
    const r=await fetch('/api/method',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method})});
    const d=await r.json();
    if(d.error){note.textContent='错误: '+d.error;return}
    currentMethod=d.method;
    updateMethodBtns();
    updateEmbedderRow();
    note.textContent='';
    const curQ=document.getElementById('q').value.trim();
    if(curQ) run();
  }catch(e){note.textContent='切换失败: '+e}
  btns.forEach(b=>b.disabled=false);
}

async function switchEmb(type){
  if(type===currentEmb)return;
  const btns=document.querySelectorAll('#toggle .toggle-btn');
  const note=document.getElementById('embNote');
  btns.forEach(b=>b.disabled=true);
  note.textContent='切换中，加载本地模型（约 2 秒）…';
  try{
    const r=await fetch('/api/embedder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({embedder:type})});
    const d=await r.json();
    if(d.error){note.textContent='错误: '+d.error;return}
    currentEmb=d.embedder;
    note.textContent=d.note||'';
    if(d.note)setTimeout(()=>{note.textContent=''},3000);
    updateEmbBtns();
    const curQ=document.getElementById('q').value.trim();
    if(curQ) run();
  }catch(e){note.textContent='切换失败: '+e}
  btns.forEach(b=>b.disabled=false);
}

async function run(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out'), load=document.getElementById('load'), go=document.getElementById('go');
  out.innerHTML=''; load.style.display='block'; go.disabled=true;

  let apiUrl = '/api/map';
  if(currentMethod==='pageindex') apiUrl='/api/pageindex';
  else if(currentMethod==='hybrid') apiUrl='/api/hybrid';

  try{
    const r=await fetch(apiUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({product:q})});
    const d=await r.json();
    if(currentMethod==='hybrid') renderHybrid(d);
    else if(currentMethod==='pageindex') renderPageIndex(d);
    else renderRAG(d);
  }catch(e){ out.innerHTML='<div class="card miss">请求失败：'+e+'</div>'; }
  load.style.display='none'; go.disabled=false;
}

function pathHtml(p){ if(!p)return''; const a=p.split(' > ');
  return a.map((x,i)=>i===a.length-1?'<span class="leaf">'+x+'</span>':x+'<span class="arrow">/</span>').join(''); }

function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}

function badgeForSource(source){
  if(source==='llm') return '<span class="badge b-llm">LLM 精排</span>';
  if(source==='exact_match'||source==='pageindex_exact') return '<span class="badge b-exact">精确匹配</span>';
  if(source==='pageindex') return '<span class="badge b-pageindex">PageIndex 树搜索</span>';
  if(source==='pageindex_trigram'||source==='fusion') return '<span class="badge b-fusion">本地候选</span>';
  if(source==='pageindex_trigram_weak') return '<span class="badge b-fusion">弱候选</span>';
  if(source==='hybrid_raga') return '<span class="badge b-hybrid">Hybrid · Route A</span>';
  if(source==='hybrid_pageindex') return '<span class="badge b-pageindex">Hybrid · Route B</span>';
  return '<span class="badge b-fusion">'+esc(source||'empty')+'</span>';
}

function resultCard(title, R, reliable){
  const hit=R&&R.node_id!==null&&R.node_id!==undefined;
  const cls=hit?(reliable?'hit':'miss'):'miss';
  const status=hit?(reliable?'可靠命中':'弱命中 / 待复核'):'未命中';
  const conf=hit? Number(R.confidence||0):0;
  const head=hit
    ? '<h3>'+esc(title)+' · '+status+'</h3><div class="path">'+pathHtml(R.path)+'</div>'
    : '<h3>'+esc(title)+' · 未命中</h3><div class="path g">没有找到可直接接受的标准节点</div>';
  return '<div class="card '+cls+'">'+head+
    '<div class="meta">'+badgeForSource(R&&R.source)+'<span>node_id: '+(hit?R.node_id:'-')+'</span>'+
    '<span>置信度 '+conf.toFixed(3)+'</span><span>耗时 '+(R&&R.latency_ms!=null?R.latency_ms:'-')+' ms</span></div>'+
    (hit?'<div class="conf"><i style="width:'+Math.round(conf*100)+'%"></i></div>':'')+
    (R&&R.reason?'<div class="reason">理由：'+esc(R.reason)+'</div>':'')+'</div>';
}

function renderFlow(steps){
  return '<div class="flow">'+(steps||[]).map(s=>{
    const cls=s.status==='stop'?'stop':(s.status==='warn'?'warn':'done');
    return '<div class="flow-step '+cls+'"><div class="flow-title">'+esc(s.title)+'</div><div class="flow-desc">'+esc(s.desc)+'</div></div>';
  }).join('')+'</div>';
}

function renderExtension(ext){
  if(!ext)return '';
  const syn=(ext.synonyms||[]).join('、')||'-';
  return '<div class="card extension"><h3>体系扩展建议</h3>'+
    '<div class="path">'+esc(ext.action||'-')+'</div>'+
    '<div class="kv">'+
      '<div class="k">建议新增节点</div><div class="v">'+esc(ext.new_node_name||'-')+'</div>'+
      '<div class="k">建议父节点</div><div class="v">'+esc(ext.parent_path||'-')+'</div>'+
      '<div class="k">父节点ID</div><div class="v">'+esc(ext.parent_node_id||'-')+'</div>'+
      '<div class="k">建议同义词</div><div class="v">'+esc(syn)+'</div>'+
      '<div class="k">最接近候选</div><div class="v">'+esc(ext.nearest_path||'-')+'</div>'+
      '<div class="k">优先级</div><div class="v">'+esc(ext.priority||'-')+'</div>'+
      '<div class="k">复核状态</div><div class="v">'+esc(ext.review_status||'待复核')+'</div>'+
      '<div class="k">保存结果</div><div class="v">'+(ext.saved?'已写入 cache/extension_suggestions':'仅页面展示')+'</div>'+
    '</div>'+
    '<div class="reason">建议理由：'+esc(ext.reason||'-')+'</div></div>';
}

function renderHybrid(d){
  const a=d.route_a||{}, b=d.route_b||{}, final=d.final||{};
  let finalHtml='';
  if(final.node_id!==null&&final.node_id!==undefined){
    finalHtml='<div class="card hit"><h3>最终采用路线</h3><div class="path">'+esc(final.route||'-')+'</div>'+
      '<div class="meta"><span>node_id: '+final.node_id+'</span><span>'+esc(final.path||'')+'</span></div></div>';
  }else{
    finalHtml='<div class="card miss"><h3>最终结果</h3><div class="path g">Route A 与 Route B 均未可靠命中，进入体系扩展流程</div></div>';
  }
  document.getElementById('out').innerHTML=
    renderFlow(d.flow_steps)+
    '<div class="grid">'+
    resultCard('Route A · RAG', a.result||{}, a.reliable)+
    resultCard('Route B · PageIndex', b.result||{}, b.reliable)+
    '</div>'+
    finalHtml+
    renderExtension(d.extension);
}

function renderRAG(d){
  const R=d.result, C=d.candidates||[], hit=R.node_id!==null;
  let badge='';
  if(R.source==='llm') badge='<span class="badge b-llm">LLM 精排</span>';
  else if(R.source==='exact_match') badge='<span class="badge b-exact">[精确匹配]</span>';
  else if(R.source==='fusion') badge='<span class="badge b-fusion">融合分兜底</span>';
  else if(R.source==='hybrid_raga') badge='<span class="badge b-hybrid">Hybrid · RAG 命中</span>';
  else if(R.source==='hybrid_pageindex') badge='<span class="badge b-pageindex">Hybrid · PageIndex 命中</span>';
  else badge='<span class="badge b-fusion">'+R.source+'</span>';

  let head = hit
    ? '<h3>命中标准节点</h3><div class="path">'+pathHtml(R.path)+'</div>'
    : '<h3>未找到合适节点</h3><div class="path g">建议触发【体系扩展】：为该产品新增节点</div>';
  let conf = hit? '<div>置信度 '+R.confidence.toFixed(3)+'<div class="conf"><i style="width:'+Math.round(R.confidence*100)+'%"></i></div></div>':'';
  let card='<div class="card '+(hit?'hit':'miss')+'">'+head+
    '<div class="meta">'+badge+'<span>node_id: '+(R.node_id!=null?R.node_id:'-')+'</span>'+
    '<span>耗时 '+R.latency_ms+' ms</span></div>'+
    conf + (R.reason?'<div class="reason">理由：'+R.reason+'</div>':'')+'</div>';

  let table='';
  if(C.length>0){
    let rows=C.map(c=>'<tr class="'+(c.chosen?'sel':'')+'"><td>'+(c.chosen?'[OK] ':'')+c.name+
      '<div class="note">'+c.path+(c.synonyms&&c.synonyms.length?'　·　同义词: '+c.synonyms.join('、'):'')+'</div></td>'+
      '<td class="num">'+(c.trgm!=null?c.trgm.toFixed(3):'-')+'</td><td class="num">'+(c.vec!=null?c.vec.toFixed(3):'-')+'</td><td class="num">'+(c.fused!=null?c.fused.toFixed(3):'-')+'</td></tr>').join('');
    table='<div class="card"><h3>召回候选与融合打分（Top '+C.length+'）</h3>'+
      '<table><thead><tr><th>候选标准节点</th><th>trgm 字面</th><th>向量 语义</th><th>融合分</th></tr></thead>'+
      '<tbody>'+rows+'</tbody></table></div>';
  }
  document.getElementById('out').innerHTML=card+table;
}

function renderPageIndex(d){
  const R=d.result, trace=d.trace||[], hit=R.node_id!==null;
  let badge='';
  if(R.source==='pageindex_exact') badge='<span class="badge b-exact">[精确匹配]</span>';
  else if(R.source==='pageindex') badge='<span class="badge b-pageindex">PageIndex 树搜索</span>';
  else if(R.source==='pageindex_trigram') badge='<span class="badge b-fusion">trigram 降级</span>';
  else badge='<span class="badge b-fusion">'+R.source+'</span>';

  let head = hit
    ? '<h3>命中标准节点</h3><div class="path">'+pathHtml(R.path)+'</div>'
    : '<h3>未找到合适节点</h3><div class="path g">建议触发【体系扩展】：为该产品新增节点</div>';
  let conf = hit? '<div>置信度 '+R.confidence.toFixed(3)+'<div class="conf"><i style="width:'+Math.round(R.confidence*100)+'%"></i></div></div>':'';
  let card='<div class="card '+(hit?'hit':'miss')+'">'+head+
    '<div class="meta">'+badge+'<span>node_id: '+(R.node_id!=null?R.node_id:'-')+'</span>'+
    '<span>搜索层数 '+R.n_layers_visited+'</span><span>耗时 '+R.latency_ms+' ms</span></div>'+
    conf + (R.reason?'<div class="reason">推理链：'+R.reason+'</div>':'')+'</div>';

  // Trace visualization
  let traceHtml='';
  if(trace.length>0){
    traceHtml='<div class="card"><h3>树搜索推理路径</h3><div class="trace-card">';
    trace.forEach((t,i)=>{
      traceHtml+='<div class="trace-step">'+
        '<div class="trace-num">'+(i+1)+'</div>'+
        '<div class="trace-info"><div class="trace-name">'+t.name+'</div>'+
        '<div class="trace-reason">'+t.reason+'</div></div>'+
        '<div class="trace-conf">conf: '+t.confidence.toFixed(2)+'</div>'+
        '</div>';
    });
    traceHtml+='</div></div>';
  }
  document.getElementById('out').innerHTML=card+traceHtml;
}
 getState();
</script>
</body></html>"""


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
        elif self.path == "/api/state":
            self._send(200, json.dumps({
                "embedder": CURRENT_EMBEDDER,
                "method": CURRENT_METHOD,
            }))
        elif self.path == "/api/embedder":
            self._send(200, json.dumps({"embedder": CURRENT_EMBEDDER}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/map":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            product = (payload.get("product") or "").strip()
            if not product:
                return self._send(400, json.dumps({"error": "empty product"}))
            result, candidates = MAPPER.explain(product)
            body = json.dumps({"result": result, "candidates": candidates},
                              ensure_ascii=False)
            self._send(200, body)

        elif self.path == "/api/pageindex":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            product = (payload.get("product") or "").strip()
            if not product:
                return self._send(400, json.dumps({"error": "empty product"}))
            result, trace = PI_MAPPER.explain(product)
            body = json.dumps({
                "result": result,
                "trace": trace,
            }, ensure_ascii=False)
            self._send(200, body)

        elif self.path == "/api/hybrid":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            product = (payload.get("product") or "").strip()
            if not product:
                return self._send(400, json.dumps({"error": "empty product"}))

            # Hybrid demo runs both routes so the whole decision process is visible.
            result_a, candidates = MAPPER.explain(product)
            result_b, trace_b = PI_MAPPER.explain(product)

            a_ok = route_a_reliable(result_a)
            b_ok = route_b_reliable(result_b)
            extension = None
            saved = False
            if a_ok:
                final = {
                    "route": "Route A",
                    "node_id": result_a.get("node_id"),
                    "name": result_a.get("name"),
                    "path": result_a.get("path"),
                    "confidence": result_a.get("confidence", 0),
                    "source": result_a.get("source", ""),
                }
            elif b_ok:
                final = {
                    "route": "Route B",
                    "node_id": result_b.get("node_id"),
                    "name": result_b.get("name"),
                    "path": result_b.get("path"),
                    "confidence": result_b.get("confidence", 0),
                    "source": result_b.get("source", ""),
                }
            else:
                extension = suggest_extension(product, MAPPER, result_a, result_b, use_llm=True)
                try:
                    append_extension_record(extension)
                    extension["saved"] = True
                    saved = True
                except Exception as e:
                    extension["saved"] = False
                    extension["save_error"] = str(e)
                final = {
                    "route": "体系扩展",
                    "node_id": None,
                    "name": None,
                    "path": None,
                    "confidence": 0.0,
                    "source": "extension",
                }

            flow_steps = [
                {"title": "输入产品名", "desc": product, "status": "done"},
                {
                    "title": "Route A 判断",
                    "desc": "可靠命中" if a_ok else ("弱命中/待复核" if result_a.get("node_id") else "未命中"),
                    "status": "done" if a_ok else "warn",
                },
                {
                    "title": "Route B 判断",
                    "desc": "可靠命中" if b_ok else ("弱命中/待复核" if result_b.get("node_id") else "未命中"),
                    "status": "done" if b_ok else "warn",
                },
                {
                    "title": "最终流向",
                    "desc": f"采用 {final['route']}" if final.get("node_id") else "进入体系扩展建议",
                    "status": "done" if final.get("node_id") else "stop",
                },
            ]
            body = json.dumps({
                "route_a": {"result": result_a, "candidates": candidates, "reliable": a_ok},
                "route_b": {"result": result_b, "trace": trace_b, "reliable": b_ok},
                "final": final,
                "extension": extension,
                "flow_steps": flow_steps,
                "extension_saved": saved,
            }, ensure_ascii=False)
            self._send(200, body)

        elif self.path == "/api/method":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            method = (payload.get("method") or "").strip()
            if method not in ("raga", "pageindex", "hybrid"):
                return self._send(400, json.dumps({"error": "method must be raga, pageindex, or hybrid"}))
            global CURRENT_METHOD
            CURRENT_METHOD = method
            self._send(200, json.dumps({"method": method}))

        elif self.path == "/api/embedder":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            emb = (payload.get("embedder") or "").strip()
            if emb not in ("hash", "st"):
                return self._send(400, json.dumps({"error": "embedder must be hash or st"}))
            if emb == "st" and not st_available():
                return self._send(400, json.dumps({"error": "sentence-transformers 未安装，无法使用 ST"}))
            try:
                dt = MAPPER.set_embedder(emb)
                global CURRENT_EMBEDDER
                CURRENT_EMBEDDER = emb
                note = ""
                if dt > 0.5:
                    note = f"切换成功，重建向量索引耗时 {dt:.1f}s"
                self._send(200, json.dumps({"embedder": emb, "note": note}))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *args):
        pass


def main(port: int = 8000):
    global MAPPER, PI_MAPPER, CURRENT_EMBEDDER, CURRENT_METHOD
    print("\n正在构建索引…")

    # Route A initialization
    print("  [Route A] 加载 RAG 索引…")
    MAPPER = ProductMapper()

    # Route B initialization
    print("  [Route B] 加载 PageIndex 树…")
    PI_MAPPER = PageIndexMapper()

    # Warmup ST model
    if st_available():
        try:
            from .embedder import STEmbedder
            st = STEmbedder()
            texts = [n.search_text() for n in MAPPER.nodes]
            st_emb = st.encode(texts)
            MAPPER.recaller._emb_cache['st'] = (st, st_emb)
            print('  ST model warmed up with pre-computed embeddings')
        except Exception as e:
            print(f'  ST warmup failed (hash mode unaffected): {e}')

    CURRENT_EMBEDDER = MAPPER.embedder_type
    CURRENT_METHOD = "raga"
    st_info = "可用" if st_available() else "未安装"

    print(f"\n索引就绪：{len(MAPPER.nodes)} 节点")
    print(f"  Route A (RAG):        Embedder={CURRENT_EMBEDDER}, LLM={'已启用' if config.has_llm() else '未启用'}")
    print(f"  Route B (PageIndex):  LLM={'已启用' if config.has_llm() else '未启用'}, ST={st_info}")
    print(f"  Hybrid:               A → B fallback (低置信度时触发)")

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n打开浏览器访问：http://localhost:{port}   （Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
