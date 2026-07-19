import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const navigation = [
  ["chat", "✦", "智能问答"], ["knowledge", "▦", "知识空间"],
  ["code", "</>", "代码工作台"], ["history", "◷", "问答记录"], ["system", "◉", "模型中心"]
];

function Sidebar({ view, setView, health }) {
  return <aside className="sidebar">
    <div className="brand"><span>知</span><div><strong>知问</strong><small>KNOWLEDGE COPILOT</small></div></div>
    <div className="workspace"><small>当前工作区</small><b>毕业设计知识库</b><span>个人空间</span></div>
    <nav>{navigation.map(([id, icon, label]) => <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}><i>{icon}</i>{label}</button>)}</nav>
    <div className="infra-status"><span className={health?.infraAvailable ? "online" : ""}/><div><b>{health?.infraAvailable ? "AI Infra 已连接" : "AI Infra 未启动"}</b><small>{health?.model || "等待 FastAPI"}</small></div></div>
  </aside>;
}

function Header({ eyebrow, title, description, action }) {
  return <header className="page-header"><div><p>{eyebrow}</p><h1>{title}</h1><span>{description}</span></div>{action}</header>;
}

function ModelSelector({ providers, value, onChange }) {
  const provider = providers.find((item) => item.id === value);
  return <div className="model-selector"><span className="model-spark">✦</span><div><small>本次对话模型</small><select value={value} onChange={(event) => onChange(event.target.value)}>{providers.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.model}</option>)}</select></div><i className={provider?.configured ? "ready" : ""}/></div>;
}

function Stat({ value = 0, label, hint }) {
  return <div className="stat"><div><b>{Number(value).toLocaleString()}</b><span>{label}</span></div>{hint && <small>{hint}</small>}</div>;
}

function Avatar({ user = false }) { return <div className={`avatar ${user ? "user-avatar" : ""}`}>{user ? "你" : "AI"}</div>; }

function Message({ message }) {
  const user = message.role === "user";
  return <article className={`message ${user ? "user" : "assistant"}`}>
    {!user && <Avatar/>}<div className={`bubble ${message.failed ? "failed" : ""}`}><div className="answer">{message.answer}</div>
      {message.citations?.length > 0 && <details><summary>引用了 {message.citations.length} 条知识依据</summary><div className="citation-list">{message.citations.map((item) => <div className="citation" key={`${item.documentId}-${item.chunk}`}><div><b>{item.id} {item.title}</b><span>相关度 {item.score}</span></div><p>{item.excerpt}…</p></div>)}</div></details>}
      {message.timing && <div className="message-meta"><span>{message.provider}</span><span>{message.model}</span><span>检索 {message.timing.searchMs}ms</span><span>总耗时 {message.timing.totalMs}ms</span></div>}
    </div>{user && <Avatar user/>}
  </article>;
}

function Chat({ stats, providers, provider, setProvider, refresh, notify }) {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("chat");
  const [routeMode, setRouteMode] = useState("infra");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([{ role: "assistant", answer: "你好，我是知问。你可以直接调用 DeepSeek，也可以开启 AI Infra，体验 C++ 请求调度、Token Budget 批处理、限流和性能指标。" }]);
  const list = useRef(null);
  useEffect(() => {
    if (list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [messages, loading]);
  async function submit(event) {
    event?.preventDefault(); const value = question.trim(); if (!value || loading) return;
    setQuestion(""); setMessages((items) => [...items, { role: "user", answer: value }]); setLoading(true);
    try {
      const history = messages.slice(1).slice(-12).map((item) => ({ role: item.role, content: item.answer }));
      const result = await api("/api/ask", { method: "POST", body: JSON.stringify({ question: value, model_id: provider, mode, history, route_mode: routeMode, use_ai_infra: routeMode === "infra" }) });
      setMessages((items) => [...items, { role: "assistant", ...result }]); refresh();
      if (result.warning) notify(result.warning, true);
    } catch (error) { setMessages((items) => [...items, { role: "assistant", answer: error.message, failed: true }]); }
    finally { setLoading(false); }
  }
  return <section className="view chat-view">
    <Header eyebrow="AI KNOWLEDGE COPILOT" title="与你的 AI 一起思考" description="像常用 AI 助手一样自由对话，也可以切换到知识库模式获得带引用的回答。" action={<ModelSelector providers={providers} value={provider} onChange={setProvider}/>}/>
    <div className="stat-row"><Stat value={stats.documents} label="知识文档"/><Stat value={stats.chunks} label="可检索片段"/><Stat value={stats.conversations} label="历史问答"/><Stat value={stats.characters} label="知识字符"/></div>
    <div className="chat-shell"><div className="chat-toolbar"><div className="mode-switch"><button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>✦ 通用对话</button><button className={mode === "knowledge" ? "active" : ""} onClick={() => setMode("knowledge")}>▦ 知识库问答</button><button className={mode === "auto" ? "active" : ""} onClick={() => setMode("auto")}>◈ 自动模式</button></div><div className="route-switch"><button className={routeMode === "direct" ? "active" : ""} onClick={() => setRouteMode("direct")}>DeepSeek 直连</button><button className={routeMode === "baseline" ? "active" : ""} onClick={() => setRouteMode("baseline")}>Python 基线</button><button className={routeMode === "infra" ? "active" : ""} onClick={() => setRouteMode("infra")}>AI Infra 加速</button></div></div><div className="route-note">{{direct: "标准消息格式从 Python 直接调用 DeepSeek API", baseline: "与 AI Infra 使用相同 Prompt 和 Token 条件，但跳过 C++ 调度层", infra: "请求经过 C++ 队列、Token Budget 批处理、限流与指标链路"}[routeMode]}</div><div className="messages" ref={list}>{messages.map((item, index) => <Message message={item} key={index}/>)}{loading && <div className="message assistant"><Avatar/><div className="bubble typing"><i/><i/><i/><span>{mode === "chat" ? "大模型正在思考" : "正在检索知识并调用模型"}</span></div></div>}</div>
      <form className="composer" onSubmit={submit}><div className="composer-input"><textarea value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} placeholder={mode === "chat" ? "向当前模型询问任何问题……" : "询问知识库中的内容……"}/><span>Enter 发送 · Shift + Enter 换行 · 保留最近 12 条上下文</span></div><button disabled={!question.trim() || loading}>↑</button></form>
    </div>
  </section>;
}

function Knowledge({ documents, reload, notify }) {
  const [form, setForm] = useState({ title: "", content: "", source: "manual" });
  const [saving, setSaving] = useState(false);
  async function fileChange(event) { const file = event.target.files[0]; if (!file) return; if (file.size > 5e6) return notify("文件不能超过 5MB", true); setForm({ title: form.title || file.name.replace(/\.[^.]+$/, ""), content: await file.text(), source: file.name }); }
  async function submit(event) { event.preventDefault(); setSaving(true); try { await api("/api/documents", { method: "POST", body: JSON.stringify(form) }); setForm({ title: "", content: "", source: "manual" }); await reload(); notify("资料已解析并加入知识空间"); } catch (error) { notify(error.message, true); } finally { setSaving(false); } }
  async function remove(id) { if (!confirm("确定删除这份资料及其全部知识片段吗？")) return; await api(`/api/documents/${id}`, { method: "DELETE" }); await reload(); notify("资料已删除"); }
  return <section className="view"><Header eyebrow="KNOWLEDGE WORKSPACE" title="构建你的知识空间" description="导入文本资料，系统会自动清洗、切分并建立可检索索引。"/>
    <div className="knowledge-grid"><form className="panel import-panel" onSubmit={submit}><div className="panel-heading"><div><span className="section-icon">＋</span><div><h2>添加知识</h2><p>TXT、Markdown、JSON、CSV</p></div></div><label className="file-button">浏览文件<input type="file" accept=".txt,.md,.markdown,.json,.csv" onChange={fileChange}/></label></div><label>资料名称<input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="例如：AI Infra 技术设计文档" required/></label><label>资料内容<textarea rows="15" value={form.content} onChange={(e) => setForm({ ...form, content: e.target.value })} placeholder="粘贴或从文件读取资料内容……" required/></label><button className="primary" disabled={saving}>{saving ? "正在构建索引…" : "加入知识空间"}</button></form>
      <div className="panel library"><div className="panel-heading"><div><span className="section-icon dark">▦</span><div><h2>资料库</h2><p>{documents.length} 份资料正在参与检索</p></div></div><button className="icon-button" onClick={reload}>↻</button></div><div className="document-list">{documents.length ? documents.map((doc) => <article className="document" key={doc.id}><div className="doc-icon">{doc.source.split(".").pop().slice(0,3).toUpperCase()}</div><div className="doc-info"><b>{doc.title}</b><span>{doc.characters.toLocaleString()} 字符 · {doc.chunkCount} 个知识片段</span><small>{new Date(doc.createdAt).toLocaleString()}</small></div><button onClick={() => remove(doc.id)}>×</button></article>) : <Empty title="知识空间还是空的" text="导入第一份资料后，即可开始知识问答"/>}</div></div>
    </div>
  </section>;
}

function Empty({ title, text }) { return <div className="empty"><span>◇</span><strong>{title}</strong><small>{text}</small></div>; }

function History({ conversations, reload }) {
  return <section className="view"><Header eyebrow="CONVERSATION ARCHIVE" title="问答记录" description="查看问题、模型、引用来源和端到端响应性能。" action={<button className="secondary" onClick={reload}>刷新记录</button>}/><div className="history-list">{conversations.length ? conversations.map((item) => <article className="panel history" key={item.id}><div className="history-top"><div><span>QUESTION</span><h3>{item.question}</h3></div><time>{new Date(item.createdAt).toLocaleString()}</time></div><p>{item.answer}</p><div className="history-meta"><span>{item.provider}</span><span>{item.model}</span><span>{item.citations?.length || 0} 条引用</span><span>{item.timing?.totalMs || 0}ms</span></div></article>) : <div className="panel"><Empty title="还没有问答记录" text="你的知识问答会安全地保存在这里"/></div>}</div></section>;
}

function Models({ providers, selected, setSelected, health, stats }) {
  return <section className="view"><Header eyebrow="MODEL ORCHESTRATION" title="连接你的模型生态" description="同一套 RAG 工作流，自由切换云端 API、本地模型和高性能推理基础设施。"/>
    <div className="system-grid"><div className="panel status-panel"><div className="large-status"><span className={health?.infraAvailable ? "online" : ""}/><div><small>唯一推理入口</small><h2>C++ AI Infra</h2><p>{health?.infraAvailable ? `已连接 ${health.infraUrl}` : "未启动：所有模型请求将被拒绝，不会绕过 Infra"}</p></div></div></div><div className="panel mini-stats"><Stat value={stats.documents} label="文档"/><Stat value={stats.chunks} label="片段"/><Stat value={stats.conversations} label="请求"/></div></div>
    <div className="provider-grid">{providers.map((item) => <button key={item.id} className={`provider-card ${selected === item.id ? "selected" : ""}`} onClick={() => setSelected(item.id)}><div><span className="provider-logo">{item.name.slice(0, 2)}</span><i className={item.configured ? "configured" : ""}/></div><h3>{item.name}</h3><b>{item.model}</b><p>{item.description}</p><div className="model-tags">{item.free && <em>免费</em>}{item.local && <em>本地</em>}{item.recommended && <em>推荐</em>}</div><small>{item.configured ? "● 配置完整" : "○ 需要配置 API Key"}</small></button>)}</div>
    <div className="flow"><Architecture number="01" title="React Client" text="对话与知识库交互"/><i>REST API</i><Architecture number="02" title="FastAPI RAG" text="检索、引用和上下文构建" active/><i>/v1/infer</i><Architecture number="03" title="C++ AI Infra" text="调度免费的 DeepSeek-R1"/></div>
  </section>;
}

function CodeWorkspace({ notify }) {
  const [root, setRoot] = useState(""); const [rootInput, setRootInput] = useState(""); const [files, setFiles] = useState([]); const [selected, setSelected] = useState("");
  const [content, setContent] = useState(""); const [proposal, setProposal] = useState(""); const [instruction, setInstruction] = useState(""); const [busy, setBusy] = useState(false);
  const loadFiles = useCallback(async () => { try { const data = await api("/api/code/files"); setRoot(data.root); setRootInput(data.root); setFiles(data.files); } catch (error) { notify(error.message, true); } }, [notify]);
  useEffect(() => { loadFiles(); }, [loadFiles]);
  async function openFile(path) { setBusy(true); try { const data = await api(`/api/code/file?path=${encodeURIComponent(path)}`); setSelected(path); setContent(data.content); setProposal(""); } catch (error) { notify(error.message, true); } finally { setBusy(false); } }
  async function propose() { if (!selected || !instruction.trim()) return; setBusy(true); try { const data = await api("/api/code/propose", { method: "POST", body: JSON.stringify({ path: selected, instruction, content }) }); setProposal(data.content); notify(`已由 ${data.provider} 生成修改建议`); } catch (error) { notify(error.message, true); } finally { setBusy(false); } }
  async function save(value = content) { if (!selected) return; setBusy(true); try { await api("/api/code/file", { method: "PUT", body: JSON.stringify({ path: selected, content: value }) }); setContent(value); setProposal(""); setInstruction(""); notify("代码已保存到本地项目"); } catch (error) { notify(error.message, true); } finally { setBusy(false); } }
  async function switchWorkspace(event) { event.preventDefault(); if (!rootInput.trim()) return; setBusy(true); try { const data = await api("/api/code/workspace", { method: "PUT", body: JSON.stringify({ root: rootInput.trim() }) }); setRoot(data.root); setRootInput(data.root); setFiles(data.files); setSelected(""); setContent(""); setProposal(""); notify(`已切换工作目录，共 ${data.files.length} 个代码文件`); } catch (error) { notify(error.message, true); } finally { setBusy(false); } }
  return <section className="view"><Header eyebrow="LOCAL CODING AGENT" title="代码工作台" description="浏览本地项目，让 DeepSeek 通过 AI Infra 生成修改建议，确认后再写入文件。" action={<button className="secondary" onClick={loadFiles}>刷新文件</button>}/>
    <form className="workspace-picker" onSubmit={switchWorkspace}><label>本地工作目录</label><input value={rootInput} onChange={(event) => setRootInput(event.target.value)} placeholder="例如 C:\\Users\\name\\Desktop\\my-project"/><button className="secondary" disabled={busy || rootInput.trim() === root}>切换目录</button></form><div className="code-root">当前目录：{root || "正在连接……"} · {files.length} 个代码文件</div><div className="code-layout"><aside className="panel code-files">{files.map((file) => <button className={selected === file ? "active" : ""} key={file} onClick={() => openFile(file)}>{file}</button>)}</aside>
      <div className="panel code-editor"><div className="code-title"><b>{selected || "请选择代码文件"}</b><button className="secondary" disabled={!selected || busy} onClick={() => save(content)}>保存当前文件</button></div><textarea spellCheck="false" value={content} disabled={!selected || busy} onChange={(event) => setContent(event.target.value)} placeholder="从左侧选择文件……"/>
        <div className="code-agent"><textarea rows="3" value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="例如：为这个接口增加参数校验，并保持现有功能不变"/><button className="primary" disabled={!selected || !instruction.trim() || busy} onClick={propose}>{busy ? "处理中…" : "让 AI 生成修改"}</button></div>
        {proposal && <div className="proposal"><div><b>AI 修改预览</b><span>尚未写入磁盘</span></div><textarea spellCheck="false" value={proposal} onChange={(event) => setProposal(event.target.value)}/><div className="proposal-actions"><button className="secondary" onClick={() => setProposal("")}>放弃</button><button className="primary" onClick={() => save(proposal)}>确认并写入</button></div></div>}
      </div></div></section>;
}

function Architecture({ number, title, text, active }) { return <div className={`arch ${active ? "active" : ""}`}><span>{number}</span><h3>{title}</h3><p>{text}</p></div>; }

export default function App() {
  const [view, setView] = useState("chat"); const [stats, setStats] = useState({}); const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]); const [conversations, setConversations] = useState([]); const [providers, setProviders] = useState([]); const [provider, setProvider] = useState("deepseek-chat"); const [toast, setToast] = useState(null);
  const notify = useCallback((message, error = false) => { setToast({ message, error }); setTimeout(() => setToast(null), 3500); }, []);
  const overview = useCallback(async () => { try { const [h, s, p] = await Promise.all([api("/api/health"), api("/api/stats"), api("/api/providers")]); setHealth(h); setStats(s); setProviders(p.providers); setProvider((current) => current || p.default); } catch { setHealth(null); } }, []);
  const loadDocuments = useCallback(async () => { const data = await api("/api/documents"); setDocuments(data.documents); await overview(); }, [overview]);
  const loadHistory = useCallback(async () => { const data = await api("/api/conversations"); setConversations(data.conversations); }, []);
  useEffect(() => { overview(); loadDocuments(); loadHistory(); }, [overview, loadDocuments, loadHistory]);
  return <div className="app"><Sidebar view={view} setView={setView} health={health}/><main>{view === "chat" && <Chat stats={stats} providers={providers} provider={provider} setProvider={setProvider} refresh={() => { overview(); loadHistory(); }} notify={notify}/>} {view === "knowledge" && <Knowledge documents={documents} reload={loadDocuments} notify={notify}/>} {view === "code" && <CodeWorkspace notify={notify}/>} {view === "history" && <History conversations={conversations} reload={loadHistory}/>} {view === "system" && <Models providers={providers} selected={provider} setSelected={setProvider} health={health} stats={stats}/>}</main><div className={`toast ${toast ? "show" : ""} ${toast?.error ? "error" : ""}`}>{toast?.message}</div></div>;
}
