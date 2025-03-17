"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[8155],{6879:(s,e,n)=>{n.r(e),n.d(e,{assets:()=>c,contentTitle:()=>t,default:()=>h,frontMatter:()=>r,metadata:()=>i,toc:()=>d});const i=JSON.parse('{"id":"guides/logs","title":"Logging System","description":"Goose uses a unified storage system for conversations and interactions. All conversations and interactions (both CLI and Desktop) are stored locally in the following locations:","source":"@site/docs/guides/logs.md","sourceDirName":"guides","slug":"/guides/logs","permalink":"/goose/pr-preview/pr-1723/docs/guides/logs","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":5,"frontMatter":{"title":"Logging System","sidebar_position":5},"sidebar":"tutorialSidebar","previous":{"title":"LLM Rate Limits","permalink":"/goose/pr-preview/pr-1723/docs/guides/handling-llm-rate-limits-with-goose"},"next":{"title":"File Management","permalink":"/goose/pr-preview/pr-1723/docs/guides/file-management"}}');var o=n(4848),l=n(8453);const r={title:"Logging System",sidebar_position:5},t="Goose Logging System",c={},d=[{value:"Command History",id:"command-history",level:2},{value:"Session Records",id:"session-records",level:2},{value:"System Logs",id:"system-logs",level:2},{value:"Main System Log",id:"main-system-log",level:3},{value:"Desktop Application Log",id:"desktop-application-log",level:3},{value:"CLI Logs",id:"cli-logs",level:3},{value:"Server Logs",id:"server-logs",level:3}];function a(s){const e={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",p:"p",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",ul:"ul",...(0,l.R)(),...s.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(e.header,{children:(0,o.jsx)(e.h1,{id:"goose-logging-system",children:"Goose Logging System"})}),"\n",(0,o.jsxs)(e.p,{children:["Goose uses a unified storage system for conversations and interactions. All conversations and interactions (both CLI and Desktop) are stored ",(0,o.jsx)(e.strong,{children:"locally"})," in the following locations:"]}),"\n",(0,o.jsxs)(e.table,{children:[(0,o.jsx)(e.thead,{children:(0,o.jsxs)(e.tr,{children:[(0,o.jsx)(e.th,{children:(0,o.jsx)(e.strong,{children:"Type"})}),(0,o.jsx)(e.th,{children:(0,o.jsx)(e.strong,{children:"Unix-like (macOS, Linux)"})}),(0,o.jsx)(e.th,{children:(0,o.jsx)(e.strong,{children:"Windows"})})]})}),(0,o.jsxs)(e.tbody,{children:[(0,o.jsxs)(e.tr,{children:[(0,o.jsx)(e.td,{children:(0,o.jsx)(e.strong,{children:"Command History"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"~/.config/goose/history.txt"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\history.txt"})})]}),(0,o.jsxs)(e.tr,{children:[(0,o.jsx)(e.td,{children:(0,o.jsx)(e.strong,{children:"Session Records"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"~/.local/share/goose/sessions/"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\sessions\\"})})]}),(0,o.jsxs)(e.tr,{children:[(0,o.jsx)(e.td,{children:(0,o.jsx)(e.strong,{children:"System Logs"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"~/.local/state/goose/logs/"})}),(0,o.jsx)(e.td,{children:(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\logs\\"})})]})]})]}),"\n",(0,o.jsx)(e.admonition,{title:"Privacy",type:"info",children:(0,o.jsx)(e.p,{children:"Goose is a local application and all log files are stored locally. These logs are never sent to external servers or third parties, ensuring that all data remains private and under your control."})}),"\n",(0,o.jsx)(e.h2,{id:"command-history",children:"Command History"}),"\n",(0,o.jsx)(e.p,{children:"Goose stores command history persistently across chat sessions, allowing Goose to recall previous commands."}),"\n",(0,o.jsx)(e.p,{children:"Command history logs are stored in:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["Unix-like: ",(0,o.jsx)(e.code,{children:" ~/.config/goose/history.txt"})]}),"\n",(0,o.jsxs)(e.li,{children:["Windows: ",(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\history.txt"})]}),"\n"]}),"\n",(0,o.jsx)(e.h2,{id:"session-records",children:"Session Records"}),"\n",(0,o.jsxs)(e.p,{children:["Goose maintains session records in ",(0,o.jsx)(e.code,{children:"~/.local/share/goose/sessions/"})," that track the conversation history and interactions for each session. These files use the ",(0,o.jsx)(e.code,{children:".jsonl"})," format (JSON Lines), where each line is a valid JSON object representing a message or interaction."]}),"\n",(0,o.jsxs)(e.p,{children:["Session files are named with the pattern ",(0,o.jsx)(e.code,{children:"[session-id].jsonl"})," where the session ID matches the identifier used in the corresponding log files. For example, ",(0,o.jsx)(e.code,{children:"ccK9OTmS.jsonl"})," corresponds to log files like ",(0,o.jsx)(e.code,{children:"20250211_133920-ccK9OTmS.log"}),"."]}),"\n",(0,o.jsx)(e.p,{children:"Each session file contains a chronological record of:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["User messages and commands  (commands are also stored persistently in ",(0,o.jsx)(e.code,{children:"history.txt"}),")"]}),"\n",(0,o.jsx)(e.li,{children:"Assistant (Goose) responses"}),"\n",(0,o.jsx)(e.li,{children:"Tool requests and their results"}),"\n",(0,o.jsx)(e.li,{children:"Timestamps for all interactions"}),"\n",(0,o.jsx)(e.li,{children:"Role information (user/assistant)"}),"\n",(0,o.jsx)(e.li,{children:"Message content and formatting"}),"\n",(0,o.jsxs)(e.li,{children:["Tool call details including:","\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Tool IDs"}),"\n",(0,o.jsx)(e.li,{children:"Arguments passed"}),"\n",(0,o.jsx)(e.li,{children:"Results returned"}),"\n",(0,o.jsx)(e.li,{children:"Success/failure status"}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,o.jsx)(e.p,{children:"Each line in a session file is a JSON object with the following key fields:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.code,{children:"role"}),': Identifies the source ("user" or "assistant")']}),"\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.code,{children:"created"}),": Timestamp of the interaction"]}),"\n",(0,o.jsxs)(e.li,{children:[(0,o.jsx)(e.code,{children:"content"}),": Array of interaction elements, which may include:","\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Text messages"}),"\n",(0,o.jsx)(e.li,{children:"Tool requests"}),"\n",(0,o.jsx)(e.li,{children:"Tool responses"}),"\n",(0,o.jsx)(e.li,{children:"Error messages"}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,o.jsx)(e.h2,{id:"system-logs",children:"System Logs"}),"\n",(0,o.jsx)(e.h3,{id:"main-system-log",children:"Main System Log"}),"\n",(0,o.jsx)(e.p,{children:"The main system log locations:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["Unix-like: ",(0,o.jsx)(e.code,{children:"~/.local/state/goose/logs/goose.log"})]}),"\n",(0,o.jsxs)(e.li,{children:["Windows: ",(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\logs\\goose.log"})]}),"\n"]}),"\n",(0,o.jsx)(e.p,{children:"This log contains general application-level logging including:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Session file locations"}),"\n",(0,o.jsx)(e.li,{children:"Token usage statistics as well as token counts (input, output, total)"}),"\n",(0,o.jsx)(e.li,{children:"LLM information (model names, versions)"}),"\n"]}),"\n",(0,o.jsx)(e.h3,{id:"desktop-application-log",children:"Desktop Application Log"}),"\n",(0,o.jsx)(e.p,{children:"The desktop application maintains its own logs:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["macOS: ",(0,o.jsx)(e.code,{children:"~/Library/Application Support/Goose/logs/main.log"})]}),"\n",(0,o.jsxs)(e.li,{children:["Windows: ",(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\logs\\main.log"})]}),"\n"]}),"\n",(0,o.jsxs)(e.p,{children:["The Desktop application follows platform conventions for its own operational logs and state data, but uses the standard Goose ",(0,o.jsx)(e.a,{href:"#session-records",children:"session records"})," for actual conversations and interactions. This means your conversation history is consistent regardless of which interface you use to interact with Goose."]}),"\n",(0,o.jsx)(e.h3,{id:"cli-logs",children:"CLI Logs"}),"\n",(0,o.jsx)(e.p,{children:"CLI logs are stored in:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["Unix-like: ",(0,o.jsx)(e.code,{children:"~/.local/state/goose/logs/cli/"})]}),"\n",(0,o.jsxs)(e.li,{children:["Windows: ",(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\logs\\cli\\"})]}),"\n"]}),"\n",(0,o.jsx)(e.p,{children:"CLI session logs contain:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Tool invocations and responses"}),"\n",(0,o.jsx)(e.li,{children:"Command execution details"}),"\n",(0,o.jsx)(e.li,{children:"Session identifiers"}),"\n",(0,o.jsx)(e.li,{children:"Timestamps"}),"\n"]}),"\n",(0,o.jsx)(e.p,{children:"Extension logs contain:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Tool initialization"}),"\n",(0,o.jsx)(e.li,{children:"Tool capabilities and schemas"}),"\n",(0,o.jsx)(e.li,{children:"Extension-specific operations"}),"\n",(0,o.jsx)(e.li,{children:"Command execution results"}),"\n",(0,o.jsx)(e.li,{children:"Error messages and debugging information"}),"\n",(0,o.jsx)(e.li,{children:"Extension configuration states"}),"\n",(0,o.jsx)(e.li,{children:"Extension-specific protocol information"}),"\n"]}),"\n",(0,o.jsx)(e.h3,{id:"server-logs",children:"Server Logs"}),"\n",(0,o.jsx)(e.p,{children:"Server logs are stored in:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["Unix-like: ",(0,o.jsx)(e.code,{children:"~/.local/state/goose/logs/server/"})]}),"\n",(0,o.jsxs)(e.li,{children:["Windows: ",(0,o.jsx)(e.code,{children:"%APPDATA%\\Block\\goose\\data\\logs\\server\\"})]}),"\n"]}),"\n",(0,o.jsxs)(e.p,{children:["The Server logs contain information about the Goose daemon (",(0,o.jsx)(e.code,{children:"goosed"}),"), which is a local server process that runs on your computer. This server component manages communication between the CLI, extensions, and LLMs."]}),"\n",(0,o.jsx)(e.p,{children:"Server logs include:"}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsx)(e.li,{children:"Server initialization details"}),"\n",(0,o.jsx)(e.li,{children:"JSON-RPC communication logs"}),"\n",(0,o.jsx)(e.li,{children:"Server capabilities"}),"\n",(0,o.jsx)(e.li,{children:"Protocol version information"}),"\n",(0,o.jsx)(e.li,{children:"Client-server interactions"}),"\n",(0,o.jsx)(e.li,{children:"Extension loading and initialization"}),"\n",(0,o.jsx)(e.li,{children:"Tool definitions and schemas"}),"\n",(0,o.jsx)(e.li,{children:"Extension instructions and capabilities"}),"\n",(0,o.jsx)(e.li,{children:"Debug-level transport information"}),"\n",(0,o.jsx)(e.li,{children:"System capabilities and configurations"}),"\n",(0,o.jsx)(e.li,{children:"Operating system information"}),"\n",(0,o.jsx)(e.li,{children:"Working directory information"}),"\n",(0,o.jsx)(e.li,{children:"Transport layer communication details"}),"\n",(0,o.jsx)(e.li,{children:"Message parsing and handling information"}),"\n",(0,o.jsx)(e.li,{children:"Request/response cycles"}),"\n",(0,o.jsx)(e.li,{children:"Error states and handling"}),"\n",(0,o.jsx)(e.li,{children:"Extension initialization sequences"}),"\n"]})]})}function h(s={}){const{wrapper:e}={...(0,l.R)(),...s.components};return e?(0,o.jsx)(e,{...s,children:(0,o.jsx)(a,{...s})}):a(s)}},8453:(s,e,n)=>{n.d(e,{R:()=>r,x:()=>t});var i=n(6540);const o={},l=i.createContext(o);function r(s){const e=i.useContext(l);return i.useMemo((function(){return"function"==typeof s?s(e):{...e,...s}}),[e,s])}function t(s){let e;return e=s.disableParentContext?"function"==typeof s.components?s.components(o):s.components||o:r(s.components),i.createElement(l.Provider,{value:e},s.children)}}}]);