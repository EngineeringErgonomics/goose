"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[2808],{1634:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>u,contentTitle:()=>i,default:()=>g,frontMatter:()=>a,metadata:()=>s,toc:()=>l});const s=JSON.parse('{"id":"tutorials/langfuse","title":"Observability with Langfuse","description":"Integrate Goose with Langfuse to observe performance","source":"@site/docs/tutorials/langfuse.md","sourceDirName":"tutorials","slug":"/tutorials/langfuse","permalink":"/goose/pr-preview/pr-1256/docs/tutorials/langfuse","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"description":"Integrate Goose with Langfuse to observe performance"},"sidebar":"tutorialSidebar","previous":{"title":"JetBrains Extension","permalink":"/goose/pr-preview/pr-1256/docs/tutorials/jetbrains-mcp"},"next":{"title":"Memory Extension","permalink":"/goose/pr-preview/pr-1256/docs/tutorials/memory-mcp"}}');var o=t(4848),r=t(8453);const a={description:"Integrate Goose with Langfuse to observe performance"},i="Observability with Langfuse",u={},l=[{value:"What is Langfuse",id:"what-is-langfuse",level:2},{value:"Set up Langfuse",id:"set-up-langfuse",level:2},{value:"Configure Goose to Connect to Langfuse",id:"configure-goose-to-connect-to-langfuse",level:2},{value:"Run Goose with Langfuse Integration",id:"run-goose-with-langfuse-integration",level:2}];function c(e){const n={a:"a",code:"code",em:"em",h1:"h1",h2:"h2",header:"header",img:"img",p:"p",pre:"pre",...(0,r.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.header,{children:(0,o.jsx)(n.h1,{id:"observability-with-langfuse",children:"Observability with Langfuse"})}),"\n",(0,o.jsx)(n.p,{children:"This tutorial covers how to integrate Goose with Langfuse to monitor your Goose requests and understand how the agent is performing."}),"\n",(0,o.jsx)(n.h2,{id:"what-is-langfuse",children:"What is Langfuse"}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.a,{href:"https://langfuse.com/",children:"Langfuse"})," is an ",(0,o.jsx)(n.a,{href:"https://github.com/langfuse/langfuse",children:"open-source"})," LLM engineering platform that enables teams to collaboratively monitor, evaluate, and debug their LLM applications."]}),"\n",(0,o.jsx)(n.h2,{id:"set-up-langfuse",children:"Set up Langfuse"}),"\n",(0,o.jsxs)(n.p,{children:["Sign up for Langfuse Cloud ",(0,o.jsx)(n.a,{href:"https://cloud.langfuse.com",children:"here"})," or self-host Langfuse ",(0,o.jsx)(n.a,{href:"https://langfuse.com/self-hosting/local",children:"Docker Compose"})," to get your Langfuse API keys."]}),"\n",(0,o.jsx)(n.h2,{id:"configure-goose-to-connect-to-langfuse",children:"Configure Goose to Connect to Langfuse"}),"\n",(0,o.jsx)(n.p,{children:"Set the environment variables so that Goose (written in Rust) can connect to the Langfuse server."}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"export LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-...\nexport LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-...\nexport LANGFUSE_URL=https://cloud.langfuse.com # EU data region \ud83c\uddea\ud83c\uddfa\n\n# https://us.cloud.langfuse.com if you're using the US region \ud83c\uddfa\ud83c\uddf8\n# https://localhost:3000 if you're self-hosting\n"})}),"\n",(0,o.jsx)(n.h2,{id:"run-goose-with-langfuse-integration",children:"Run Goose with Langfuse Integration"}),"\n",(0,o.jsx)(n.p,{children:"Now, you can run Goose and monitor your AI requests and actions through Langfuse."}),"\n",(0,o.jsx)(n.p,{children:"With Goose running and the environment variables set, Langfuse will start capturing traces of your Goose activities."}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.em,{children:(0,o.jsx)(n.a,{href:"https://cloud.langfuse.com/project/cloramnkj0002jz088vzn1ja4/traces/cea4ed38-0c44-4b0a-8c20-4b0b6b9e8d73?timestamp=2025-01-31T15%3A52%3A30.362Z&observation=7c8e5807-3c29-4c28-9c6f-7d7427be401f",children:"Example trace (public) in Langfuse"})})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{src:"https://langfuse.com//images/docs/goose-integration/goose-example-trace.png",alt:"Goose trace in Langfuse"})})]})}function g(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(c,{...e})}):c(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>a,x:()=>i});var s=t(6540);const o={},r=s.createContext(o);function a(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function i(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:a(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);