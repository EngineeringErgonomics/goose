"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6959],{8553:(e,o,i)=>{i.r(o),i.d(o,{assets:()=>d,contentTitle:()=>a,default:()=>u,frontMatter:()=>r,metadata:()=>t,toc:()=>l});const t=JSON.parse('{"id":"guidance/using-goose-free","title":"Using Goose for Free","description":"Goose is a free and open-source developer agent that you can start using right away, but not all supported LLM Providers provide a free tier.","source":"@site/docs/guidance/using-goose-free.md","sourceDirName":"guidance","slug":"/guidance/using-goose-free","permalink":"/goose/v1/docs/guidance/using-goose-free","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/guidance/using-goose-free.md","tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2,"title":"Using Goose for Free"},"sidebar":"tutorialSidebar","previous":{"title":"Getting Started","permalink":"/goose/v1/docs/guidance/getting-started"},"next":{"title":"Managing Goose Sessions","permalink":"/goose/v1/docs/guidance/managing-goose-sessions"}}');var s=i(4848),n=i(8453);const r={sidebar_position:2,title:"Using Goose for Free"},a="Using Goose for Free",d={},l=[{value:"Google Gemini",id:"google-gemini",level:2},{value:"Limitations",id:"limitations",level:2}];function c(e){const o={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",header:"header",hr:"hr",mdxAdmonitionTitle:"mdxAdmonitionTitle",p:"p",pre:"pre",...(0,n.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(o.header,{children:(0,s.jsx)(o.h1,{id:"using-goose-for-free",children:"Using Goose for Free"})}),"\n",(0,s.jsxs)(o.p,{children:["Goose is a free and open-source developer agent that you can start using right away, but not all supported ",(0,s.jsx)(o.a,{href:"https://block.github.io/goose/plugins/providers.html",children:"LLM Providers"})," provide a free tier."]}),"\n",(0,s.jsx)(o.p,{children:"Below, we outline a couple of free options and how to get started with them."}),"\n",(0,s.jsx)(o.h2,{id:"google-gemini",children:"Google Gemini"}),"\n",(0,s.jsxs)(o.p,{children:["Google Gemini provides free access to its AI capabilities with some limitations. To start using the Gemini API with Goose, you need an API Key from ",(0,s.jsx)(o.a,{href:"https://aistudio.google.com/app/apikey",children:"Google AI studio"}),"."]}),"\n",(0,s.jsxs)(o.p,{children:["Update your ",(0,s.jsx)(o.code,{children:"~/.config/goose/profiles.yaml"})," file with the following configuration:"]}),"\n",(0,s.jsx)(o.pre,{children:(0,s.jsx)(o.code,{className:"language-yaml",metastring:'title="profiles.yaml"',children:"default:\n  provider: google\n  processor: gemini-1.5-flash\n  accelerator: gemini-1.5-flash\n  moderator: passive\n  toolkits:\n  - name: developer\n    requires: {}\n"})}),"\n",(0,s.jsxs)(o.p,{children:["When you run ",(0,s.jsx)(o.code,{children:"goose session start"}),", you will be prompted to enter your Google API Key."]}),"\n",(0,s.jsxs)(o.admonition,{type:"info",children:[(0,s.jsx)(o.mdxAdmonitionTitle,{}),(0,s.jsxs)(o.p,{children:["At the moment, the ",(0,s.jsx)(o.code,{children:"synopsis"})," toolkit isn't supported by Google Gemini, so we use the ",(0,s.jsx)(o.code,{children:"developer"})," toolkit to interact with the API."]})]}),"\n",(0,s.jsx)(o.h2,{id:"limitations",children:"Limitations"}),"\n",(0,s.jsx)(o.p,{children:"These free options are a great way to get started with Goose and explore its capabilities. However, if you need more advanced features or higher usage limits, you can always upgrade to a paid plan."}),"\n",(0,s.jsx)(o.hr,{}),"\n",(0,s.jsxs)(o.p,{children:["This guide will continue to be updated with more free options as they become available. If you have any questions or need help with a specific provider, feel free to reach out to us on ",(0,s.jsx)(o.a,{href:"https://discord.gg/block-opensource",children:"Discord"})," or on the ",(0,s.jsx)(o.a,{href:"https://github.com/block/goose",children:"Goose repo"}),"."]})]})}function u(e={}){const{wrapper:o}={...(0,n.R)(),...e.components};return o?(0,s.jsx)(o,{...e,children:(0,s.jsx)(c,{...e})}):c(e)}},8453:(e,o,i)=>{i.d(o,{R:()=>r,x:()=>a});var t=i(6540);const s={},n=t.createContext(s);function r(e){const o=t.useContext(n);return t.useMemo((function(){return"function"==typeof e?e(o):{...o,...e}}),[o,e])}function a(e){let o;return o=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:r(e.components),t.createElement(n.Provider,{value:o},e.children)}}}]);