"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6190],{2515:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>h,contentTitle:()=>c,default:()=>m,frontMatter:()=>d,metadata:()=>s,toc:()=>u});const s=JSON.parse('{"id":"tutorials/memory-mcp","title":"Memory Extension","description":"Use Memory MCP Server as a Goose Extension","source":"@site/docs/tutorials/memory-mcp.md","sourceDirName":"tutorials","slug":"/tutorials/memory-mcp","permalink":"/goose/pr-preview/pr-1529/docs/tutorials/memory-mcp","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Memory Extension","description":"Use Memory MCP Server as a Goose Extension"},"sidebar":"tutorialSidebar","previous":{"title":"Observability with Langfuse","permalink":"/goose/pr-preview/pr-1529/docs/tutorials/langfuse"},"next":{"title":"Puppeteer Extension","permalink":"/goose/pr-preview/pr-1529/docs/tutorials/puppeteer-mcp"}}');var t=o(4848),r=o(8453),i=o(5537),l=o(9329),a=o(5887);const d={title:"Memory Extension",description:"Use Memory MCP Server as a Goose Extension"},c=void 0,h={},u=[{value:"Configuration",id:"configuration",level:2},{value:"Why Use Memory?",id:"why-use-memory",level:2},{value:"Example Usage",id:"example-usage",level:2},{value:"Step 1: Teach Goose Your API Standards",id:"step-1-teach-goose-your-api-standards",level:3},{value:"Goose Prompt #1",id:"goose-prompt-1",level:4},{value:"Goose Output",id:"goose-output",level:4},{value:"Step 2: Use Stored Knowledge to Create a New API Endpoint",id:"step-2-use-stored-knowledge-to-create-a-new-api-endpoint",level:3},{value:"Goose Prompt # 2",id:"goose-prompt--2",level:4},{value:"Goose Output",id:"goose-output-1",level:4}];function p(e){const n={blockquote:"blockquote",code:"code",em:"em",h2:"h2",h3:"h3",h4:"h4",li:"li",ol:"ol",p:"p",pre:"pre",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(a.A,{videoUrl:"https://youtube.com/embed/BZ0yrSLXQwk"}),"\n",(0,t.jsx)(n.p,{children:"The Memory extension turns Goose into a knowledgeable assistant by allowing you to teach it personalized key information (e.g. commands, code snippets, preferences and configurations) that it can recall and apply later. Whether it\u2019s project-specific (local) or universal (global) knowledge, Goose learns and remembers what matters most to you."}),"\n",(0,t.jsx)(n.p,{children:"This tutorial covers enabling and using the Memory MCP Server, which is a built-in Goose extension."}),"\n",(0,t.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Ensure extension is enabled:"}),"\n"]}),"\n",(0,t.jsxs)(i.A,{groupId:"interface",children:[(0,t.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Run the ",(0,t.jsx)(n.code,{children:"configure"})," command:"]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Choose to add a ",(0,t.jsx)(n.code,{children:"Built-in Extension"})]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c6  What type of extension would you like to add?\n// highlight-start    \n\u2502  \u25cf Built-in Extension (Use an extension that comes with Goose)\n// highlight-end  \n\u2502  \u25cb Command-line Extension \n\u2502  \u25cb Remote Extension \n\u2514  \n"})}),(0,t.jsxs)(n.ol,{start:"3",children:["\n",(0,t.jsxs)(n.li,{children:["Arrow down to the ",(0,t.jsx)(n.code,{children:"Memory"})," extension and press Enter"]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c7  What type of extension would you like to add?\n\u2502  Built-in Extension \n\u2502\n\u25c6  Which built-in extension would you like to enable?\n\u2502  \u25cb Developer Tools \n\u2502  \u25cb Computer Controller \n// highlight-start\n\u2502  \u25cf Memory \n// highlight-end\n|  \u25cb JetBrains\n\u2514  Enabled Memory extension\n"})})]}),(0,t.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Click ",(0,t.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,t.jsxs)(n.li,{children:["Click ",(0,t.jsx)(n.code,{children:"Settings"})]}),"\n",(0,t.jsxs)(n.li,{children:["Under ",(0,t.jsx)(n.code,{children:"Extensions"}),", toggle ",(0,t.jsx)(n.code,{children:"Memory"})," to on."]}),"\n",(0,t.jsxs)(n.li,{children:["Scroll to the top and click ",(0,t.jsx)(n.code,{children:"Exit"})," from the upper left corner"]}),"\n"]})})]}),"\n",(0,t.jsx)(n.h2,{id:"why-use-memory",children:"Why Use Memory?"}),"\n",(0,t.jsx)(n.p,{children:"With the Memory extension, you\u2019re not just storing static notes, you\u2019re teaching Goose how to assist you better. Imagine telling Goose:"}),"\n",(0,t.jsxs)(n.blockquote,{children:["\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.em,{children:"learn everything about MCP servers and save it to memory."})}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"Later, you can ask:"}),"\n",(0,t.jsxs)(n.blockquote,{children:["\n",(0,t.jsx)(n.p,{children:(0,t.jsx)(n.em,{children:"utilizing our MCP server knowledge help me build an MCP server."})}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"Goose will recall everything you\u2019ve saved as long as you instruct it to remember. This makes it easier to have consistent results when working with Goose."}),"\n",(0,t.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,t.jsx)(n.p,{children:"In this example, I\u2019ll show you how to make Goose a knowledgeable development assistant by teaching it about your project\u2019s API standards. With the Memory extension, Goose can store structured information and retrieve it later to help with your tasks."}),"\n",(0,t.jsx)(n.p,{children:"This means you no longer have to repeat yourself. Goose will remember your project\u2019s requirements and automatically apply them to new tasks."}),"\n",(0,t.jsxs)(i.A,{groupId:"interface",children:[(0,t.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Open a terminal and start a new Goose session:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose session\n"})})]}),(0,t.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Open a new session in Goose Desktop"}),"\n"]})})]}),"\n",(0,t.jsx)(n.h3,{id:"step-1-teach-goose-your-api-standards",children:"Step 1: Teach Goose Your API Standards"}),"\n",(0,t.jsx)(n.h4,{id:"goose-prompt-1",children:"Goose Prompt #1"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"Goose, learn our API development standards:\n\n- All endpoints must use TypeScript interfaces\n- Error responses follow RFC 7807 Problem Details spec\n- Rate limiting headers are required\n- Authentication uses JWT in Authorization header\n- All endpoints need OpenAPI documentation\n\nRemember this so you can help me follow these practices.\n"})}),"\n",(0,t.jsx)(n.h4,{id:"goose-output",children:"Goose Output"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"I've learned your API standards. I'll help enforce these requirements when working on API endpoints.\n\nStoring in:\n    - Category: development_standards\n    - Tags: #api #typescript #security #documentation\n    - Scope: global (these are general standards)\n\nI can now help with:\n    - API endpoint implementation\n    - Error handling patterns\n    - Authentication flows\n    - Documentation requirements\n"})}),"\n",(0,t.jsx)(n.h3,{id:"step-2-use-stored-knowledge-to-create-a-new-api-endpoint",children:"Step 2: Use Stored Knowledge to Create a New API Endpoint"}),"\n",(0,t.jsx)(n.h4,{id:"goose-prompt--2",children:"Goose Prompt # 2"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"Help me create a new endpoint for user preferences\n"})}),"\n",(0,t.jsx)(n.h4,{id:"goose-output-1",children:"Goose Output"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"I see you need a new endpoint. Based on our API development standards that I have stored in memory, I'll ensure it meets all our requirements.\n\n[.. Checking stored API standards ...]\n\nI've included all our required elements:\n\n\u2705 TypeScript interfaces\n\u2705 RFC 7807 error response format\n\u2705 Rate limiting\n\u2705 JWT auth header\n\u2705 OpenAPI documentation\n\nHere's your compliant endpoint:\n\ninterface UserPreferences {\n    userId: string;\n    theme: 'light' | 'dark';\n    notifications: boolean;\n    timezone: string;\n}\n\ninterface ErrorResponse { \n    type: string;\n    title: string;\n    status: number;\n    detail: string;\n    instance: string;\n}\n\n@Route('/api/users/:userId/preferences')\nclass UserPreferencesController {\n    @Get()\n    @RateLimit(100)  // Adding required rate limiting\n    @OpenAPI({ ... }) // Including mandatory OpenAPI docs\n    async getUserPreferences(\n        @Header('Authorization') token: string,  // JWT auth as per our standard\n        @Param('userId') userId: string\n    ): Promise<UserPreferences | ErrorResponse> {\n        // Implementation following our error handling pattern\n    }\n}\n\nWould you like me to implement the full endpoint logic?\n"})})]})}function m(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(p,{...e})}):p(e)}},5887:(e,n,o)=>{o.d(n,{A:()=>r});o(6540);var s=o(2362),t=o(4848);const r=e=>{let{videoUrl:n}=e;return(0,t.jsxs)("div",{children:[(0,t.jsx)(s.A,{type:"info",icon:"\ud83c\udfa5",title:"Plug & Play",className:"alert--video",children:(0,t.jsxs)("details",{children:[(0,t.jsx)("summary",{children:"Watch the demo"}),(0,t.jsx)("div",{style:{textAlign:"center",margin:"20px 0"},children:(0,t.jsx)("iframe",{width:"100%",height:"540",src:n,title:"YouTube Short",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})]})}),(0,t.jsx)("hr",{})]})}}}]);