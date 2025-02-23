"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6879],{177:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>h,contentTitle:()=>d,default:()=>x,frontMatter:()=>c,metadata:()=>t,toc:()=>u});const t=JSON.parse('{"id":"tutorials/tavily-mcp","title":"Tavily Web Search Extension","description":"Add Tavily MCP Server as a Goose Extension","source":"@site/docs/tutorials/tavily-mcp.md","sourceDirName":"tutorials","slug":"/tutorials/tavily-mcp","permalink":"/goose/pr-preview/pr-1357/docs/tutorials/tavily-mcp","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Tavily Web Search Extension","description":"Add Tavily MCP Server as a Goose Extension"},"sidebar":"tutorialSidebar","previous":{"title":"Memory Extension","permalink":"/goose/pr-preview/pr-1357/docs/tutorials/memory-mcp"},"next":{"title":"Architecture Overview","permalink":"/goose/pr-preview/pr-1357/docs/category/architecture-overview"}}');var s=i(4848),o=i(8453),a=i(5537),r=i(9329),l=i(5887);const c={title:"Tavily Web Search Extension",description:"Add Tavily MCP Server as a Goose Extension"},d=void 0,h={},u=[{value:"Configuration",id:"configuration",level:2},{value:"Example Usage",id:"example-usage",level:2},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3}];function m(e){const n={a:"a",admonition:"admonition",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(l.A,{videoUrl:"https://youtube.com/embed/mxS2G9afGxE"}),"\n",(0,s.jsxs)(n.p,{children:["This tutorial covers how to add the ",(0,s.jsx)(n.a,{href:"https://github.com/RamXX/mcp-tavily",children:"Tavily Web Search MCP Server"})," as a Goose extension to enable AI-powered web search functionality."]}),"\n",(0,s.jsxs)(n.admonition,{title:"TLDR",type:"tip",children:[(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Command"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"uvx mcp-tavily\n"})}),(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Environment Variable"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"TAVILY_API_KEY: <YOUR_API_KEY>\n"})})]}),"\n",(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsxs)(n.p,{children:["Note that you'll need ",(0,s.jsx)(n.a,{href:"https://docs.astral.sh/uv/#installation",children:"uv"})," installed on your system to run this command, as it uses ",(0,s.jsx)(n.code,{children:"uvx"}),"."]})}),"\n",(0,s.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,s.jsxs)(a.A,{groupId:"interface",children:[(0,s.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Run the ",(0,s.jsx)(n.code,{children:"configure"})," command:"]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,s.jsxs)(n.ol,{start:"2",children:["\n",(0,s.jsxs)(n.li,{children:["Choose to add a ",(0,s.jsx)(n.code,{children:"Command-line Extension"})]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c6  What type of extension would you like to add?\n  \u2502  \u25cb Built-in Extension \n  // highlight-start    \n  \u2502  \u25cf Command-line Extension (Run a local command or script)\n  // highlight-end    \n  \u2502  \u25cb Remote Extension \n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"3",children:["\n",(0,s.jsx)(n.li,{children:"Give your extension a name"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  // highlight-start\n  \u25c6  What would you like to call this extension?\n  \u2502  tavily\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"4",children:["\n",(0,s.jsx)(n.li,{children:"Enter the command"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  \u25c7  What would you like to call this extension?\n  \u2502  tavily\n  \u2502\n  // highlight-start\n  \u25c6  What command should be run?\n  \u2502  uvx mcp-tavily\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"5",children:["\n",(0,s.jsxs)(n.li,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://tavily.com/",children:"Tavily API Key"})," and paste it in."]}),"\n"]}),(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsxs)(n.p,{children:["You can get your API key by signing up at ",(0,s.jsx)(n.a,{href:"https://app.tavily.com/",children:"tavily.com"})," and navigating to your account settings."]})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:" \u250c   goose-configure \n \u2502\n \u25c7  What would you like to configure?\n \u2502  Add Extension \n \u2502\n \u25c7  What type of extension would you like to add?\n \u2502  Command-line Extension \n \u2502\n \u25c7  What would you like to call this extension?\n \u2502  tavily\n \u2502\n \u25c7  What command should be run?\n \u2502  uvx mcp-tavily\n \u2502\n // highlight-start\n \u25c6  Would you like to add environment variables?\n \u2502  Yes \n \u2502\n \u25c7  Environment variable name:\n \u2502  TAVILY_API_KEY\n \u2502\n \u25c7  Environment variable value:\n \u2502  \u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\n \u2502\n \u25c7  Add another environment variable?\n \u2502  No \n // highlight-end\n \u2514  Added tavily extension\n"})})]}),(0,s.jsx)(r.A,{value:"ui",label:"Goose Desktop",children:(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Settings"})]}),"\n",(0,s.jsxs)(n.li,{children:["Under ",(0,s.jsx)(n.code,{children:"Extensions"}),", click the ",(0,s.jsx)(n.code,{children:"Add"})," link"]}),"\n",(0,s.jsxs)(n.li,{children:["On the ",(0,s.jsx)(n.code,{children:"Add Extension Manually"})," modal, enter the following:","\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Type"}),": ",(0,s.jsx)(n.code,{children:"Standard IO"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"ID"}),": ",(0,s.jsx)(n.code,{children:"tavily-mcp"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"Tavily"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Description"}),": ",(0,s.jsx)(n.code,{children:"Tavily Web Search"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Command"}),": ",(0,s.jsx)(n.code,{children:"uvx mcp-tavily"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Environment Variables"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"TAVILY_API_KEY"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Value"}),": (",(0,s.jsxs)(n.em,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://tavily.com/",children:"Tavily API Key"})," and paste it in."]}),")"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add"})," button"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add Extension"})," button"]}),"\n"]})})]}),"\n",(0,s.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,s.jsx)(n.p,{children:"The Tavily MCP server enables AI-powered web search functionality in your Goose interactions. When you obtain your API key, you'll have access to Tavily's advanced search capabilities including:"}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsx)(n.li,{children:"Real-time web search"}),"\n",(0,s.jsx)(n.li,{children:"Structured data extraction"}),"\n",(0,s.jsx)(n.li,{children:"Content analysis"}),"\n",(0,s.jsx)(n.li,{children:"Topic-specific search"}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"Search for recent news about artificial intelligence breakthroughs in medicine and summarize the key findings.\n"})}),"\n",(0,s.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:'I\'ll use the Tavily news search tool to find recent news about AI breakthroughs in medicine.\n\n\u2500\u2500\u2500 tavily_news_search | tavily \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nmax_results: 5\nquery: ...\n\n\n\nLet me summarize the key findings from the recent news about AI in medicine:\n\n1. Personalized AI-Driven Healthcare:\n- Development of AI systems that act as "personal doctors" that work alongside human clinicians rather than replacing them\n- Focus on augmenting clinical capabilities while maintaining human oversight\n\n2. Breakthrough in Medical Imaging:\n- A significant advancement in CT scan analysis has been reported in Nature\n- Development of a "Unified total-body CT image" (Uni-CT) algorithm that:\n  - Automatically generates single series of total-body CT images\n  - Concatenates multiple images of segmented organs using organ-specific window settings\n  - Reduces diagnosis time by 35% without compromising accuracy\n  - Improves efficiency in trauma cases\n\n3. Future Trends:\n- Movement toward democratized AI in healthcare, potentially reducing the gap between large and small healthcare providers\n- Development of hyper-personalized medical experiences powered by AI\n- Integration of AI systems that can perform human-like reasoning in medical contexts\n\nThe most concrete breakthrough appears to be in medical imaging, with the Uni-CT algorithm showing promising results in improving both the speed and accuracy of trauma diagnosis. This represents a practical application of AI that could have immediate benefits in emergency medicine.\n\nWould you like me to search for more specific information about any of these developments or explore other areas of AI in medicine?\n'})}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsxs)(n.p,{children:["You can adjust the ",(0,s.jsx)(n.a,{href:"https://github.com/RamXX/mcp-tavily?tab=readme-ov-file#prompts",children:"search parameters"})," for different types of queries and depths of information. The extension supports both quick searches and comprehensive research."]})})]})}function x(e={}){const{wrapper:n}={...(0,o.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(m,{...e})}):m(e)}},5887:(e,n,i)=>{i.d(n,{A:()=>o});i(6540);var t=i(2362),s=i(4848);const o=e=>{let{videoUrl:n}=e;return(0,s.jsxs)("div",{children:[(0,s.jsx)(t.A,{type:"info",icon:"\ud83c\udfa5",title:"Plug & Play",className:"alert--video",children:(0,s.jsxs)("details",{children:[(0,s.jsx)("summary",{children:"Watch the demo"}),(0,s.jsx)("div",{style:{textAlign:"center",margin:"20px 0"},children:(0,s.jsx)("iframe",{width:"100%",height:"540",src:n,title:"YouTube Short",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})]})}),(0,s.jsx)("hr",{})]})}}}]);