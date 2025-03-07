"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[1783],{9510:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>a,contentTitle:()=>l,default:()=>c,frontMatter:()=>r,metadata:()=>i,toc:()=>u});const i=JSON.parse('{"id":"tutorials/tutorial-extension","title":"Tutorial Extension","description":"Learn how to use Goose\'s built-in Tutorial extension for guided learning","source":"@site/docs/tutorials/tutorial-extension.md","sourceDirName":"tutorials","slug":"/tutorials/tutorial-extension","permalink":"/goose/pr-preview/pr-1535/docs/tutorials/tutorial-extension","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Tutorial Extension","description":"Learn how to use Goose\'s built-in Tutorial extension for guided learning"},"sidebar":"tutorialSidebar","previous":{"title":"Tavily Web Search Extension","permalink":"/goose/pr-preview/pr-1535/docs/tutorials/tavily-mcp"},"next":{"title":"Architecture Overview","permalink":"/goose/pr-preview/pr-1535/docs/category/architecture-overview"}}');var o=t(4848),s=t(8453);const r={title:"Tutorial Extension",description:"Learn how to use Goose's built-in Tutorial extension for guided learning"},l=void 0,a={},u=[{value:"Configuration",id:"configuration",level:2},{value:"Available Tutorials",id:"available-tutorials",level:2},{value:"Using the Tutorial Extension",id:"using-the-tutorial-extension",level:2},{value:"Starting a Tutorial",id:"starting-a-tutorial",level:3},{value:"During the Tutorial",id:"during-the-tutorial",level:3},{value:"Best Practices",id:"best-practices",level:3},{value:"Example Usage",id:"example-usage",level:2},{value:"Starting a Tutorial Session",id:"starting-a-tutorial-session",level:3},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3},{value:"Next Steps",id:"next-steps",level:2},{value:"Need Further Help?",id:"need-further-help",level:3}];function d(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.p,{children:"The Tutorial extension is a built-in feature of Goose that provides interactive, step-by-step guidance for learning various aspects of Goose and its capabilities. It's designed to help users get comfortable with Goose's features through hands-on practice."}),"\n",(0,o.jsx)(n.p,{children:"The Tutorial extension serves as an interactive learning tool that:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"Provides structured, step-by-step tutorials"}),"\n",(0,o.jsx)(n.li,{children:"Allows hands-on practice with Goose features"}),"\n",(0,o.jsx)(n.li,{children:"Offers immediate feedback and guidance"}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsx)(n.li,{children:"Ensure the Tutorial extension is enabled:"}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),"\n",(0,o.jsxs)(n.ol,{start:"2",children:["\n",(0,o.jsxs)(n.li,{children:["Choose to add a ",(0,o.jsx)(n.code,{children:"Built-in Extension"})]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c6  What type of extension would you like to add?\n// highlight-start  \n\u2502  \u25cf Built-in Extension (Use an extension that comes with Goose)\n// highlight-end  \n\u2502  \u25cb Command-line Extension \n\u2502  \u25cb Remote Extension \n\u2514  \n"})}),"\n",(0,o.jsxs)(n.ol,{start:"3",children:["\n",(0,o.jsxs)(n.li,{children:["Select the ",(0,o.jsx)(n.code,{children:"Tutorial"})," extension"]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c7  What type of extension would you like to add?\n\u2502  Built-in Extension \n\u2502\n\u25c6  Which built-in extension would you like to enable?\n\u2502  \u25cb Developer Tools \n\u2502  \u25cb Computer Controller \n// highlight-start  \n\u2502  \u25cf Tutorial\n// highlight-end\n\u2502  \u25cb Memory \n|  \u25cb JetBrains\n\u2514 \n"})}),"\n",(0,o.jsxs)(n.ol,{start:"4",children:["\n",(0,o.jsx)(n.li,{children:"Enter the number of seconds Goose should wait for actions to complete before timing out. Default is 300s"}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"   \u250c   goose-configure \n   \u2502\n   \u25c7  What would you like to configure?\n   \u2502  Add Extension \n   \u2502\n   \u25c7  What type of extension would you like to add?\n   \u2502  Built-in Extension \n   \u2502\n   \u25c7  Which built-in extension would you like to enable?\n   \u2502  Tutorial\n   \u2502\n   // highlight-start\n   \u25c6  Please set the timeout for this tool (in secs):\n   \u2502  300\n   // highlight-end\n   \u2514  Enabled Tutorials extension\n"})}),"\n",(0,o.jsx)(n.h2,{id:"available-tutorials",children:"Available Tutorials"}),"\n",(0,o.jsx)(n.p,{children:"Currently, the following tutorials are available:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"build-mcp-extension"}),": Learn how to build an extension using the Model Context Protocol (MCP)"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"first-game"}),": Create your first game with Goose"]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:"More tutorials are being added regularly to cover additional features and use cases."}),"\n",(0,o.jsx)(n.h2,{id:"using-the-tutorial-extension",children:"Using the Tutorial Extension"}),"\n",(0,o.jsx)(n.h3,{id:"starting-a-tutorial",children:"Starting a Tutorial"}),"\n",(0,o.jsx)(n.p,{children:"There are two main ways to interact with tutorials:"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Ask Goose directly"}),": You can simply ask about a topic you're interested in, and Goose will suggest relevant tutorials."]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"( O)> I'd like to learn how to build an extension for Goose\n"})}),"\n",(0,o.jsxs)(n.ol,{start:"2",children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Request a specific tutorial"}),": You can directly request a tutorial by name."]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"( O)> Can you walk me through the first-game tutorial?\n"})}),"\n",(0,o.jsx)(n.h3,{id:"during-the-tutorial",children:"During the Tutorial"}),"\n",(0,o.jsx)(n.p,{children:"While going through a tutorial:"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsx)(n.li,{children:"Goose will guide you step-by-step through the process"}),"\n",(0,o.jsx)(n.li,{children:"You'll receive clear instructions before any actions are taken"}),"\n",(0,o.jsx)(n.li,{children:"You can ask questions at any time for clarification"}),"\n",(0,o.jsx)(n.li,{children:"You can take breaks and resume later"}),"\n"]}),"\n",(0,o.jsx)(n.h3,{id:"best-practices",children:"Best Practices"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Take your time"}),": Don't rush through the tutorials. Make sure you understand each step before moving on."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Ask questions"}),": If something isn't clear, ask Goose to explain it further."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Experiment"}),": Feel free to try variations of what you're learning."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Take notes"}),": Document any interesting findings or techniques you learn."]}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,o.jsx)(n.h3,{id:"starting-a-tutorial-session",children:"Starting a Tutorial Session"}),"\n",(0,o.jsx)(n.p,{children:"Here's an example of how to start learning about building extensions:"}),"\n",(0,o.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"I'd like to learn how to build an extension for Goose\n"})}),"\n",(0,o.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"I'll help you learn about building extensions! The \"build-mcp-extension\" tutorial would be perfect for this. Let me load it and guide you through it.\n\n\u2500\u2500\u2500 load_tutorial | tutorial \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nname: build-mcp-extension\n\n\n\nBefore we dive into building your extension, I need to understand a few things to better guide you:\n\n1. Which SDK would you prefer to use? We have three options:\n   - Python (beginner-friendly, great for prototyping)\n   - TypeScript (great for web-based tools)\n   - Kotlin (robust, type-safe, good for complex applications)\n\n2. What type of extension would you like to build? For example:\n   - A tool that performs specific actions\n   - A resource that provides data to the LLM\n   - Or do you have a specific idea in mind?\n\nLet me know your preferences, and I'll help you get started with the appropriate setup and implementation!\n\nFor reference, here's what we'll cover:\n1. Setting up your development environment\n2. Creating the basic MCP server\n3. Implementing your specific functionality\n4. Testing and debugging\n\nWhich SDK would you like to use to get started?\n"})}),"\n",(0,o.jsx)(n.h2,{id:"next-steps",children:"Next Steps"}),"\n",(0,o.jsx)(n.p,{children:"After completing tutorials, you can:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"Apply learned concepts to your own projects"}),"\n",(0,o.jsx)(n.li,{children:"Contribute to the Goose community by sharing your experiences"}),"\n",(0,o.jsx)(n.li,{children:"Suggest new tutorial topics that would be helpful for others"}),"\n"]}),"\n",(0,o.jsx)(n.h3,{id:"need-further-help",children:"Need Further Help?"}),"\n",(0,o.jsxs)(n.p,{children:["If you have questions, run into issues, or just need to brainstorm ideas join the ",(0,o.jsx)(n.a,{href:"https://discord.gg/block-opensource",children:"Discord Community"}),"!"]})]})}function c(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}},8453:(e,n,t)=>{t.d(n,{R:()=>r,x:()=>l});var i=t(6540);const o={},s=i.createContext(o);function r(e){const n=i.useContext(s);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:r(e.components),i.createElement(s.Provider,{value:n},e.children)}}}]);