"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[1783],{9510:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>u,default:()=>p,frontMatter:()=>a,metadata:()=>i,toc:()=>d});const i=JSON.parse('{"id":"tutorials/tutorial-extension","title":"Tutorial Extension","description":"Learn how to use Goose\'s built-in Tutorial extension for guided learning","source":"@site/docs/tutorials/tutorial-extension.md","sourceDirName":"tutorials","slug":"/tutorials/tutorial-extension","permalink":"/goose/pr-preview/pr-1722/docs/tutorials/tutorial-extension","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Tutorial Extension","description":"Learn how to use Goose\'s built-in Tutorial extension for guided learning"},"sidebar":"tutorialSidebar","previous":{"title":"Tavily Web Search Extension","permalink":"/goose/pr-preview/pr-1722/docs/tutorials/tavily-mcp"},"next":{"title":"Architecture Overview","permalink":"/goose/pr-preview/pr-1722/docs/category/architecture-overview"}}');var o=t(4848),r=t(8453),s=t(5537),l=t(9329);const a={title:"Tutorial Extension",description:"Learn how to use Goose's built-in Tutorial extension for guided learning"},u=void 0,c={},d=[{value:"Configuration",id:"configuration",level:2},{value:"Available Tutorials",id:"available-tutorials",level:2},{value:"Using the Tutorial Extension",id:"using-the-tutorial-extension",level:2},{value:"Starting a Tutorial",id:"starting-a-tutorial",level:3},{value:"During the Tutorial",id:"during-the-tutorial",level:3},{value:"Best Practices",id:"best-practices",level:3},{value:"Example Usage",id:"example-usage",level:2},{value:"Starting a Tutorial Session",id:"starting-a-tutorial-session",level:3},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3},{value:"Next Steps",id:"next-steps",level:2},{value:"Need Further Help?",id:"need-further-help",level:3}];function h(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.p,{children:"The Tutorial extension is a built-in feature of Goose that provides interactive, step-by-step guidance for learning various aspects of Goose and its capabilities. It's designed to help users get comfortable with Goose's features through hands-on practice."}),"\n",(0,o.jsx)(n.p,{children:"The Tutorial extension serves as an interactive learning tool that:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"Provides structured, step-by-step tutorials"}),"\n",(0,o.jsx)(n.li,{children:"Allows hands-on practice with Goose features"}),"\n",(0,o.jsx)(n.li,{children:"Offers immediate feedback and guidance"}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsx)(n.li,{children:"Ensure the Tutorial extension is enabled:"}),"\n"]}),"\n",(0,o.jsxs)(s.A,{groupId:"interface",children:[(0,o.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,o.jsxs)(n.ol,{start:"2",children:["\n",(0,o.jsxs)(n.li,{children:["Choose to add a ",(0,o.jsx)(n.code,{children:"Built-in Extension"})]}),"\n"]}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c6  What type of extension would you like to add?\n// highlight-start  \n\u2502  \u25cf Built-in Extension (Use an extension that comes with Goose)\n// highlight-end  \n\u2502  \u25cb Command-line Extension \n\u2502  \u25cb Remote Extension \n\u2514  \n"})}),(0,o.jsxs)(n.ol,{start:"3",children:["\n",(0,o.jsxs)(n.li,{children:["Select the ",(0,o.jsx)(n.code,{children:"Tutorial"})," extension"]}),"\n"]}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Add Extension \n\u2502\n\u25c7  What type of extension would you like to add?\n\u2502  Built-in Extension \n\u2502\n\u25c6  Which built-in extension would you like to enable?\n\u2502  \u25cb Developer Tools \n\u2502  \u25cb Computer Controller \n// highlight-start  \n\u2502  \u25cf Tutorial\n// highlight-end\n\u2502  \u25cb Memory \n|  \u25cb JetBrains\n\u2514 \n"})}),(0,o.jsxs)(n.ol,{start:"4",children:["\n",(0,o.jsx)(n.li,{children:"Enter the number of seconds Goose should wait for actions to complete before timing out. Default is 300s"}),"\n"]}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-sh",children:"   \u250c   goose-configure \n   \u2502\n   \u25c7  What would you like to configure?\n   \u2502  Add Extension \n   \u2502\n   \u25c7  What type of extension would you like to add?\n   \u2502  Built-in Extension \n   \u2502\n   \u25c7  Which built-in extension would you like to enable?\n   \u2502  Tutorial\n   \u2502\n   // highlight-start\n   \u25c6  Please set the timeout for this tool (in secs):\n   \u2502  300\n   // highlight-end\n   \u2514  Enabled Tutorials extension\n"})})]}),(0,o.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsxs)(n.li,{children:["Click ",(0,o.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,o.jsxs)(n.li,{children:["Click ",(0,o.jsx)(n.code,{children:"Settings"})]}),"\n",(0,o.jsxs)(n.li,{children:["Under ",(0,o.jsx)(n.code,{children:"Extensions"}),", toggle ",(0,o.jsx)(n.code,{children:"Tutorial"})," to on."]}),"\n"]})})]}),"\n",(0,o.jsx)(n.h2,{id:"available-tutorials",children:"Available Tutorials"}),"\n",(0,o.jsx)(n.p,{children:"Currently, the following tutorials are available:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"build-mcp-extension"}),": Learn how to build an extension using the Model Context Protocol (MCP)"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"first-game"}),": Create your first game with Goose"]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:"More tutorials are being added regularly to cover additional features and use cases."}),"\n",(0,o.jsx)(n.h2,{id:"using-the-tutorial-extension",children:"Using the Tutorial Extension"}),"\n",(0,o.jsx)(n.h3,{id:"starting-a-tutorial",children:"Starting a Tutorial"}),"\n",(0,o.jsx)(n.p,{children:"There are two main ways to interact with tutorials:"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Ask Goose directly"}),": You can simply ask about a topic you're interested in, and Goose will suggest relevant tutorials."]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"( O)> I'd like to learn how to build an extension for Goose\n"})}),"\n",(0,o.jsxs)(n.ol,{start:"2",children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Request a specific tutorial"}),": You can directly request a tutorial by name."]}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"( O)> Can you walk me through the first-game tutorial?\n"})}),"\n",(0,o.jsx)(n.h3,{id:"during-the-tutorial",children:"During the Tutorial"}),"\n",(0,o.jsx)(n.p,{children:"While going through a tutorial:"}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsx)(n.li,{children:"Goose will guide you step-by-step through the process"}),"\n",(0,o.jsx)(n.li,{children:"You'll receive clear instructions before any actions are taken"}),"\n",(0,o.jsx)(n.li,{children:"You can ask questions at any time for clarification"}),"\n",(0,o.jsx)(n.li,{children:"You can take breaks and resume later"}),"\n"]}),"\n",(0,o.jsx)(n.h3,{id:"best-practices",children:"Best Practices"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Take your time"}),": Don't rush through the tutorials. Make sure you understand each step before moving on."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Ask questions"}),": If something isn't clear, ask Goose to explain it further."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Experiment"}),": Feel free to try variations of what you're learning."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Take notes"}),": Document any interesting findings or techniques you learn."]}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,o.jsx)(n.h3,{id:"starting-a-tutorial-session",children:"Starting a Tutorial Session"}),"\n",(0,o.jsx)(n.p,{children:"Here's an example of how to start learning about building extensions:"}),"\n",(0,o.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"I'd like to learn how to build an extension for Goose\n"})}),"\n",(0,o.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{children:"I'll help you learn about building extensions! The \"build-mcp-extension\" tutorial would be perfect for this. Let me load it and guide you through it.\n\n\u2500\u2500\u2500 load_tutorial | tutorial \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nname: build-mcp-extension\n\n\n\nBefore we dive into building your extension, I need to understand a few things to better guide you:\n\n1. Which SDK would you prefer to use? We have three options:\n   - Python (beginner-friendly, great for prototyping)\n   - TypeScript (great for web-based tools)\n   - Kotlin (robust, type-safe, good for complex applications)\n\n2. What type of extension would you like to build? For example:\n   - A tool that performs specific actions\n   - A resource that provides data to the LLM\n   - Or do you have a specific idea in mind?\n\nLet me know your preferences, and I'll help you get started with the appropriate setup and implementation!\n\nFor reference, here's what we'll cover:\n1. Setting up your development environment\n2. Creating the basic MCP server\n3. Implementing your specific functionality\n4. Testing and debugging\n\nWhich SDK would you like to use to get started?\n"})}),"\n",(0,o.jsx)(n.h2,{id:"next-steps",children:"Next Steps"}),"\n",(0,o.jsx)(n.p,{children:"After completing tutorials, you can:"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"Apply learned concepts to your own projects"}),"\n",(0,o.jsx)(n.li,{children:"Contribute to the Goose community by sharing your experiences"}),"\n",(0,o.jsx)(n.li,{children:"Suggest new tutorial topics that would be helpful for others"}),"\n"]}),"\n",(0,o.jsx)(n.h3,{id:"need-further-help",children:"Need Further Help?"}),"\n",(0,o.jsxs)(n.p,{children:["If you have questions, run into issues, or just need to brainstorm ideas join the ",(0,o.jsx)(n.a,{href:"https://discord.gg/block-opensource",children:"Discord Community"}),"!"]})]})}function p(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(h,{...e})}):h(e)}},9329:(e,n,t)=>{t.d(n,{A:()=>s});t(6540);var i=t(4164);const o={tabItem:"tabItem_Ymn6"};var r=t(4848);function s(e){let{children:n,hidden:t,className:s}=e;return(0,r.jsx)("div",{role:"tabpanel",className:(0,i.A)(o.tabItem,s),hidden:t,children:n})}},5537:(e,n,t)=>{t.d(n,{A:()=>w});var i=t(6540),o=t(4164),r=t(5627),s=t(6347),l=t(372),a=t(604),u=t(1861),c=t(8749);function d(e){return i.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,i.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:t}=e;return(0,i.useMemo)((()=>{const e=n??function(e){return d(e).map((e=>{let{props:{value:n,label:t,attributes:i,default:o}}=e;return{value:n,label:t,attributes:i,default:o}}))}(t);return function(e){const n=(0,u.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,t])}function p(e){let{value:n,tabValues:t}=e;return t.some((e=>e.value===n))}function x(e){let{queryString:n=!1,groupId:t}=e;const o=(0,s.W6)(),r=function(e){let{queryString:n=!1,groupId:t}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!t)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return t??null}({queryString:n,groupId:t});return[(0,a.aZ)(r),(0,i.useCallback)((e=>{if(!r)return;const n=new URLSearchParams(o.location.search);n.set(r,e),o.replace({...o.location,search:n.toString()})}),[r,o])]}function g(e){const{defaultValue:n,queryString:t=!1,groupId:o}=e,r=h(e),[s,a]=(0,i.useState)((()=>function(e){let{defaultValue:n,tabValues:t}=e;if(0===t.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!p({value:n,tabValues:t}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${t.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const i=t.find((e=>e.default))??t[0];if(!i)throw new Error("Unexpected error: 0 tabValues");return i.value}({defaultValue:n,tabValues:r}))),[u,d]=x({queryString:t,groupId:o}),[g,f]=function(e){let{groupId:n}=e;const t=function(e){return e?`docusaurus.tab.${e}`:null}(n),[o,r]=(0,c.Dv)(t);return[o,(0,i.useCallback)((e=>{t&&r.set(e)}),[t,r])]}({groupId:o}),m=(()=>{const e=u??g;return p({value:e,tabValues:r})?e:null})();(0,l.A)((()=>{m&&a(m)}),[m]);return{selectedValue:s,selectValue:(0,i.useCallback)((e=>{if(!p({value:e,tabValues:r}))throw new Error(`Can't select invalid tab value=${e}`);a(e),d(e),f(e)}),[d,f,r]),tabValues:r}}var f=t(9136);const m={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var b=t(4848);function v(e){let{className:n,block:t,selectedValue:i,selectValue:s,tabValues:l}=e;const a=[],{blockElementScrollPositionUntilNextRender:u}=(0,r.a_)(),c=e=>{const n=e.currentTarget,t=a.indexOf(n),o=l[t].value;o!==i&&(u(n),s(o))},d=e=>{let n=null;switch(e.key){case"Enter":c(e);break;case"ArrowRight":{const t=a.indexOf(e.currentTarget)+1;n=a[t]??a[0];break}case"ArrowLeft":{const t=a.indexOf(e.currentTarget)-1;n=a[t]??a[a.length-1];break}}n?.focus()};return(0,b.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,o.A)("tabs",{"tabs--block":t},n),children:l.map((e=>{let{value:n,label:t,attributes:r}=e;return(0,b.jsx)("li",{role:"tab",tabIndex:i===n?0:-1,"aria-selected":i===n,ref:e=>{a.push(e)},onKeyDown:d,onClick:c,...r,className:(0,o.A)("tabs__item",m.tabItem,r?.className,{"tabs__item--active":i===n}),children:t??n},n)}))})}function j(e){let{lazy:n,children:t,selectedValue:r}=e;const s=(Array.isArray(t)?t:[t]).filter(Boolean);if(n){const e=s.find((e=>e.props.value===r));return e?(0,i.cloneElement)(e,{className:(0,o.A)("margin-top--md",e.props.className)}):null}return(0,b.jsx)("div",{className:"margin-top--md",children:s.map(((e,n)=>(0,i.cloneElement)(e,{key:n,hidden:e.props.value!==r})))})}function y(e){const n=g(e);return(0,b.jsxs)("div",{className:(0,o.A)("tabs-container",m.tabList),children:[(0,b.jsx)(v,{...n,...e}),(0,b.jsx)(j,{...n,...e})]})}function w(e){const n=(0,f.A)();return(0,b.jsx)(y,{...e,children:d(e.children)},String(n))}},8453:(e,n,t)=>{t.d(n,{R:()=>s,x:()=>l});var i=t(6540);const o={},r=i.createContext(o);function s(e){const n=i.useContext(r);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:s(e.components),i.createElement(r.Provider,{value:n},e.children)}}}]);