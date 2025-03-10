"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[701],{8876:(e,s,n)=>{n.r(s),n.d(s,{assets:()=>d,contentTitle:()=>c,default:()=>p,frontMatter:()=>l,metadata:()=>o,toc:()=>u});const o=JSON.parse('{"id":"guides/managing-goose-sessions","title":"Managing Sessions","description":"A session is a single, continuous interaction between you and Goose, providing a space to ask questions and prompt action. In this guide, we\'ll cover how to start, exit, and resume a session.","source":"@site/docs/guides/managing-goose-sessions.md","sourceDirName":"guides","slug":"/guides/managing-goose-sessions","permalink":"/goose/pr-preview/pr-1599/docs/guides/managing-goose-sessions","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1,"title":"Managing Sessions"},"sidebar":"tutorialSidebar","previous":{"title":"Guides","permalink":"/goose/pr-preview/pr-1599/docs/category/guides"},"next":{"title":"Updating Goose","permalink":"/goose/pr-preview/pr-1599/docs/guides/updating-goose"}}');var i=n(4848),t=n(8453),r=n(5537),a=n(9329);const l={sidebar_position:1,title:"Managing Sessions"},c="Managing Goose Sessions",d={},u=[{value:"Start Session",id:"start-session",level:2},{value:"Name Session",id:"name-session",level:2},{value:"Exit Session",id:"exit-session",level:2},{value:"Resume Session",id:"resume-session",level:2},{value:"Resume Session Across Interfaces",id:"resume-session-across-interfaces",level:3}];function h(e){const s={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,t.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(s.header,{children:(0,i.jsx)(s.h1,{id:"managing-goose-sessions",children:"Managing Goose Sessions"})}),"\n",(0,i.jsx)(s.p,{children:"A session is a single, continuous interaction between you and Goose, providing a space to ask questions and prompt action. In this guide, we'll cover how to start, exit, and resume a session."}),"\n",(0,i.jsx)(s.h2,{id:"start-session",children:"Start Session"}),"\n",(0,i.jsxs)(r.A,{children:[(0,i.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,i.jsx)(s.p,{children:"From your terminal, navigate to the directory from which you'd like to start, and run:"}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{className:"language-sh",children:"goose session \n"})})]}),(0,i.jsxs)(a.A,{value:"ui",label:"Goose Desktop",children:[(0,i.jsx)(s.p,{children:"After choosing an LLM provider, you\u2019ll see the session interface ready for use. Type your questions, tasks, or instructions directly into the input field, and Goose will immediately get to work."}),(0,i.jsxs)(s.p,{children:["To start a new session at any time, click the three dots in the top-right corner of the application and select ",(0,i.jsx)(s.strong,{children:"New Session"})," from the dropdown menu."]})]})]}),"\n",(0,i.jsx)(s.admonition,{type:"info",children:(0,i.jsxs)(s.p,{children:["If this is your first session, Goose will prompt you for an API key to access an LLM (Large Language Model) of your choice. For more information on setting up your API key, see the ",(0,i.jsx)(s.a,{href:"/docs/getting-started/installation#set-llm-provider",children:"Installation Guide"}),". Here is the list of ",(0,i.jsx)(s.a,{href:"/docs/getting-started/providers",children:"supported LLMs"}),"."]})}),"\n",(0,i.jsx)(s.h2,{id:"name-session",children:"Name Session"}),"\n",(0,i.jsxs)(r.A,{children:[(0,i.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,i.jsxs)(s.p,{children:["By default, Goose names your session using the current timestamp in the format ",(0,i.jsx)(s.code,{children:"YYYYMMDD_HHMMSS"}),". If you'd like to provide a specific name, this is where you'd do so. For example to name your session ",(0,i.jsx)(s.code,{children:"react-migration"}),", you would run:"]}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"goose session -n react-migration\n"})}),(0,i.jsx)(s.p,{children:"You'll know your session has started when your terminal looks similar to the following:"}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"starting session | provider: openai model: gpt-4o\nlogging to ~/.local/share/goose/sessions/react-migration.json1\n"})})]}),(0,i.jsx)(a.A,{value:"ui",label:"Goose Desktop",children:(0,i.jsxs)(s.p,{children:["Within the Desktop app, sessions are automatically named using the current timestamp in the format ",(0,i.jsx)(s.code,{children:"YYYYMMDD_HHMMSS"}),". Goose also provides a description of the session based on context."]})})]}),"\n",(0,i.jsx)(s.h2,{id:"exit-session",children:"Exit Session"}),"\n",(0,i.jsx)(s.p,{children:"Note that sessions are automatically saved when you exit."}),"\n",(0,i.jsxs)(r.A,{children:[(0,i.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,i.jsxs)(s.p,{children:["To exit a session, type ",(0,i.jsx)(s.code,{children:"exit"}),". Alternatively, you exit the session by holding down ",(0,i.jsx)(s.code,{children:"Ctrl+C"}),"."]}),(0,i.jsxs)(s.p,{children:["Your session will be stored locally in ",(0,i.jsx)(s.code,{children:"~/.local/share/goose/sessions"}),"."]})]}),(0,i.jsx)(a.A,{value:"ui",label:"Goose Desktop",children:(0,i.jsx)(s.p,{children:"To exit a session, simply close the application."})})]}),"\n",(0,i.jsx)(s.h2,{id:"resume-session",children:"Resume Session"}),"\n",(0,i.jsxs)(r.A,{children:[(0,i.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,i.jsx)(s.p,{children:"To resume your latest session, you can run the following command:"}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:" goose session -r\n"})}),(0,i.jsx)(s.p,{children:"To resume a specific session, run the following command:"}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"goose session -r --name <name>\n"})}),(0,i.jsxs)(s.p,{children:["For example, to resume the session named ",(0,i.jsx)(s.code,{children:"react-migration"}),", you would run:"]}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{children:"goose session -r --name react-migration\n"})}),(0,i.jsx)(s.admonition,{type:"tip",children:(0,i.jsxs)(s.p,{children:["While you can resume sessions using the commands above, we recommend creating new sessions for new tasks to reduce the chance of ",(0,i.jsx)(s.a,{href:"/docs/troubleshooting#stuck-in-a-loop-or-unresponsive",children:"doom spiraling"}),"."]})})]}),(0,i.jsx)(a.A,{value:"ui",label:"Goose Desktop",children:(0,i.jsxs)(s.ol,{children:["\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"..."})," in the upper right corner"]}),"\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"Previous Sessions"})]}),"\n",(0,i.jsx)(s.li,{children:"Click a session"}),"\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"Resume Session"})," in the upper right corner"]}),"\n"]})})]}),"\n",(0,i.jsx)(s.h3,{id:"resume-session-across-interfaces",children:"Resume Session Across Interfaces"}),"\n",(0,i.jsx)(s.p,{children:"You can resume a CLI session in Desktop and vice versa."}),"\n",(0,i.jsxs)(r.A,{children:[(0,i.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,i.jsxs)(s.p,{children:["To resume a Desktop session within CLI, get the name of the session from the Desktop app. Note that unless you specifically named the session, its default name is a timestamp in the format ",(0,i.jsx)(s.code,{children:"YYYYMMDD_HHMMSS"}),"."]}),(0,i.jsxs)(s.ol,{children:["\n",(0,i.jsx)(s.li,{children:"Open Goose Desktop"}),"\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"..."})," in the upper right corner"]}),"\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"Previous Sessions"})]}),"\n",(0,i.jsxs)(s.li,{children:["Find the session that you want to resume, and copy the basename (without the ",(0,i.jsx)(s.code,{children:".jsonl"})," extension)."]}),"\n"]}),(0,i.jsxs)(s.admonition,{title:"Example",type:"note",children:[(0,i.jsx)(s.p,{children:(0,i.jsx)(s.strong,{children:"Desktop Session"})}),(0,i.jsxs)(s.table,{children:[(0,i.jsx)(s.thead,{children:(0,i.jsxs)(s.tr,{children:[(0,i.jsx)(s.th,{children:"Session Description"}),(0,i.jsx)(s.th,{children:"Session Filename"})]})}),(0,i.jsx)(s.tbody,{children:(0,i.jsxs)(s.tr,{children:[(0,i.jsx)(s.td,{children:"GitHub PR Access Issue"}),(0,i.jsxs)(s.td,{children:[(0,i.jsx)(s.strong,{children:"20250305_113223"}),".jsonl"]})]})})]}),(0,i.jsx)(s.p,{children:(0,i.jsx)(s.strong,{children:"CLI Command"})}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{className:"language-sh",children:"goose session -r --name 20250305_113223\n"})})]})]}),(0,i.jsxs)(a.A,{value:"ui",label:"Goose Desktop",children:[(0,i.jsx)(s.p,{children:"All saved sessions are listed in the Desktop app, even CLI sessions. To resume a CLI session within the Desktop:"}),(0,i.jsxs)(s.ol,{children:["\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"..."})," in the upper right corner"]}),"\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"Previous Sessions"})]}),"\n",(0,i.jsx)(s.li,{children:"Click the session you'd like to resume"}),"\n"]}),(0,i.jsx)(s.admonition,{type:"tip",children:(0,i.jsx)(s.p,{children:"If you named the session, you'll recognize the filename. However, if you don't remember the exact session name, there is a description of the topic."})}),(0,i.jsxs)(s.ol,{start:"4",children:["\n",(0,i.jsxs)(s.li,{children:["Click ",(0,i.jsx)(s.code,{children:"Resume Session"})," in the upper right corner"]}),"\n"]}),(0,i.jsxs)(s.admonition,{title:"Example",type:"note",children:[(0,i.jsx)(s.p,{children:(0,i.jsx)(s.strong,{children:"CLI Command"})}),(0,i.jsx)(s.pre,{children:(0,i.jsx)(s.code,{className:"language-sh",children:"goose session -n react-migration\n"})}),(0,i.jsx)(s.p,{children:(0,i.jsx)(s.strong,{children:"Desktop Session"})}),(0,i.jsxs)(s.table,{children:[(0,i.jsx)(s.thead,{children:(0,i.jsxs)(s.tr,{children:[(0,i.jsx)(s.th,{children:"Session Description"}),(0,i.jsx)(s.th,{children:"Session Filename"})]})}),(0,i.jsx)(s.tbody,{children:(0,i.jsxs)(s.tr,{children:[(0,i.jsx)(s.td,{children:"Code Migration to React"}),(0,i.jsxs)(s.td,{children:[(0,i.jsx)(s.strong,{children:"react-migration"}),".jsonl"]})]})})]})]})]})]})]})}function p(e={}){const{wrapper:s}={...(0,t.R)(),...e.components};return s?(0,i.jsx)(s,{...e,children:(0,i.jsx)(h,{...e})}):h(e)}},9329:(e,s,n)=>{n.d(s,{A:()=>r});n(6540);var o=n(4164);const i={tabItem:"tabItem_Ymn6"};var t=n(4848);function r(e){let{children:s,hidden:n,className:r}=e;return(0,t.jsx)("div",{role:"tabpanel",className:(0,o.A)(i.tabItem,r),hidden:n,children:s})}},5537:(e,s,n)=>{n.d(s,{A:()=>w});var o=n(6540),i=n(4164),t=n(5627),r=n(6347),a=n(372),l=n(604),c=n(1861),d=n(8749);function u(e){return o.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,o.isValidElement)(e)&&function(e){const{props:s}=e;return!!s&&"object"==typeof s&&"value"in s}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:s,children:n}=e;return(0,o.useMemo)((()=>{const e=s??function(e){return u(e).map((e=>{let{props:{value:s,label:n,attributes:o,default:i}}=e;return{value:s,label:n,attributes:o,default:i}}))}(n);return function(e){const s=(0,c.XI)(e,((e,s)=>e.value===s.value));if(s.length>0)throw new Error(`Docusaurus error: Duplicate values "${s.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[s,n])}function p(e){let{value:s,tabValues:n}=e;return n.some((e=>e.value===s))}function m(e){let{queryString:s=!1,groupId:n}=e;const i=(0,r.W6)(),t=function(e){let{queryString:s=!1,groupId:n}=e;if("string"==typeof s)return s;if(!1===s)return null;if(!0===s&&!n)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return n??null}({queryString:s,groupId:n});return[(0,l.aZ)(t),(0,o.useCallback)((e=>{if(!t)return;const s=new URLSearchParams(i.location.search);s.set(t,e),i.replace({...i.location,search:s.toString()})}),[t,i])]}function x(e){const{defaultValue:s,queryString:n=!1,groupId:i}=e,t=h(e),[r,l]=(0,o.useState)((()=>function(e){let{defaultValue:s,tabValues:n}=e;if(0===n.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(s){if(!p({value:s,tabValues:n}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${s}" but none of its children has the corresponding value. Available values are: ${n.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return s}const o=n.find((e=>e.default))??n[0];if(!o)throw new Error("Unexpected error: 0 tabValues");return o.value}({defaultValue:s,tabValues:t}))),[c,u]=m({queryString:n,groupId:i}),[x,j]=function(e){let{groupId:s}=e;const n=function(e){return e?`docusaurus.tab.${e}`:null}(s),[i,t]=(0,d.Dv)(n);return[i,(0,o.useCallback)((e=>{n&&t.set(e)}),[n,t])]}({groupId:i}),g=(()=>{const e=c??x;return p({value:e,tabValues:t})?e:null})();(0,a.A)((()=>{g&&l(g)}),[g]);return{selectedValue:r,selectValue:(0,o.useCallback)((e=>{if(!p({value:e,tabValues:t}))throw new Error(`Can't select invalid tab value=${e}`);l(e),u(e),j(e)}),[u,j,t]),tabValues:t}}var j=n(9136);const g={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var f=n(4848);function b(e){let{className:s,block:n,selectedValue:o,selectValue:r,tabValues:a}=e;const l=[],{blockElementScrollPositionUntilNextRender:c}=(0,t.a_)(),d=e=>{const s=e.currentTarget,n=l.indexOf(s),i=a[n].value;i!==o&&(c(s),r(i))},u=e=>{let s=null;switch(e.key){case"Enter":d(e);break;case"ArrowRight":{const n=l.indexOf(e.currentTarget)+1;s=l[n]??l[0];break}case"ArrowLeft":{const n=l.indexOf(e.currentTarget)-1;s=l[n]??l[l.length-1];break}}s?.focus()};return(0,f.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,i.A)("tabs",{"tabs--block":n},s),children:a.map((e=>{let{value:s,label:n,attributes:t}=e;return(0,f.jsx)("li",{role:"tab",tabIndex:o===s?0:-1,"aria-selected":o===s,ref:e=>{l.push(e)},onKeyDown:u,onClick:d,...t,className:(0,i.A)("tabs__item",g.tabItem,t?.className,{"tabs__item--active":o===s}),children:n??s},s)}))})}function v(e){let{lazy:s,children:n,selectedValue:t}=e;const r=(Array.isArray(n)?n:[n]).filter(Boolean);if(s){const e=r.find((e=>e.props.value===t));return e?(0,o.cloneElement)(e,{className:(0,i.A)("margin-top--md",e.props.className)}):null}return(0,f.jsx)("div",{className:"margin-top--md",children:r.map(((e,s)=>(0,o.cloneElement)(e,{key:s,hidden:e.props.value!==t})))})}function y(e){const s=x(e);return(0,f.jsxs)("div",{className:(0,i.A)("tabs-container",g.tabList),children:[(0,f.jsx)(b,{...s,...e}),(0,f.jsx)(v,{...s,...e})]})}function w(e){const s=(0,j.A)();return(0,f.jsx)(y,{...e,children:u(e.children)},String(s))}},8453:(e,s,n)=>{n.d(s,{R:()=>r,x:()=>a});var o=n(6540);const i={},t=o.createContext(i);function r(e){const s=o.useContext(t);return o.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function a(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),o.createElement(t.Provider,{value:s},e.children)}}}]);