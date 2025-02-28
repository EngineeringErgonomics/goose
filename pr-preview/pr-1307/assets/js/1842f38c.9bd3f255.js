"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6568],{2364:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>d,contentTitle:()=>u,default:()=>g,frontMatter:()=>l,metadata:()=>s,toc:()=>c});const s=JSON.parse('{"id":"guides/goose-permissions","title":"Goose Permissions","description":"Goose\u2019s permissions determine how much autonomy it has when modifying files, using extensions, and performing automated actions. By selecting a permission mode, you have full control over how Goose interacts with your development environment.","source":"@site/docs/guides/goose-permissions.md","sourceDirName":"guides","slug":"/guides/goose-permissions","permalink":"/goose/pr-preview/pr-1307/docs/guides/goose-permissions","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":3,"frontMatter":{"sidebar_position":3,"title":"Goose Permissions"},"sidebar":"tutorialSidebar","previous":{"title":"Updating Goose","permalink":"/goose/pr-preview/pr-1307/docs/guides/updating-goose"},"next":{"title":"Using Goosehints","permalink":"/goose/pr-preview/pr-1307/docs/guides/using-goosehints"}}');var t=o(4848),r=o(8453),i=o(5537),a=o(9329);const l={sidebar_position:3,title:"Goose Permissions"},u="Managing Goose Permissions",d={},c=[{value:"Permission Modes",id:"permission-modes",level:2},{value:"Configuring Goose Permissions",id:"configuring-goose-permissions",level:2}];function h(e){const n={admonition:"admonition",code:"code",h1:"h1",h2:"h2",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",table:"table",tbody:"tbody",td:"td",th:"th",thead:"thead",tr:"tr",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.header,{children:(0,t.jsx)(n.h1,{id:"managing-goose-permissions",children:"Managing Goose Permissions"})}),"\n",(0,t.jsxs)(n.p,{children:["Goose\u2019s ",(0,t.jsx)(n.strong,{children:"permissions"})," determine how much autonomy it has when modifying files, using extensions, and performing automated actions. By selecting a permission mode, you have full control over how Goose interacts with your development environment."]}),"\n",(0,t.jsx)(n.h2,{id:"permission-modes",children:"Permission Modes"}),"\n",(0,t.jsxs)(n.table,{children:[(0,t.jsx)(n.thead,{children:(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.th,{children:"Mode"}),(0,t.jsx)(n.th,{children:"Description"}),(0,t.jsx)(n.th,{children:"Best For"})]})}),(0,t.jsxs)(n.tbody,{children:[(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:(0,t.jsx)(n.strong,{children:"Auto Mode"})}),(0,t.jsxs)(n.td,{children:["Goose can modify files, use extensions, and delete files ",(0,t.jsx)(n.strong,{children:"without requiring approval"}),"."]}),(0,t.jsxs)(n.td,{children:["Users who want ",(0,t.jsx)(n.strong,{children:"full automation"})," and seamless integration into their workflow."]})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:(0,t.jsx)(n.strong,{children:"Approve Mode"})}),(0,t.jsxs)(n.td,{children:["Goose ",(0,t.jsx)(n.strong,{children:"asks for confirmation"})," before modifying, creating, deleting files and before using extensions."]}),(0,t.jsxs)(n.td,{children:["Users who want to ",(0,t.jsx)(n.strong,{children:"review and approve"})," changes and extension use before they happen."]})]}),(0,t.jsxs)(n.tr,{children:[(0,t.jsx)(n.td,{children:(0,t.jsx)(n.strong,{children:"Chat Mode"})}),(0,t.jsxs)(n.td,{children:["Goose ",(0,t.jsx)(n.strong,{children:"only engages in chat"}),", with no extension use or file modifications."]}),(0,t.jsxs)(n.td,{children:["Users who prefer a ",(0,t.jsx)(n.strong,{children:"conversational AI experience"})," without automation."]})]})]})]}),"\n",(0,t.jsx)(n.admonition,{type:"warning",children:(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.code,{children:"Auto Mode"})," is applied by default unless you specify otherwise."]})}),"\n",(0,t.jsx)(n.h2,{id:"configuring-goose-permissions",children:"Configuring Goose Permissions"}),"\n",(0,t.jsx)(n.p,{children:"Here's how to configure your chosen goose permissions:"}),"\n",(0,t.jsxs)(i.A,{groupId:"interface",children:[(0,t.jsxs)(a.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Run the following command:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Goose Settings"})," from the menu and press Enter."]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c6  What would you like to configure?\n| \u25cb Configure Providers \n| \u25cb Add Extension \n| \u25cb Toggle Extensions \n| \u25cb Remove Extension \n// highlight-start  \n| \u25cf Goose Settings (Set the Goose Mode, Tool Output, Experiment and more)\n// highlight-end\n\u2514  \n"})}),(0,t.jsxs)(n.ol,{start:"3",children:["\n",(0,t.jsxs)(n.li,{children:["Choose ",(0,t.jsx)(n.code,{children:"Goose Mode"})," from the menu and press Enter."]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Goose Settings \n\u2502\n\u25c6  What setting would you like to configure?\n// highlight-start\n\u2502  \u25cf Goose Mode (Configure Goose mode)\n// highlight-end\n|  \u25cb Tool Output \n\u2514  \n"})}),(0,t.jsxs)(n.ol,{start:"4",children:["\n",(0,t.jsx)(n.li,{children:"Choose the Goose mode you would like to configure."}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"\u250c   goose-configure \n\u2502\n\u25c7  What would you like to configure?\n\u2502  Goose Settings  \n\u2502\n\u25c7  What setting would you like to configure?\n\u2502  Goose Mode\n\u2502\n\u25c6  Which Goose mode would you like to configure?\n// highlight-start\n\u2502  \u25cf Auto Mode\n// highlight-end\n|  \u25cb Approve Mode\n|  \u25cb Chat Mode\n|\n\u2514  Set to Auto Mode - full file modification enabled\n"})})]}),(0,t.jsx)(a.A,{value:"ui",label:"Goose Desktop",children:(0,t.jsxs)(n.admonition,{title:"\ud83d\ude80 Goose Modes in Desktop \u2013 Coming Soon!",type:"info",children:[(0,t.jsx)(n.p,{children:"Currently, Goose Modes can only be configured via the CLI."}),(0,t.jsxs)(n.p,{children:["By default, Goose Desktop operates in ",(0,t.jsx)(n.strong,{children:"Auto Mode"}),", allowing full automation."]}),(0,t.jsxs)(n.p,{children:["A future update will bring ",(0,t.jsx)(n.strong,{children:"Goose Mode selection"})," to the Desktop app. Stay tuned!"]})]})})]})]})}function g(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(h,{...e})}):h(e)}},9329:(e,n,o)=>{o.d(n,{A:()=>i});o(6540);var s=o(4164);const t={tabItem:"tabItem_Ymn6"};var r=o(4848);function i(e){let{children:n,hidden:o,className:i}=e;return(0,r.jsx)("div",{role:"tabpanel",className:(0,s.A)(t.tabItem,i),hidden:o,children:n})}},5537:(e,n,o)=>{o.d(n,{A:()=>w});var s=o(6540),t=o(4164),r=o(5627),i=o(6347),a=o(372),l=o(604),u=o(1861),d=o(8749);function c(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:o}=e;return(0,s.useMemo)((()=>{const e=n??function(e){return c(e).map((e=>{let{props:{value:n,label:o,attributes:s,default:t}}=e;return{value:n,label:o,attributes:s,default:t}}))}(o);return function(e){const n=(0,u.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,o])}function g(e){let{value:n,tabValues:o}=e;return o.some((e=>e.value===n))}function p(e){let{queryString:n=!1,groupId:o}=e;const t=(0,i.W6)(),r=function(e){let{queryString:n=!1,groupId:o}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!o)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return o??null}({queryString:n,groupId:o});return[(0,l.aZ)(r),(0,s.useCallback)((e=>{if(!r)return;const n=new URLSearchParams(t.location.search);n.set(r,e),t.replace({...t.location,search:n.toString()})}),[r,t])]}function m(e){const{defaultValue:n,queryString:o=!1,groupId:t}=e,r=h(e),[i,l]=(0,s.useState)((()=>function(e){let{defaultValue:n,tabValues:o}=e;if(0===o.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!g({value:n,tabValues:o}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${o.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const s=o.find((e=>e.default))??o[0];if(!s)throw new Error("Unexpected error: 0 tabValues");return s.value}({defaultValue:n,tabValues:r}))),[u,c]=p({queryString:o,groupId:t}),[m,f]=function(e){let{groupId:n}=e;const o=function(e){return e?`docusaurus.tab.${e}`:null}(n),[t,r]=(0,d.Dv)(o);return[t,(0,s.useCallback)((e=>{o&&r.set(e)}),[o,r])]}({groupId:t}),x=(()=>{const e=u??m;return g({value:e,tabValues:r})?e:null})();(0,a.A)((()=>{x&&l(x)}),[x]);return{selectedValue:i,selectValue:(0,s.useCallback)((e=>{if(!g({value:e,tabValues:r}))throw new Error(`Can't select invalid tab value=${e}`);l(e),c(e),f(e)}),[c,f,r]),tabValues:r}}var f=o(9136);const x={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var b=o(4848);function j(e){let{className:n,block:o,selectedValue:s,selectValue:i,tabValues:a}=e;const l=[],{blockElementScrollPositionUntilNextRender:u}=(0,r.a_)(),d=e=>{const n=e.currentTarget,o=l.indexOf(n),t=a[o].value;t!==s&&(u(n),i(t))},c=e=>{let n=null;switch(e.key){case"Enter":d(e);break;case"ArrowRight":{const o=l.indexOf(e.currentTarget)+1;n=l[o]??l[0];break}case"ArrowLeft":{const o=l.indexOf(e.currentTarget)-1;n=l[o]??l[l.length-1];break}}n?.focus()};return(0,b.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,t.A)("tabs",{"tabs--block":o},n),children:a.map((e=>{let{value:n,label:o,attributes:r}=e;return(0,b.jsx)("li",{role:"tab",tabIndex:s===n?0:-1,"aria-selected":s===n,ref:e=>{l.push(e)},onKeyDown:c,onClick:d,...r,className:(0,t.A)("tabs__item",x.tabItem,r?.className,{"tabs__item--active":s===n}),children:o??n},n)}))})}function v(e){let{lazy:n,children:o,selectedValue:r}=e;const i=(Array.isArray(o)?o:[o]).filter(Boolean);if(n){const e=i.find((e=>e.props.value===r));return e?(0,s.cloneElement)(e,{className:(0,t.A)("margin-top--md",e.props.className)}):null}return(0,b.jsx)("div",{className:"margin-top--md",children:i.map(((e,n)=>(0,s.cloneElement)(e,{key:n,hidden:e.props.value!==r})))})}function y(e){const n=m(e);return(0,b.jsxs)("div",{className:(0,t.A)("tabs-container",x.tabList),children:[(0,b.jsx)(j,{...n,...e}),(0,b.jsx)(v,{...n,...e})]})}function w(e){const n=(0,f.A)();return(0,b.jsx)(y,{...e,children:c(e.children)},String(n))}},8453:(e,n,o)=>{o.d(n,{R:()=>i,x:()=>a});var s=o(6540);const t={},r=s.createContext(t);function i(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:i(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);