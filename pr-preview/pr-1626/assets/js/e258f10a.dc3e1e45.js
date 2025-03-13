"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[2279],{3935:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>d,contentTitle:()=>c,default:()=>g,frontMatter:()=>u,metadata:()=>s,toc:()=>p});const s=JSON.parse('{"id":"guides/updating-goose","title":"Updating Goose","description":"You can update Goose by running:","source":"@site/docs/guides/updating-goose.md","sourceDirName":"guides","slug":"/guides/updating-goose","permalink":"/goose/pr-preview/pr-1626/docs/guides/updating-goose","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2},"sidebar":"tutorialSidebar","previous":{"title":"Managing Sessions","permalink":"/goose/pr-preview/pr-1626/docs/guides/managing-goose-sessions"},"next":{"title":"Goose Permissions","permalink":"/goose/pr-preview/pr-1626/docs/guides/goose-permissions"}}');var o=n(4848),a=n(8453),r=n(5537),l=n(9329),i=n(9773);const u={sidebar_position:2},c="Updating Goose",d={},p=[];function h(e){const t={a:"a",admonition:"admonition",code:"code",h1:"h1",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",...(0,a.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(t.header,{children:(0,o.jsx)(t.h1,{id:"updating-goose",children:"Updating Goose"})}),"\n","\n",(0,o.jsxs)(r.A,{groupId:"interface",children:[(0,o.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,o.jsx)(t.p,{children:"You can update Goose by running:"}),(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-sh",children:"goose update\n"})}),(0,o.jsxs)(t.p,{children:["Additional ",(0,o.jsx)(t.a,{href:"/docs/guides/goose-cli-commands#update-options",children:"options"}),":"]}),(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-sh",children:"# Update to latest canary (development) version\ngoose update --canary\n\n# Update and reconfigure settings\ngoose update --reconfigure\n"})}),(0,o.jsxs)(t.p,{children:["Or you can run the ",(0,o.jsx)(t.a,{href:"/docs/getting-started/installation",children:"installation"})," script again:"]}),(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-sh",children:"curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash\n"})}),(0,o.jsx)(t.p,{children:"To check your current Goose version, use the following command:"}),(0,o.jsx)(t.pre,{children:(0,o.jsx)(t.code,{className:"language-sh",children:"goose --version\n"})})]}),(0,o.jsxs)(l.A,{value:"ui",label:"Goose Desktop",children:[(0,o.jsx)(t.admonition,{type:"info",children:(0,o.jsx)(t.p,{children:"To update Goose to the latest stable version, reinstall using the instructions below"})}),(0,o.jsx)("div",{style:{marginTop:"1rem"},children:(0,o.jsxs)(t.ol,{children:["\n",(0,o.jsxs)(t.li,{children:["\n",(0,o.jsx)(i.A,{}),"\n"]}),"\n",(0,o.jsx)(t.li,{children:"Unzip the downloaded zip file."}),"\n",(0,o.jsx)(t.li,{children:"Run the executable file to launch the Goose Desktop application."}),"\n",(0,o.jsx)(t.li,{children:"Overwrite the existing Goose application with the new version."}),"\n",(0,o.jsx)(t.li,{children:"Run the executable file to launch the Goose desktop application."}),"\n"]})})]})]})]})}function g(e={}){const{wrapper:t}={...(0,a.R)(),...e.components};return t?(0,o.jsx)(t,{...e,children:(0,o.jsx)(h,{...e})}):h(e)}},9329:(e,t,n)=>{n.d(t,{A:()=>r});n(6540);var s=n(4164);const o={tabItem:"tabItem_Ymn6"};var a=n(4848);function r(e){let{children:t,hidden:n,className:r}=e;return(0,a.jsx)("div",{role:"tabpanel",className:(0,s.A)(o.tabItem,r),hidden:n,children:t})}},5537:(e,t,n)=>{n.d(t,{A:()=>y});var s=n(6540),o=n(4164),a=n(5627),r=n(6347),l=n(372),i=n(604),u=n(1861),c=n(8749);function d(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:t}=e;return!!t&&"object"==typeof t&&"value"in t}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function p(e){const{values:t,children:n}=e;return(0,s.useMemo)((()=>{const e=t??function(e){return d(e).map((e=>{let{props:{value:t,label:n,attributes:s,default:o}}=e;return{value:t,label:n,attributes:s,default:o}}))}(n);return function(e){const t=(0,u.XI)(e,((e,t)=>e.value===t.value));if(t.length>0)throw new Error(`Docusaurus error: Duplicate values "${t.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[t,n])}function h(e){let{value:t,tabValues:n}=e;return n.some((e=>e.value===t))}function g(e){let{queryString:t=!1,groupId:n}=e;const o=(0,r.W6)(),a=function(e){let{queryString:t=!1,groupId:n}=e;if("string"==typeof t)return t;if(!1===t)return null;if(!0===t&&!n)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return n??null}({queryString:t,groupId:n});return[(0,i.aZ)(a),(0,s.useCallback)((e=>{if(!a)return;const t=new URLSearchParams(o.location.search);t.set(a,e),o.replace({...o.location,search:t.toString()})}),[a,o])]}function m(e){const{defaultValue:t,queryString:n=!1,groupId:o}=e,a=p(e),[r,i]=(0,s.useState)((()=>function(e){let{defaultValue:t,tabValues:n}=e;if(0===n.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(t){if(!h({value:t,tabValues:n}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${t}" but none of its children has the corresponding value. Available values are: ${n.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return t}const s=n.find((e=>e.default))??n[0];if(!s)throw new Error("Unexpected error: 0 tabValues");return s.value}({defaultValue:t,tabValues:a}))),[u,d]=g({queryString:n,groupId:o}),[m,b]=function(e){let{groupId:t}=e;const n=function(e){return e?`docusaurus.tab.${e}`:null}(t),[o,a]=(0,c.Dv)(n);return[o,(0,s.useCallback)((e=>{n&&a.set(e)}),[n,a])]}({groupId:o}),f=(()=>{const e=u??m;return h({value:e,tabValues:a})?e:null})();(0,l.A)((()=>{f&&i(f)}),[f]);return{selectedValue:r,selectValue:(0,s.useCallback)((e=>{if(!h({value:e,tabValues:a}))throw new Error(`Can't select invalid tab value=${e}`);i(e),d(e),b(e)}),[d,b,a]),tabValues:a}}var b=n(9136);const f={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var v=n(4848);function x(e){let{className:t,block:n,selectedValue:s,selectValue:r,tabValues:l}=e;const i=[],{blockElementScrollPositionUntilNextRender:u}=(0,a.a_)(),c=e=>{const t=e.currentTarget,n=i.indexOf(t),o=l[n].value;o!==s&&(u(t),r(o))},d=e=>{let t=null;switch(e.key){case"Enter":c(e);break;case"ArrowRight":{const n=i.indexOf(e.currentTarget)+1;t=i[n]??i[0];break}case"ArrowLeft":{const n=i.indexOf(e.currentTarget)-1;t=i[n]??i[i.length-1];break}}t?.focus()};return(0,v.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,o.A)("tabs",{"tabs--block":n},t),children:l.map((e=>{let{value:t,label:n,attributes:a}=e;return(0,v.jsx)("li",{role:"tab",tabIndex:s===t?0:-1,"aria-selected":s===t,ref:e=>{i.push(e)},onKeyDown:d,onClick:c,...a,className:(0,o.A)("tabs__item",f.tabItem,a?.className,{"tabs__item--active":s===t}),children:n??t},t)}))})}function j(e){let{lazy:t,children:n,selectedValue:a}=e;const r=(Array.isArray(n)?n:[n]).filter(Boolean);if(t){const e=r.find((e=>e.props.value===a));return e?(0,s.cloneElement)(e,{className:(0,o.A)("margin-top--md",e.props.className)}):null}return(0,v.jsx)("div",{className:"margin-top--md",children:r.map(((e,t)=>(0,s.cloneElement)(e,{key:t,hidden:e.props.value!==a})))})}function w(e){const t=m(e);return(0,v.jsxs)("div",{className:(0,o.A)("tabs-container",f.tabList),children:[(0,v.jsx)(x,{...t,...e}),(0,v.jsx)(j,{...t,...e})]})}function y(e){const t=(0,b.A)();return(0,v.jsx)(w,{...e,children:d(e.children)},String(t))}},9773:(e,t,n)=>{n.d(t,{A:()=>r});var s=n(6289),o=n(6960),a=n(4848);const r=()=>(0,a.jsxs)("div",{children:[(0,a.jsx)("p",{children:"To download Goose Desktop for macOS, click one of the buttons below:"}),(0,a.jsxs)("div",{className:"pill-button",children:[(0,a.jsxs)(s.A,{className:"button button--primary button--lg",to:"https://github.com/block/goose/releases/download/stable/Goose.zip",children:[(0,a.jsx)(o.i,{})," macOS Silicon"]}),(0,a.jsxs)(s.A,{className:"button button--primary button--lg",to:"https://github.com/block/goose/releases/download/stable/Goose_intel_mac.zip",children:[(0,a.jsx)(o.i,{})," macOS Intel"]})]})]})},6960:(e,t,n)=>{n.d(t,{i:()=>o});var s=n(4848);const o=e=>{let{className:t=""}=e;return(0,s.jsx)("svg",{width:"1.5rem",height:"1.5rem",fill:"none",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 24 24","aria-hidden":"true",className:t,children:(0,s.jsx)("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M15.76 12.15a1 1 0 0 0-1.52-1.3L13 12.296V3a1 1 0 1 0-2 0v9.297l-1.24-1.448a1 1 0 0 0-1.52 1.302l3 3.5a1 1 0 0 0 1.52 0l3-3.5ZM5 16a1 1 0 1 0-2 0v4a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-4a1 1 0 1 0-2 0v3H5v-3Z",fill:"currentColor"})})}},8453:(e,t,n)=>{n.d(t,{R:()=>r,x:()=>l});var s=n(6540);const o={},a=s.createContext(o);function r(e){const t=s.useContext(a);return s.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function l(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:r(e.components),s.createElement(a.Provider,{value:t},e.children)}}}]);