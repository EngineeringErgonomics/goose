"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[3023],{1837:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>u,contentTitle:()=>c,default:()=>g,frontMatter:()=>l,metadata:()=>s,toc:()=>d});const s=JSON.parse('{"id":"configuration/managing-extensions","title":"Managing Goose Extensions","description":"Goose Extensions are add-ons that provide are a way to extend the functionality of Goose. They also provide a way to connect Goose with applications and tools you already use in your workflow. These extensions can be used to add new features, automate tasks, or integrate with other systems.","source":"@site/docs/configuration/managing-extensions.md","sourceDirName":"configuration","slug":"/configuration/managing-extensions","permalink":"/goose/v1/docs/configuration/managing-extensions","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/configuration/managing-extensions.md","tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"sidebar_position":2,"title":"Managing Goose Extensions"},"sidebar":"tutorialSidebar","previous":{"title":"Supported Providers","permalink":"/goose/v1/docs/configuration/providers"},"next":{"title":"Extensions Design Guide","permalink":"/goose/v1/docs/configuration/extensions"}}');var o=t(4848),a=t(8453),i=t(5537),r=t(9329);const l={sidebar_position:2,title:"Managing Goose Extensions"},c=void 0,u={},d=[{value:"Built-in Extensions",id:"built-in-extensions",level:3},{value:"Managing Extensions in Goose",id:"managing-extensions-in-goose",level:2},{value:"Discovering Extensions",id:"discovering-extensions",level:3},{value:"Adding or Removing Extensions",id:"adding-or-removing-extensions",level:3},{value:"Starting a Goose Session with Extensions",id:"starting-a-goose-session-with-extensions",level:2},{value:"Developing Extensions",id:"developing-extensions",level:2}];function h(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h3:"h3",img:"img",p:"p",pre:"pre",strong:"strong",...(0,a.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.p,{children:"Goose Extensions are add-ons that provide are a way to extend the functionality of Goose. They also provide a way to connect Goose with applications and tools you already use in your workflow. These extensions can be used to add new features, automate tasks, or integrate with other systems."}),"\n",(0,o.jsx)(n.h3,{id:"built-in-extensions",children:"Built-in Extensions"}),"\n",(0,o.jsx)(n.p,{children:"Goose comes with a few built-in extensions that provide additional functionality."}),"\n",(0,o.jsx)(n.p,{children:"To see which extensions are available, you can run the following command:"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose system list\n"})}),"\n",(0,o.jsx)(n.h2,{id:"managing-extensions-in-goose",children:"Managing Extensions in Goose"}),"\n",(0,o.jsx)(n.h3,{id:"discovering-extensions",children:"Discovering Extensions"}),"\n",(0,o.jsxs)(n.p,{children:["Goose comes with a ",(0,o.jsx)(n.a,{href:"https://silver-disco-nvm6v4e.pages.github.io/",children:"central directory"})," of extensions that you can install and use. You can install extensions from the Goose CLI or from the Goose GUI."]}),"\n",(0,o.jsxs)(n.p,{children:["You can also bring in any third-party extension of your choice using the ",(0,o.jsx)(n.a,{href:"https://github.com/modelcontextprotocol/servers",children:"MCP server"})," link as the ",(0,o.jsx)(n.code,{children:"system_url"}),"."]}),"\n",(0,o.jsx)(n.h3,{id:"adding-or-removing-extensions",children:"Adding or Removing Extensions"}),"\n",(0,o.jsxs)(i.A,{children:[(0,o.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,o.jsx)(n.p,{children:"To add or remove an extension on Goose CLI, copy the extension URL and run the following command:"}),(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Add Extension"})}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose system add <system_url>\n"})}),(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Remove Extension"})}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose system remove <system_url>\n"})})]}),(0,o.jsxs)(r.A,{value:"ui",label:"Goose UI",children:[(0,o.jsx)(n.p,{children:"Extensions can be installed directly from the directory to the Goose UI as shown below."}),(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"Install Extension",src:t(21).A+"",width:"1050",height:"646"})}),(0,o.jsx)(n.p,{children:"They can then be toggled on or off from the Extensions tab under settings."}),(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"Manage Extensions",src:t(9058).A+"",width:"749",height:"289"})})]})]}),"\n",(0,o.jsx)(n.h2,{id:"starting-a-goose-session-with-extensions",children:"Starting a Goose Session with Extensions"}),"\n",(0,o.jsx)(n.p,{children:"You can start a tailored goose session with specific extensions directly from the CLI. To do this, run the following command:"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:'goose session --with-system "<system_url>"\n'})}),"\n",(0,o.jsx)(n.admonition,{type:"note",children:(0,o.jsx)(n.p,{children:"You may need to set necessary environment variables for the extension to work correctly."})}),"\n",(0,o.jsx)(n.h2,{id:"developing-extensions",children:"Developing Extensions"}),"\n",(0,o.jsxs)(n.p,{children:["Goose extensions are implemented with the Model Context Protocol (MCP) - a system that allows AI models and agents to securely connect with local or remote resources using standard protocols. Learn how to build your own ",(0,o.jsx)(n.a,{href:"https://modelcontextprotocol.io/quickstart/server",children:"extension as an MCP server"}),"."]})]})}function g(e={}){const{wrapper:n}={...(0,a.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(h,{...e})}):h(e)}},9329:(e,n,t)=>{t.d(n,{A:()=>i});t(6540);var s=t(4164);const o={tabItem:"tabItem_Ymn6"};var a=t(4848);function i(e){let{children:n,hidden:t,className:i}=e;return(0,a.jsx)("div",{role:"tabpanel",className:(0,s.A)(o.tabItem,i),hidden:t,children:n})}},5537:(e,n,t)=>{t.d(n,{A:()=>j});var s=t(6540),o=t(4164),a=t(5627),i=t(6347),r=t(372),l=t(604),c=t(1861),u=t(8749);function d(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:t}=e;return(0,s.useMemo)((()=>{const e=n??function(e){return d(e).map((e=>{let{props:{value:n,label:t,attributes:s,default:o}}=e;return{value:n,label:t,attributes:s,default:o}}))}(t);return function(e){const n=(0,c.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,t])}function g(e){let{value:n,tabValues:t}=e;return t.some((e=>e.value===n))}function m(e){let{queryString:n=!1,groupId:t}=e;const o=(0,i.W6)(),a=function(e){let{queryString:n=!1,groupId:t}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!t)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return t??null}({queryString:n,groupId:t});return[(0,l.aZ)(a),(0,s.useCallback)((e=>{if(!a)return;const n=new URLSearchParams(o.location.search);n.set(a,e),o.replace({...o.location,search:n.toString()})}),[a,o])]}function p(e){const{defaultValue:n,queryString:t=!1,groupId:o}=e,a=h(e),[i,l]=(0,s.useState)((()=>function(e){let{defaultValue:n,tabValues:t}=e;if(0===t.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!g({value:n,tabValues:t}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${t.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const s=t.find((e=>e.default))??t[0];if(!s)throw new Error("Unexpected error: 0 tabValues");return s.value}({defaultValue:n,tabValues:a}))),[c,d]=m({queryString:t,groupId:o}),[p,x]=function(e){let{groupId:n}=e;const t=function(e){return e?`docusaurus.tab.${e}`:null}(n),[o,a]=(0,u.Dv)(t);return[o,(0,s.useCallback)((e=>{t&&a.set(e)}),[t,a])]}({groupId:o}),f=(()=>{const e=c??p;return g({value:e,tabValues:a})?e:null})();(0,r.A)((()=>{f&&l(f)}),[f]);return{selectedValue:i,selectValue:(0,s.useCallback)((e=>{if(!g({value:e,tabValues:a}))throw new Error(`Can't select invalid tab value=${e}`);l(e),d(e),x(e)}),[d,x,a]),tabValues:a}}var x=t(9136);const f={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var v=t(4848);function b(e){let{className:n,block:t,selectedValue:s,selectValue:i,tabValues:r}=e;const l=[],{blockElementScrollPositionUntilNextRender:c}=(0,a.a_)(),u=e=>{const n=e.currentTarget,t=l.indexOf(n),o=r[t].value;o!==s&&(c(n),i(o))},d=e=>{let n=null;switch(e.key){case"Enter":u(e);break;case"ArrowRight":{const t=l.indexOf(e.currentTarget)+1;n=l[t]??l[0];break}case"ArrowLeft":{const t=l.indexOf(e.currentTarget)-1;n=l[t]??l[l.length-1];break}}n?.focus()};return(0,v.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,o.A)("tabs",{"tabs--block":t},n),children:r.map((e=>{let{value:n,label:t,attributes:a}=e;return(0,v.jsx)("li",{role:"tab",tabIndex:s===n?0:-1,"aria-selected":s===n,ref:e=>{l.push(e)},onKeyDown:d,onClick:u,...a,className:(0,o.A)("tabs__item",f.tabItem,a?.className,{"tabs__item--active":s===n}),children:t??n},n)}))})}function y(e){let{lazy:n,children:t,selectedValue:a}=e;const i=(Array.isArray(t)?t:[t]).filter(Boolean);if(n){const e=i.find((e=>e.props.value===a));return e?(0,s.cloneElement)(e,{className:(0,o.A)("margin-top--md",e.props.className)}):null}return(0,v.jsx)("div",{className:"margin-top--md",children:i.map(((e,n)=>(0,s.cloneElement)(e,{key:n,hidden:e.props.value!==a})))})}function w(e){const n=p(e);return(0,v.jsxs)("div",{className:(0,o.A)("tabs-container",f.tabList),children:[(0,v.jsx)(b,{...n,...e}),(0,v.jsx)(y,{...n,...e})]})}function j(e){const n=(0,x.A)();return(0,v.jsx)(w,{...e,children:d(e.children)},String(n))}},21:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/install-extension-ui-af312043d371e94c90255342e33e3399.png"},9058:(e,n,t)=>{t.d(n,{A:()=>s});const s=t.p+"assets/images/manage-extensions-ui-566a099ac6d76775e599085e06f02ba4.png"},8453:(e,n,t)=>{t.d(n,{R:()=>i,x:()=>r});var s=t(6540);const o={},a=s.createContext(o);function i(e){const n=s.useContext(a);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:i(e.components),s.createElement(a.Provider,{value:n},e.children)}}}]);