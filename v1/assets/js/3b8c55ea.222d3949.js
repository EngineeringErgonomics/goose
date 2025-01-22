"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6803],{23:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>d,contentTitle:()=>c,default:()=>x,frontMatter:()=>a,metadata:()=>s,toc:()=>u});const s=JSON.parse('{"id":"installation","title":"Installation","description":"Installing the Goose Cli","source":"@site/docs/installation.md","sourceDirName":".","slug":"/installation","permalink":"/goose/v1/docs/installation","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/installation.md","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","next":{"title":"Quickstart","permalink":"/goose/v1/docs/quickstart"}}');var t=o(4848),i=o(8453),l=o(5537),r=o(9329);const a={sidebar_position:1},c="Installation",d={},u=[{value:"Installing the Goose Cli",id:"installing-the-goose-cli",level:4},{value:"Instaling the Goose UI",id:"instaling-the-goose-ui",level:4},{value:"Configuration",id:"configuration",level:3},{value:"Set up a provider",id:"set-up-a-provider",level:4},{value:"Toggle Extensions",id:"toggle-extensions",level:4},{value:"Adding An Extension",id:"adding-an-extension",level:4},{value:"Running Goose",id:"running-goose",level:2},{value:"Additional Resources",id:"additional-resources",level:2}];function h(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",h3:"h3",h4:"h4",header:"header",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.header,{children:(0,t.jsx)(n.h1,{id:"installation",children:"Installation"})}),"\n",(0,t.jsx)(n.h4,{id:"installing-the-goose-cli",children:"Installing the Goose Cli"}),"\n",(0,t.jsx)(n.p,{children:"To install Goose CLI, run the following script on macOS or Linux."}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | sh\n"})}),"\n",(0,t.jsx)(n.p,{children:"This script will fetch the latest version of Goose and set it up on your system."}),"\n",(0,t.jsx)(n.h4,{id:"instaling-the-goose-ui",children:"Instaling the Goose UI"}),"\n",(0,t.jsx)(n.p,{children:"To install the Goose desktop UI, follow these steps:"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Visit the ",(0,t.jsx)(n.a,{href:"https://github.com/block/goose/releases/tag/stable",children:"Goose Releases page"})]}),"\n",(0,t.jsxs)(n.li,{children:["Download the ",(0,t.jsx)(n.code,{children:"Goose.zip"})," file."]}),"\n",(0,t.jsxs)(n.li,{children:["Open the downloaded ",(0,t.jsx)(n.code,{children:"Goose.zip"})," file and launch the desktop application."]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"configuration",children:"Configuration"}),"\n",(0,t.jsx)(n.p,{children:"Goose allows you to configure settings for both its command-line interface (CLI) and desktop UI. You can update your LLM provider and API key, enable or disable extensions, and add new extensions to enhance Goose's functionality."}),"\n",(0,t.jsx)(n.h4,{id:"set-up-a-provider",children:"Set up a provider"}),"\n",(0,t.jsxs)(n.p,{children:["Goose works with a set of ",(0,t.jsx)(n.a,{href:"https://block.github.io/goose/plugins/providers.html",children:"supported LLM providers"})," that you can obtain an API key from if you don't already have one. You'll be prompted to pick a provider and set an API key on your initial install of Goose."]}),"\n",(0,t.jsxs)(l.A,{children:[(0,t.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"To update your LLM provider and API key:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Run the following command:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Configure Providers"})," from the menu."]}),"\n",(0,t.jsx)(n.li,{children:"Follow the prompts to choose your LLM privider and enter or update your API key."}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Example:"})}),(0,t.jsx)(n.p,{children:"To select an option during configuration, hover over it and press Enter."}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"What would you like to configure?\n> Configure Providers\n  Toggle Extensions\n  Add Extension\n\nWhich Model provider should we use?\n> OpenAI\n  Databricks\n  Ollama\n.....\n\nEnter Api Key:\n>  sk-1234abcd5678efgh\n"})}),(0,t.jsx)(n.admonition,{title:"Billing",type:"info",children:(0,t.jsxs)(n.p,{children:["You will need to have credits in your LLM Provider account (when necessary) to be able to successfully make requests. Some providers also have rate limits on API usage, which can affect your experience. Check out our ",(0,t.jsx)(n.a,{href:"https://block.github.io/goose/v1/docs/guidance/handling-llm-rate-limits-with-goose",children:"Handling Rate Limits"})," guide to learn how to efficiently manage these limits while using Goose."]})})]}),(0,t.jsxs)(r.A,{value:"ui",label:"Goose UI",children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"To update your LLM provider and API key:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Click on the three dots in the top-right corner."}),"\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Provider Settings"})," from the menu."]}),"\n",(0,t.jsx)(n.li,{children:"Choose a provider from the list."}),"\n",(0,t.jsxs)(n.li,{children:["Click Edit, enter your API key, and click ",(0,t.jsx)(n.code,{children:"Set as Active"}),"."]}),"\n"]})]})]}),"\n",(0,t.jsx)(n.h4,{id:"toggle-extensions",children:"Toggle Extensions"}),"\n",(0,t.jsxs)(n.p,{children:["Goose Extensions are add-ons utilizing ",(0,t.jsx)(n.a,{href:"https://www.anthropic.com/news/model-context-protocol",children:"Anthropic's Model Context Protocol(MCP)"}),", that enhance Goose's functionality by connecting it with different applications and tools you already use in your workflow. Extensions can be used to add new features, automate tasks, and integrate with other systems."]}),"\n",(0,t.jsxs)(l.A,{children:[(0,t.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"To enable or disable extensions that are already installed:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Run the following command to open up Goose's configurations:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Toggle Extensions"})," from the menu."]}),"\n",(0,t.jsx)(n.li,{children:"A list of already installed extensions will populate."}),"\n",(0,t.jsxs)(n.li,{children:["Press the ",(0,t.jsx)(n.code,{children:"space bar"})," to toggle the extension ",(0,t.jsx)(n.code,{children:"enabled"})," or ",(0,t.jsx)(n.code,{children:"disabled"}),"."]}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Example:"})}),(0,t.jsx)(n.p,{children:"To select an option during configuration, hover over it and press Enter."}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:'What would you like to configure?\n  Configure Providers\n> Toggle Extensions\n  Add Extension\n\nEnable systems: (use "space" to toggle and "enter" to submit)\n[ ] Developer Tools \n[X] JetBrains\n'})})]}),(0,t.jsxs)(r.A,{value:"ui",label:"Goose UI",children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"To enable or disable extensions that are already installed:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Click the three dots in the top-right corner of the application."}),"\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Settings"})," from the menu, then click on the ",(0,t.jsx)(n.code,{children:"Extensions"})," section."]}),"\n",(0,t.jsx)(n.li,{children:"Use the toggle switch next to each extension to enable or disable it."}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{alt:"Manage Extensions",src:o(9058).A+"",width:"749",height:"289"})})]})]}),"\n",(0,t.jsx)(n.h4,{id:"adding-an-extension",children:"Adding An Extension"}),"\n",(0,t.jsxs)(n.p,{children:["Extensions allow you to expand Goose's capabilities by connecting it with additional tools and systems. You can add built-in extensions provided by Goose, or integrate external extensions using ",(0,t.jsx)(n.a,{href:"https://www.anthropic.com/news/model-context-protocol",children:"Anthropic's Model Context Protocol (MCP)"}),"."]}),"\n",(0,t.jsxs)(l.A,{children:[(0,t.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"To add a Built-in, Command-line or Remote extension:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"run the following command:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Select ",(0,t.jsx)(n.code,{children:"Add Extension"})," from the menu."]}),"\n",(0,t.jsxs)(n.li,{children:["Choose the type of extension you\u2019d like to add:","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"Built-In Extension"}),": Use an extension that comes pre-installed with Goose."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"Command-Line Extension"}),": Add a local command or script to run as an extension."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"Remote Extension"}),": Connect to a remote system via SSE (Server-Sent Events)."]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.li,{children:"Follow the prompts based on the type of extension you selected."}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Example: Adding Built-in Extension"})}),(0,t.jsx)(n.p,{children:"To select an option during configuration, hover over it and press Enter."}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"What would you like to configure?\n  Configure Providers\n  Toggle Extensions\n> Add Extension\n\n\nWhat type of extension would you like to add?\n> Built-in Extension\n  Command-line Extension\n  Remote Extension\n\nWhich Built-in extension would you like to enable?\n  Developer Tools\n  Non Developer\n> Jetbrains\n"})})]}),(0,t.jsxs)(r.A,{value:"ui",label:"Goose UI",children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Extensions can be installed directly from the directory to the Goose UI as shown below."})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.img,{alt:"Install Extension",src:o(21).A+"",width:"1050",height:"646"})})]})]}),"\n",(0,t.jsx)(n.h2,{id:"running-goose",children:"Running Goose"}),"\n",(0,t.jsxs)(n.p,{children:["You can run ",(0,t.jsx)(n.code,{children:"goose"})," from the command line using:"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose session start\n"})}),"\n",(0,t.jsx)(n.h2,{id:"additional-resources",children:"Additional Resources"}),"\n",(0,t.jsxs)(n.p,{children:["Visit the ",(0,t.jsx)(n.a,{href:"https://block.github.io/goose/configuration.html",children:"Configuration Guide"})," for detailed instructions on configuring Goose."]})]})}function x(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(h,{...e})}):h(e)}},9329:(e,n,o)=>{o.d(n,{A:()=>l});o(6540);var s=o(4164);const t={tabItem:"tabItem_Ymn6"};var i=o(4848);function l(e){let{children:n,hidden:o,className:l}=e;return(0,i.jsx)("div",{role:"tabpanel",className:(0,s.A)(t.tabItem,l),hidden:o,children:n})}},5537:(e,n,o)=>{o.d(n,{A:()=>w});var s=o(6540),t=o(4164),i=o(5627),l=o(6347),r=o(372),a=o(604),c=o(1861),d=o(8749);function u(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:o}=e;return(0,s.useMemo)((()=>{const e=n??function(e){return u(e).map((e=>{let{props:{value:n,label:o,attributes:s,default:t}}=e;return{value:n,label:o,attributes:s,default:t}}))}(o);return function(e){const n=(0,c.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,o])}function x(e){let{value:n,tabValues:o}=e;return o.some((e=>e.value===n))}function p(e){let{queryString:n=!1,groupId:o}=e;const t=(0,l.W6)(),i=function(e){let{queryString:n=!1,groupId:o}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!o)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return o??null}({queryString:n,groupId:o});return[(0,a.aZ)(i),(0,s.useCallback)((e=>{if(!i)return;const n=new URLSearchParams(t.location.search);n.set(i,e),t.replace({...t.location,search:n.toString()})}),[i,t])]}function g(e){const{defaultValue:n,queryString:o=!1,groupId:t}=e,i=h(e),[l,a]=(0,s.useState)((()=>function(e){let{defaultValue:n,tabValues:o}=e;if(0===o.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!x({value:n,tabValues:o}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${o.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const s=o.find((e=>e.default))??o[0];if(!s)throw new Error("Unexpected error: 0 tabValues");return s.value}({defaultValue:n,tabValues:i}))),[c,u]=p({queryString:o,groupId:t}),[g,m]=function(e){let{groupId:n}=e;const o=function(e){return e?`docusaurus.tab.${e}`:null}(n),[t,i]=(0,d.Dv)(o);return[t,(0,s.useCallback)((e=>{o&&i.set(e)}),[o,i])]}({groupId:t}),f=(()=>{const e=c??g;return x({value:e,tabValues:i})?e:null})();(0,r.A)((()=>{f&&a(f)}),[f]);return{selectedValue:l,selectValue:(0,s.useCallback)((e=>{if(!x({value:e,tabValues:i}))throw new Error(`Can't select invalid tab value=${e}`);a(e),u(e),m(e)}),[u,m,i]),tabValues:i}}var m=o(9136);const f={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var j=o(4848);function b(e){let{className:n,block:o,selectedValue:s,selectValue:l,tabValues:r}=e;const a=[],{blockElementScrollPositionUntilNextRender:c}=(0,i.a_)(),d=e=>{const n=e.currentTarget,o=a.indexOf(n),t=r[o].value;t!==s&&(c(n),l(t))},u=e=>{let n=null;switch(e.key){case"Enter":d(e);break;case"ArrowRight":{const o=a.indexOf(e.currentTarget)+1;n=a[o]??a[0];break}case"ArrowLeft":{const o=a.indexOf(e.currentTarget)-1;n=a[o]??a[a.length-1];break}}n?.focus()};return(0,j.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,t.A)("tabs",{"tabs--block":o},n),children:r.map((e=>{let{value:n,label:o,attributes:i}=e;return(0,j.jsx)("li",{role:"tab",tabIndex:s===n?0:-1,"aria-selected":s===n,ref:e=>{a.push(e)},onKeyDown:u,onClick:d,...i,className:(0,t.A)("tabs__item",f.tabItem,i?.className,{"tabs__item--active":s===n}),children:o??n},n)}))})}function v(e){let{lazy:n,children:o,selectedValue:i}=e;const l=(Array.isArray(o)?o:[o]).filter(Boolean);if(n){const e=l.find((e=>e.props.value===i));return e?(0,s.cloneElement)(e,{className:(0,t.A)("margin-top--md",e.props.className)}):null}return(0,j.jsx)("div",{className:"margin-top--md",children:l.map(((e,n)=>(0,s.cloneElement)(e,{key:n,hidden:e.props.value!==i})))})}function y(e){const n=g(e);return(0,j.jsxs)("div",{className:(0,t.A)("tabs-container",f.tabList),children:[(0,j.jsx)(b,{...n,...e}),(0,j.jsx)(v,{...n,...e})]})}function w(e){const n=(0,m.A)();return(0,j.jsx)(y,{...e,children:u(e.children)},String(n))}},21:(e,n,o)=>{o.d(n,{A:()=>s});const s=o.p+"assets/images/install-extension-ui-af312043d371e94c90255342e33e3399.png"},9058:(e,n,o)=>{o.d(n,{A:()=>s});const s=o.p+"assets/images/manage-extensions-ui-566a099ac6d76775e599085e06f02ba4.png"},8453:(e,n,o)=>{o.d(n,{R:()=>l,x:()=>r});var s=o(6540);const t={},i=s.createContext(t);function l(e){const n=s.useContext(i);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:l(e.components),s.createElement(i.Provider,{value:n},e.children)}}}]);