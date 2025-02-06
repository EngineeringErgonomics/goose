"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[884],{3948:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>d,contentTitle:()=>c,default:()=>p,frontMatter:()=>a,metadata:()=>o,toc:()=>u});const o=JSON.parse('{"id":"tutorials/github-mcp","title":"GitHub Extension","description":"Add GitHub MCP Server as a Goose Extension","source":"@site/docs/tutorials/github-mcp.md","sourceDirName":"tutorials","slug":"/tutorials/github-mcp","permalink":"/goose/pr-preview/pr-1125/docs/tutorials/github-mcp","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"GitHub Extension","description":"Add GitHub MCP Server as a Goose Extension"},"sidebar":"tutorialSidebar","previous":{"title":"Developer Extension","permalink":"/goose/pr-preview/pr-1125/docs/tutorials/developer-mcp"},"next":{"title":"JetBrains Extension","permalink":"/goose/pr-preview/pr-1125/docs/tutorials/jetbrains-mcp"}}');var s=t(4848),r=t(8453),i=t(5537),l=t(9329);const a={title:"GitHub Extension",description:"Add GitHub MCP Server as a Goose Extension"},c=void 0,d={},u=[{value:"Configuration",id:"configuration",level:2},{value:"Example Usage",id:"example-usage",level:2},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3}];function h(e){const n={a:"a",admonition:"admonition",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsxs)(n.p,{children:["This tutorial covers how to add the ",(0,s.jsx)(n.a,{href:"https://github.com/modelcontextprotocol/servers/tree/main/src/github",children:"GitHub MCP Server"})," as a Goose extension to enable file operations, repository management, search functionality, and more."]}),"\n",(0,s.jsxs)(n.admonition,{title:"TLDR",type:"tip",children:[(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Command"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"npx -y @modelcontextprotocol/server-github\n"})}),(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Environment Variable"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"GITHUB_PERSONAL_ACCESS_TOKEN: <YOUR_TOKEN>\n"})})]}),"\n",(0,s.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,s.jsxs)(i.A,{groupId:"interface",children:[(0,s.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Run the ",(0,s.jsx)(n.code,{children:"configure"})," command:"]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,s.jsxs)(n.ol,{start:"2",children:["\n",(0,s.jsxs)(n.li,{children:["Choose to add a ",(0,s.jsx)(n.code,{children:"Command-line Extension"})]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c6  What type of extension would you like to add?\n  \u2502  \u25cb Built-in Extension \n  // highlight-start    \n  \u2502  \u25cf Command-line Extension (Run a local command or script)\n  // highlight-end    \n  \u2502  \u25cb Remote Extension \n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"3",children:["\n",(0,s.jsx)(n.li,{children:"Give your extension a name"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  // highlight-start\n  \u25c6  What would you like to call this extension?\n  \u2502  github\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"4",children:["\n",(0,s.jsx)(n.li,{children:"Enter the command"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  \u25c7  What would you like to call this extension?\n  \u2502  github\n  \u2502\n  // highlight-start\n  \u25c6  What command should be run?\n  \u2502  npx -y @modelcontextprotocol/server-github\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"5",children:["\n",(0,s.jsxs)(n.li,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://github.com/settings/personal-access-tokens",children:"GitHub Personal Access Token"})," and paste it in."]}),"\n"]}),(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsx)(n.p,{children:"When creating your access token, you can specify the repositories and granular permissions you'd like Goose to have access to."})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:" \u250c   goose-configure \n \u2502\n \u25c7  What would you like to configure?\n \u2502  Add Extension \n \u2502\n \u25c7  What type of extension would you like to add?\n \u2502  Command-line Extension \n \u2502\n \u25c7  What would you like to call this extension?\n \u2502  github\n \u2502\n \u25c7  What command should be run?\n \u2502  npx -y @modelcontextprotocol/server-github\n // highlight-start\n \u25c6  Would you like to add environment variables?\n \u2502  Yes \n \u2502\n \u25c7  Environment variable name:\n \u2502  GITHUB_PERSONAL_ACCESS_TOKEN\n \u2502\n \u25c7  Environment variable value:\n \u2502  \u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\n \u2502\n \u25c7  Add another environment variable?\n \u2502  No \n // highlight-end\n \u2514  Added github extension\n"})})]}),(0,s.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Settings"})]}),"\n",(0,s.jsxs)(n.li,{children:["Under ",(0,s.jsx)(n.code,{children:"Extensions"}),", click the ",(0,s.jsx)(n.code,{children:"Add"})," link"]}),"\n",(0,s.jsxs)(n.li,{children:["On the ",(0,s.jsx)(n.code,{children:"Add Extension Manually"})," modal, enter the following:","\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Type"}),": ",(0,s.jsx)(n.code,{children:"Standard IO"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"ID"}),": ",(0,s.jsx)(n.code,{children:"gh-mcp"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"github"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Description"}),": ",(0,s.jsx)(n.code,{children:"GitHub MCP Server"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Command"}),": ",(0,s.jsx)(n.code,{children:"npx -y @modelcontextprotocol/server-github"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Environment Variables"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"GITHUB_PERSONAL_ACCESS_TOKEN"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Value"}),": (",(0,s.jsxs)(n.em,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://github.com/settings/personal-access-tokens",children:"GitHub Personal Access Token"})," and paste it in."]}),")","\n",(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsx)(n.p,{children:"When creating your access token, you can specify the repositories and granular permissions you'd like Goose to have access to."})}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add"})," button"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add Extension"})," button"]}),"\n"]})})]}),"\n",(0,s.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,s.jsxs)(n.p,{children:["When creating my ",(0,s.jsx)(n.a,{href:"https://github.com/settings/personal-access-tokens",children:"GitHub Personal Access Token"}),", I gave it the following access:"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:["Repository: ",(0,s.jsx)(n.a,{href:"https://github.com/angiejones/goose-demo",children:"goose-demo"})]}),"\n",(0,s.jsx)(n.li,{children:"Read access to metadata"}),"\n",(0,s.jsx)(n.li,{children:"Read and Write access to code, commit status, issues, and pull requests"}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:'create a new branch called hello-world in my angiejones/goose-demo repository. Update the README.md file to say "this was written by goose" and commit it. Open a pull request with your changes.\n'})}),"\n",(0,s.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:'I\'ll help you create a new branch, update the README, and open a pull request. Let me break this down into steps:\n\n1. First, let\'s create the new branch called "hello-world" in your repository\n2. Then update the README.md file\n3. Finally, create a pull request\n\nLet me execute these steps:\n\n1. Creating the new branch:\n\n\u2500\u2500\u2500 create_branch | github \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nbranch: hello-world\nowner: angiejones\nrepo: goose-demo\n\n2. Now, updating the README.md file in the new branch:\n\n\u2500\u2500\u2500 create_or_update_file | github \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nbranch: hello-world\ncontent: this was written by goose\nmessage: Update README.md with Goose message\nowner: angiejones\npath: README.md\nrepo: goose-demo\n\n\n\n3. Finally, creating the pull request:\n\n\u2500\u2500\u2500 create_pull_request | github \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nbase: main\nbody: ...\nhead: hello-world\nowner: angiejones\nrepo: goose-demo\ntitle: Update README.md\n\n\n\nGreat! I\'ve completed all the requested tasks:\n1. Created a new branch called "hello-world"\n2. Updated the README.md file with the text "this was written by goose"\n3. Created a pull request to merge these changes into the main branch\n\nYou can view the pull request at: https://github.com/angiejones/goose-demo/pull/1\n\nThe pull request is now ready for your review. Would you like me to do anything else with it?\n'})})]})}function p(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},9329:(e,n,t)=>{t.d(n,{A:()=>i});t(6540);var o=t(4164);const s={tabItem:"tabItem_Ymn6"};var r=t(4848);function i(e){let{children:n,hidden:t,className:i}=e;return(0,r.jsx)("div",{role:"tabpanel",className:(0,o.A)(s.tabItem,i),hidden:t,children:n})}},5537:(e,n,t)=>{t.d(n,{A:()=>w});var o=t(6540),s=t(4164),r=t(5627),i=t(6347),l=t(372),a=t(604),c=t(1861),d=t(8749);function u(e){return o.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,o.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:t}=e;return(0,o.useMemo)((()=>{const e=n??function(e){return u(e).map((e=>{let{props:{value:n,label:t,attributes:o,default:s}}=e;return{value:n,label:t,attributes:o,default:s}}))}(t);return function(e){const n=(0,c.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,t])}function p(e){let{value:n,tabValues:t}=e;return t.some((e=>e.value===n))}function m(e){let{queryString:n=!1,groupId:t}=e;const s=(0,i.W6)(),r=function(e){let{queryString:n=!1,groupId:t}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!t)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return t??null}({queryString:n,groupId:t});return[(0,a.aZ)(r),(0,o.useCallback)((e=>{if(!r)return;const n=new URLSearchParams(s.location.search);n.set(r,e),s.replace({...s.location,search:n.toString()})}),[r,s])]}function x(e){const{defaultValue:n,queryString:t=!1,groupId:s}=e,r=h(e),[i,a]=(0,o.useState)((()=>function(e){let{defaultValue:n,tabValues:t}=e;if(0===t.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!p({value:n,tabValues:t}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${t.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const o=t.find((e=>e.default))??t[0];if(!o)throw new Error("Unexpected error: 0 tabValues");return o.value}({defaultValue:n,tabValues:r}))),[c,u]=m({queryString:t,groupId:s}),[x,g]=function(e){let{groupId:n}=e;const t=function(e){return e?`docusaurus.tab.${e}`:null}(n),[s,r]=(0,d.Dv)(t);return[s,(0,o.useCallback)((e=>{t&&r.set(e)}),[t,r])]}({groupId:s}),b=(()=>{const e=c??x;return p({value:e,tabValues:r})?e:null})();(0,l.A)((()=>{b&&a(b)}),[b]);return{selectedValue:i,selectValue:(0,o.useCallback)((e=>{if(!p({value:e,tabValues:r}))throw new Error(`Can't select invalid tab value=${e}`);a(e),u(e),g(e)}),[u,g,r]),tabValues:r}}var g=t(9136);const b={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var j=t(4848);function f(e){let{className:n,block:t,selectedValue:o,selectValue:i,tabValues:l}=e;const a=[],{blockElementScrollPositionUntilNextRender:c}=(0,r.a_)(),d=e=>{const n=e.currentTarget,t=a.indexOf(n),s=l[t].value;s!==o&&(c(n),i(s))},u=e=>{let n=null;switch(e.key){case"Enter":d(e);break;case"ArrowRight":{const t=a.indexOf(e.currentTarget)+1;n=a[t]??a[0];break}case"ArrowLeft":{const t=a.indexOf(e.currentTarget)-1;n=a[t]??a[a.length-1];break}}n?.focus()};return(0,j.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,s.A)("tabs",{"tabs--block":t},n),children:l.map((e=>{let{value:n,label:t,attributes:r}=e;return(0,j.jsx)("li",{role:"tab",tabIndex:o===n?0:-1,"aria-selected":o===n,ref:e=>{a.push(e)},onKeyDown:u,onClick:d,...r,className:(0,s.A)("tabs__item",b.tabItem,r?.className,{"tabs__item--active":o===n}),children:t??n},n)}))})}function v(e){let{lazy:n,children:t,selectedValue:r}=e;const i=(Array.isArray(t)?t:[t]).filter(Boolean);if(n){const e=i.find((e=>e.props.value===r));return e?(0,o.cloneElement)(e,{className:(0,s.A)("margin-top--md",e.props.className)}):null}return(0,j.jsx)("div",{className:"margin-top--md",children:i.map(((e,n)=>(0,o.cloneElement)(e,{key:n,hidden:e.props.value!==r})))})}function y(e){const n=x(e);return(0,j.jsxs)("div",{className:(0,s.A)("tabs-container",b.tabList),children:[(0,j.jsx)(f,{...n,...e}),(0,j.jsx)(v,{...n,...e})]})}function w(e){const n=(0,g.A)();return(0,j.jsx)(y,{...e,children:u(e.children)},String(n))}},8453:(e,n,t)=>{t.d(n,{R:()=>i,x:()=>l});var o=t(6540);const s={},r=o.createContext(s);function i(e){const n=o.useContext(r);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:i(e.components),o.createElement(r.Provider,{value:n},e.children)}}}]);