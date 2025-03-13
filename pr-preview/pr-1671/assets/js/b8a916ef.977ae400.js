"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[5355],{3850:(e,o,n)=>{n.r(o),n.d(o,{assets:()=>d,contentTitle:()=>c,default:()=>u,frontMatter:()=>r,metadata:()=>s,toc:()=>a});const s=JSON.parse('{"id":"guides/goose-in-docker","title":"Goose in Docker","description":"There are various scenarios where you might want to build Goose in Docker. If the instructions below do not meet your needs, please contact us by replying to our discussion topic.","source":"@site/docs/guides/goose-in-docker.md","sourceDirName":"guides","slug":"/guides/goose-in-docker","permalink":"/goose/pr-preview/pr-1671/docs/guides/goose-in-docker","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":9,"frontMatter":{"title":"Goose in Docker","sidebar_position":9},"sidebar":"tutorialSidebar","previous":{"title":"Using Gooseignore","permalink":"/goose/pr-preview/pr-1671/docs/guides/using-gooseignore"},"next":{"title":"Experimental Features","permalink":"/goose/pr-preview/pr-1671/docs/guides/experimental-features"}}');var i=n(4848),t=n(8453);const r={title:"Goose in Docker",sidebar_position:9},c="Building Goose in Docker",d={},a=[];function l(e){const o={a:"a",admonition:"admonition",code:"code",h1:"h1",header:"header",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.R)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(o.header,{children:(0,i.jsx)(o.h1,{id:"building-goose-in-docker",children:"Building Goose in Docker"})}),"\n",(0,i.jsx)(o.admonition,{title:"Tell Us What You Need",type:"info",children:(0,i.jsxs)(o.p,{children:["There are various scenarios where you might want to build Goose in Docker. If the instructions below do not meet your needs, please contact us by replying to our ",(0,i.jsx)(o.a,{href:"https://github.com/block/goose/discussions/1496",children:"discussion topic"}),"."]})}),"\n",(0,i.jsx)(o.p,{children:"You can build Goose from the source file within a Docker container. This approach not only provides security benefits by creating an isolated environment but also enhances consistency and portability. For example, if you need to troubleshoot an error on a platform you don't usually work with (such as Ubuntu), you can easily debug it using Docker."}),"\n",(0,i.jsxs)(o.p,{children:["To begin, you will need to modify the ",(0,i.jsx)(o.code,{children:"Dockerfile"})," and ",(0,i.jsx)(o.code,{children:"docker-compose.yml"})," files to suit your requirements. Some changes you might consider include:"]}),"\n",(0,i.jsxs)(o.ul,{children:["\n",(0,i.jsxs)(o.li,{children:["\n",(0,i.jsxs)(o.p,{children:[(0,i.jsx)(o.strong,{children:"Required:"})," Setting your API key, provider, and model in the ",(0,i.jsx)(o.code,{children:"docker-compose.yml"})," file as environment variables because the keyring settings do not work on Ubuntu in Docker. This example uses the Google API key and its corresponding settings, but you can ",(0,i.jsx)(o.a,{href:"https://github.com/block/goose/blob/main/ui/desktop/src/components/settings/models/hardcoded_stuff.tsx",children:"find your own list of API keys"})," and the ",(0,i.jsx)(o.a,{href:"https://github.com/block/goose/blob/main/ui/desktop/src/components/settings/models/hardcoded_stuff.tsx",children:"corresponding settings"}),"."]}),"\n"]}),"\n",(0,i.jsxs)(o.li,{children:["\n",(0,i.jsxs)(o.p,{children:[(0,i.jsx)(o.strong,{children:"Optional:"})," Changing the base image to a different Linux distribution in the ",(0,i.jsx)(o.code,{children:"Dockerfile"}),". This example uses Ubuntu, but you can switch to another distribution such as CentOS, Fedora, or Alpine."]}),"\n"]}),"\n",(0,i.jsxs)(o.li,{children:["\n",(0,i.jsxs)(o.p,{children:[(0,i.jsx)(o.strong,{children:"Optional:"})," Mounting your personal Goose settings and hints files in the ",(0,i.jsx)(o.code,{children:"docker-compose.yml"})," file. This allows you to use your personal settings and hints files within the Docker container."]}),"\n"]}),"\n"]}),"\n",(0,i.jsx)(o.p,{children:"After setting the credentials, you can build the Docker image using the following command:"}),"\n",(0,i.jsx)(o.pre,{children:(0,i.jsx)(o.code,{className:"language-bash",children:"docker-compose -f documentation/docs/docker/docker-compose.yml build\n"})}),"\n",(0,i.jsx)(o.p,{children:"Next, run the container and connect to it using the following command:"}),"\n",(0,i.jsx)(o.pre,{children:(0,i.jsx)(o.code,{className:"language-bash",children:"docker-compose -f documentation/docs/docker/docker-compose.yml run --rm goose-cli\n"})}),"\n",(0,i.jsx)(o.p,{children:"Inside the container, run the following command to configure Goose:"}),"\n",(0,i.jsx)(o.pre,{children:(0,i.jsx)(o.code,{className:"language-bash",children:"goose configure\n"})}),"\n",(0,i.jsxs)(o.p,{children:["When prompted to save the API key to the keyring, select ",(0,i.jsx)(o.code,{children:"No"}),", as you are already passing the API key as an environment variable."]}),"\n",(0,i.jsxs)(o.p,{children:["Configure Goose a second time, and this time, you can ",(0,i.jsx)(o.a,{href:"/docs/getting-started/using-extensions",children:"add any extensions"})," you need."]}),"\n",(0,i.jsx)(o.p,{children:"After that, you can start a session:"}),"\n",(0,i.jsx)(o.pre,{children:(0,i.jsx)(o.code,{className:"language-bash",children:"goose session\n"})}),"\n",(0,i.jsx)(o.p,{children:"You should now be able to connect to Goose with your configured extensions enabled."})]})}function u(e={}){const{wrapper:o}={...(0,t.R)(),...e.components};return o?(0,i.jsx)(o,{...e,children:(0,i.jsx)(l,{...e})}):l(e)}},8453:(e,o,n)=>{n.d(o,{R:()=>r,x:()=>c});var s=n(6540);const i={},t=s.createContext(i);function r(e){const o=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(o):{...o,...e}}),[o,e])}function c(e){let o;return o=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),s.createElement(t.Provider,{value:o},e.children)}}}]);