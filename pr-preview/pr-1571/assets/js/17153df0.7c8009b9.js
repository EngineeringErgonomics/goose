"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6082],{6911:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>t,contentTitle:()=>d,default:()=>h,frontMatter:()=>l,metadata:()=>i,toc:()=>c});const i=JSON.parse('{"id":"guides/goose-cli-commands","title":"CLI Commands","description":"Goose provides a command-line interface (CLI) with several commands for managing sessions, configurations and extensions. Below is a list of the available commands and their  descriptions:","source":"@site/docs/guides/goose-cli-commands.md","sourceDirName":"guides","slug":"/guides/goose-cli-commands","permalink":"/goose/pr-preview/pr-1571/docs/guides/goose-cli-commands","draft":false,"unlisted":false,"tags":[],"version":"current","sidebarPosition":4,"frontMatter":{"sidebar_position":4},"sidebar":"tutorialSidebar","previous":{"title":"Using Goosehints","permalink":"/goose/pr-preview/pr-1571/docs/guides/using-goosehints"},"next":{"title":"LLM Rate Limits","permalink":"/goose/pr-preview/pr-1571/docs/guides/handling-llm-rate-limits-with-goose"}}');var o=s(4848),r=s(8453);const l={sidebar_position:4},d="CLI Commands",t={},c=[{value:"Commands",id:"commands",level:2},{value:"help",id:"help",level:3},{value:"configure [options]",id:"configure-options",level:3},{value:"session [options]",id:"session-options",level:3},{value:"info [options]",id:"info-options",level:3},{value:"version",id:"version",level:3},{value:"update [options]",id:"update-options",level:3},{value:"mcp",id:"mcp",level:3},{value:"run [options]",id:"run-options",level:3},{value:"agents",id:"agents",level:3},{value:"Keyboard Shortcuts",id:"keyboard-shortcuts",level:2},{value:"Slash Commands",id:"slash-commands",level:3},{value:"Keyboard Navigation",id:"keyboard-navigation",level:3}];function a(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",header:"header",hr:"hr",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.header,{children:(0,o.jsx)(n.h1,{id:"cli-commands",children:"CLI Commands"})}),"\n",(0,o.jsx)(n.p,{children:"Goose provides a command-line interface (CLI) with several commands for managing sessions, configurations and extensions. Below is a list of the available commands and their  descriptions:"}),"\n",(0,o.jsx)(n.h2,{id:"commands",children:"Commands"}),"\n",(0,o.jsx)(n.h3,{id:"help",children:"help"}),"\n",(0,o.jsx)(n.p,{children:"Used to display the help menu"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose --help\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"configure-options",children:"configure [options]"}),"\n",(0,o.jsx)(n.p,{children:"Configure Goose settings - providers, extensions, etc."}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose configure\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"session-options",children:"session [options]"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:["\n",(0,o.jsx)(n.p,{children:"Start a session and give it a name"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-n, --name <name>"})})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose session --name <name>\n"})}),"\n"]}),"\n",(0,o.jsxs)(n.li,{children:["\n",(0,o.jsx)(n.p,{children:"Resume a previous session"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-r, --resume"})})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose session --resume --name <name>\n"})}),"\n"]}),"\n",(0,o.jsxs)(n.li,{children:["\n",(0,o.jsx)(n.p,{children:"Start a session with the specified extension"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"--with-extension <command>"})})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose session --with-extension <command>\n"})}),"\n",(0,o.jsxs)(n.p,{children:["Can also include environment variables (e.g., ",(0,o.jsx)(n.code,{children:"'GITHUB_TOKEN={your_token} npx -y @modelcontextprotocol/server-github'"}),")"]}),"\n"]}),"\n",(0,o.jsxs)(n.li,{children:["\n",(0,o.jsxs)(n.p,{children:["Start a session with the specified ",(0,o.jsx)(n.a,{href:"/docs/getting-started/using-extensions#built-in-extensions",children:"built-in extension"})," enabled (e.g. 'developer')"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"--with-builtin <id>"})})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose session --with-builtin <id>\n"})}),"\n"]}),"\n"]}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"info-options",children:"info [options]"}),"\n",(0,o.jsx)(n.p,{children:"Shows Goose information, including the version, configuration file location, session storage, and logs."}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-v, --verbose"})}),": (Optional) Show detailed configuration settings, including environment variables and enabled extensions."]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose info\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"version",children:"version"}),"\n",(0,o.jsx)(n.p,{children:"Used to check the current Goose version you have installed"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose --version\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"update-options",children:"update [options]"}),"\n",(0,o.jsx)(n.p,{children:"Update the Goose CLI to a newer version."}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"--canary, -c"})}),": Update to the canary (development) version instead of the stable version"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"--reconfigure, -r"})}),": Forces Goose to reset configuration settings during the update process"]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"# Update to latest stable version\ngoose update\n\n# Update to latest canary version\ngoose update --canary\n\n# Update and reconfigure settings\ngoose update --reconfigure\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"mcp",children:"mcp"}),"\n",(0,o.jsxs)(n.p,{children:["Run an enabled MCP server specified by ",(0,o.jsx)(n.code,{children:"<name>"})," (e.g. 'Google Drive')"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose mcp <name>\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"run-options",children:"run [options]"}),"\n",(0,o.jsx)(n.p,{children:"Execute commands from an instruction file or stdin"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Options:"})}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-i, --instructions <FILE>"})}),": Path to instruction file containing commands"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-t, --text <TEXT>"})}),": Input text to provide to Goose directly"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-n, --name <NAME>"})}),": Name for this run session (e.g., 'daily-tasks')"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"-r, --resume"})}),": Resume from a previous run"]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose run --instructions plan.md\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h3,{id:"agents",children:"agents"}),"\n",(0,o.jsx)(n.p,{children:"Used to show the available implementations of the agent loop itself"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.strong,{children:"Usage:"})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose agents\n"})}),"\n",(0,o.jsx)(n.hr,{}),"\n",(0,o.jsx)(n.h2,{id:"keyboard-shortcuts",children:"Keyboard Shortcuts"}),"\n",(0,o.jsx)(n.p,{children:"Goose CLI supports several shortcuts and built-in commands for easier navigation."}),"\n",(0,o.jsx)(n.h3,{id:"slash-commands",children:"Slash Commands"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsxs)(n.strong,{children:[(0,o.jsx)(n.code,{children:"/exit"})," or ",(0,o.jsx)(n.code,{children:"/quit"})]})," - Exit the session"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"/t"})})," - Toggle between Light/Dark modes"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsxs)(n.strong,{children:[(0,o.jsx)(n.code,{children:"/?"})," or ",(0,o.jsx)(n.code,{children:"/help"})]})," - Display the help menu"]}),"\n"]}),"\n",(0,o.jsx)(n.h3,{id:"keyboard-navigation",children:"Keyboard Navigation"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"Ctrl+C"})})," - Interrupt the current request"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:(0,o.jsx)(n.code,{children:"Ctrl+J"})})," - Add a newline"]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.strong,{children:"Up/Down arrows"})," - Navigate through command history"]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(a,{...e})}):a(e)}},8453:(e,n,s)=>{s.d(n,{R:()=>l,x:()=>d});var i=s(6540);const o={},r=i.createContext(o);function l(e){const n=i.useContext(r);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function d(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:l(e.components),i.createElement(r.Provider,{value:n},e.children)}}}]);