"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[7063],{8717:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>d,contentTitle:()=>r,default:()=>p,frontMatter:()=>a,metadata:()=>t,toc:()=>l});var t=s(615),o=s(4848),i=s(8453);const a={draft:!1,title:"Screenshot-Driven Development",description:"AI Agent uses screenshots to assist in styling.",date:new Date("2024-11-22T00:00:00.000Z"),authors:["rizel"]},r=void 0,d={authorsImageUrls:[void 0]},l=[{value:"My original calendar:",id:"my-original-calendar",level:3},{value:"Goose prototyped the designs below:",id:"goose-prototyped-the-designs-below",level:3},{value:"Get Started with Screenshot-Driven Development",id:"get-started-with-screenshot-driven-development",level:2},{value:"Step 1: Create your UI",id:"step-1-create-your-ui",level:3},{value:"Step 2: Install Goose",id:"step-2-install-goose",level:3},{value:"Step 3: Start a session",id:"step-3-start-a-session",level:3},{value:"Bring your own LLM",id:"bring-your-own-llm",level:4},{value:"Step 4: Enable the Screen toolkit",id:"step-4-enable-the-screen-toolkit",level:3},{value:"Step 5: Prompt Goose to screenshot your UI",id:"step-5-prompt-goose-to-screenshot-your-ui",level:3},{value:"Step 6: Prompt Goose to transform your UI",id:"step-6-prompt-goose-to-transform-your-ui",level:3},{value:"Glassmorphism",id:"glassmorphism",level:4},{value:"Neumorphism",id:"neumorphism",level:4},{value:"Claymorphism",id:"claymorphism",level:4},{value:"Brutalism",id:"brutalism",level:4},{value:"Learn More",id:"learn-more",level:2}];function c(e){const n={a:"a",admonition:"admonition",blockquote:"blockquote",code:"code",h2:"h2",h3:"h3",h4:"h4",img:"img",p:"p",pre:"pre",...(0,i.R)(),...e.components},{Details:t,Head:a}=n;return t||h("Details",!0),a||h("Head",!0),(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"calendar",src:s(1956).A+"",width:"1000",height:"420"})}),"\n",(0,o.jsx)(n.p,{children:"I'm a developer at heart, so when I'm working on a personal project, the hardest part isn't writing code\u2014it's making design decisions. I recently built a calendar user interface. I wanted to enhance its visual appeal, so I researched UI design trends like \"glassmorphism\" and \"claymorphism.\""}),"\n",(0,o.jsxs)(n.p,{children:["However, I didn't want to spend hours implementing the CSS for each design trend, so I developed a faster approach: screenshot-driven development. I used an open source developer agent called ",(0,o.jsx)(n.a,{href:"https://github.com/block/goose",children:"Goose"})," to transform my user interfaces quickly."]}),"\n",(0,o.jsx)(n.admonition,{title:"Goose Beta Version",type:"warning",children:(0,o.jsx)(n.p,{children:"This post was written about a beta version of Goose and the commands and flow may have changed."})}),"\n",(0,o.jsx)(n.h3,{id:"my-original-calendar",children:"My original calendar:"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"calendar",src:s(5657).A+"",width:"1434",height:"1616"})}),"\n",(0,o.jsx)(n.h3,{id:"goose-prototyped-the-designs-below",children:"Goose prototyped the designs below:"}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"Goose prototypes",src:s(3181).A+"",width:"1364",height:"1354"})}),"\n",(0,o.jsx)(n.p,{children:"In this blog post, I'll show you how to quickly prototype design styles by letting Goose handle the CSS for you."}),"\n",(0,o.jsxs)(n.blockquote,{children:["\n",(0,o.jsx)(n.p,{children:"\ud83d\udca1 Note: Your results might look different from my examples - that's part of the fun of generative AI! Each run can produce unique variations of these design trends."}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"get-started-with-screenshot-driven-development",children:"Get Started with Screenshot-Driven Development"}),"\n",(0,o.jsx)(n.h3,{id:"step-1-create-your-ui",children:"Step 1: Create your UI"}),"\n",(0,o.jsx)(n.p,{children:"Let's create a basic UI to experiment with. Create an index.html file with the code below:"}),"\n",(0,o.jsxs)(t,{children:[(0,o.jsx)("summary",{children:"Create an index.html file with the code below"}),(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-html",children:'<!DOCTYPE html>\n<html>\n<head>\n    <style>\n        body {\n            display: flex;\n            justify-content: center;\n            align-items: center;\n            min-height: 100vh;\n            margin: 0;\n            background: linear-gradient(45deg, #6e48aa, #9c27b0);\n            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;\n        }\n\n        .calendar {\n            background: white;\n            border-radius: 12px;\n            box-shadow: 0 5px 20px rgba(0,0,0,0.1);\n            width: 400px;\n            padding: 20px;\n        }\n\n        .header {\n            display: flex;\n            justify-content: space-between;\n            align-items: center;\n            padding-bottom: 20px;\n            border-bottom: 2px solid #f0f0f0;\n        }\n\n        .month {\n            font-size: 24px;\n            font-weight: 600;\n            color: #1a1a1a;\n        }\n\n        .days {\n            display: grid;\n            grid-template-columns: repeat(7, 1fr);\n            gap: 10px;\n            margin-top: 20px;\n            text-align: center;\n        }\n\n        .days-header {\n            display: grid;\n            grid-template-columns: repeat(7, 1fr);\n            gap: 10px;\n            margin-top: 20px;\n            text-align: center;\n        }\n\n        .days-header span {\n            color: #666;\n            font-weight: 500;\n            font-size: 14px;\n        }\n\n        .day {\n            aspect-ratio: 1;\n            display: flex;\n            align-items: center;\n            justify-content: center;\n            border-radius: 50%;\n            font-size: 14px;\n            color: #333;\n            cursor: pointer;\n            transition: all 0.2s;\n        }\n\n        .day:hover {\n            background: #f0f0f0;\n        }\n\n        .day.today {\n            background: #9c27b0;\n            color: white;\n        }\n\n        .day.inactive {\n            color: #ccc;\n        }\n    </style>\n</head>\n<body>\n    <div class="calendar">\n        <div class="header">\n            <div class="month">November 2024</div>\n        </div>\n        <div class="days-header">\n            <span>Sun</span>\n            <span>Mon</span>\n            <span>Tue</span>\n            <span>Wed</span>\n            <span>Thu</span>\n            <span>Fri</span>\n            <span>Sat</span>\n        </div>\n        <div class="days">\n            <div class="day inactive">27</div>\n            <div class="day inactive">28</div>\n            <div class="day inactive">29</div>\n            <div class="day inactive">30</div>\n            <div class="day inactive">31</div>\n            <div class="day">1</div>\n            <div class="day">2</div>\n            <div class="day">3</div>\n            <div class="day">4</div>\n            <div class="day">5</div>\n            <div class="day">6</div>\n            <div class="day">7</div>\n            <div class="day">8</div>\n            <div class="day">9</div>\n            <div class="day">10</div>\n            <div class="day">11</div>\n            <div class="day">12</div>\n            <div class="day">13</div>\n            <div class="day today">14</div>\n            <div class="day">15</div>\n            <div class="day">16</div>\n            <div class="day">17</div>\n            <div class="day">18</div>\n            <div class="day">19</div>\n            <div class="day">20</div>\n            <div class="day">21</div>\n            <div class="day">22</div>\n            <div class="day">23</div>\n            <div class="day">24</div>\n            <div class="day">25</div>\n            <div class="day">26</div>\n            <div class="day">27</div>\n            <div class="day">28</div>\n            <div class="day">29</div>\n            <div class="day">30</div>\n        </div>\n    </div>\n</body>\n</html>\n'})})]}),"\n",(0,o.jsx)(n.p,{children:"Once saved, open the file in your browser. You should see a calendar!"}),"\n",(0,o.jsx)(n.h3,{id:"step-2-install-goose",children:"Step 2: Install Goose"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"brew install pipx\npipx ensurepath\npipx install goose-ai\n"})}),"\n",(0,o.jsx)(n.h3,{id:"step-3-start-a-session",children:"Step 3: Start a session"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"goose session start\n"})}),"\n",(0,o.jsx)(n.h4,{id:"bring-your-own-llm",children:"Bring your own LLM"}),"\n",(0,o.jsxs)(n.blockquote,{children:["\n",(0,o.jsx)(n.p,{children:"Goose will prompt you to set up your API key when you first run this command. You can use various LLM providers like OpenAI or Anthropic"}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"export OPENAI_API_KEY=your_api_key\n# Or for other providers:\nexport ANTHROPIC_API_KEY=your_api_key\n"})}),"\n",(0,o.jsx)(n.h3,{id:"step-4-enable-the-screen-toolkit",children:"Step 4: Enable the Screen toolkit"}),"\n",(0,o.jsxs)(n.p,{children:["Goose uses ",(0,o.jsx)(n.a,{href:"https://block.github.io/goose/plugins/plugins.html",children:"toolkits"})," to extend its capabilities. The ",(0,o.jsx)(n.a,{href:"https://block.github.io/goose/plugins/available-toolkits.html#6-screen-toolkit",children:"screen"})," toolkit lets Goose take and analyze screenshots."]}),"\n",(0,o.jsx)(n.p,{children:"To enable the Screen toolkit, add it to your Goose profile at ~/.config/goose/profiles.yaml."}),"\n",(0,o.jsxs)(n.blockquote,{children:["\n",(0,o.jsx)(n.p,{children:"Your configuration might look slightly different depending on your LLM provider preferences."}),"\n"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-yaml",children:"default:\n  provider: openai\n  processor: gpt-4o\n  accelerator: gpt-4o-mini\n  moderator: truncate\n  toolkits:\n  - name: developer\n    requires: {}\n  - name: screen\n    requires: {}\n"})}),"\n",(0,o.jsx)(n.h3,{id:"step-5-prompt-goose-to-screenshot-your-ui",children:"Step 5: Prompt Goose to screenshot your UI"}),"\n",(0,o.jsx)(n.p,{children:"Goose analyzes your UI through screenshots to understand its structure and elements. In your Gooses session, prompt Goose to take a screenshot by specifying which display your UI is on:"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"Take a screenshot of display(1)  \n"})}),"\n",(0,o.jsxs)(n.blockquote,{children:["\n",(0,o.jsx)(n.p,{children:"The display number is required - use display(1) for your main monitor or display(2) for a secondary monitor."}),"\n"]}),"\n",(0,o.jsxs)(n.p,{children:["Upon success, Goose will run a ",(0,o.jsx)(n.code,{children:"screencapture"})," command and save it as a temporary file."]}),"\n",(0,o.jsx)(n.h3,{id:"step-6-prompt-goose-to-transform-your-ui",children:"Step 6: Prompt Goose to transform your UI"}),"\n",(0,o.jsx)(n.p,{children:"Now, you can ask Goose to apply different design styles. Here are some of the prompts I gave Goose and the results it produced:"}),"\n",(0,o.jsx)(n.h4,{id:"glassmorphism",children:"Glassmorphism"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"Apply a glassmorphic effect to my UI\n"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"glassmorphism",src:s(6401).A+"",width:"1428",height:"1414"})}),"\n",(0,o.jsx)(n.h4,{id:"neumorphism",children:"Neumorphism"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"Apply neumorphic effects to my calendar and the dates\n"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"neumorphism",src:s(2469).A+"",width:"1434",height:"1332"})}),"\n",(0,o.jsx)(n.h4,{id:"claymorphism",children:"Claymorphism"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"Please replace with a claymorphic effect\n"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"claymorphism",src:s(2220).A+"",width:"1432",height:"1400"})}),"\n",(0,o.jsx)(n.h4,{id:"brutalism",children:"Brutalism"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"Apply a brutalist effect please\n"})}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"brutalism",src:s(2999).A+"",width:"1358",height:"1344"})}),"\n",(0,o.jsx)(n.h2,{id:"learn-more",children:"Learn More"}),"\n",(0,o.jsx)(n.p,{children:"Developing user interfaces is a blend of creativity and problem-solving. And I love that using Goose gives me more time to focus on creativity rather than wrestling with CSS for hours."}),"\n",(0,o.jsx)(n.p,{children:"Beyond prototyping, Goose's ability to analyze screenshots can help developers identify and resolve UI bugs."}),"\n",(0,o.jsxs)(n.p,{children:["If you're interested in learning more, check out the ",(0,o.jsx)(n.a,{href:"https://github.com/block/goose",children:"Goose repo"})," and join our ",(0,o.jsx)(n.a,{href:"https://discord.gg/block-opensource",children:"Discord community"}),"."]}),"\n",(0,o.jsxs)(a,{children:[(0,o.jsx)("meta",{property:"og:title",content:"Screenshot-Driven Development"}),(0,o.jsx)("meta",{property:"og:type",content:"article"}),(0,o.jsx)("meta",{property:"og:url",content:"https://block.github.io/goose/blog/2024/11/22/screenshot-driven-development"}),(0,o.jsx)("meta",{property:"og:description",content:"AI Agent uses screenshots to assist in styling."}),(0,o.jsx)("meta",{property:"og:image",content:"https://block.github.io/goose/assets/images/screenshot-driven-development-4ed1beaa10c6062c0bf87e2d27590ad6.png"}),(0,o.jsx)("meta",{name:"twitter:card",content:"summary_large_image"}),(0,o.jsx)("meta",{property:"twitter:domain",content:"block.github.io/goose"}),(0,o.jsx)("meta",{name:"twitter:title",content:"Screenshot-Driven Development"}),(0,o.jsx)("meta",{name:"twitter:description",content:"AI Agent uses screenshots to assist in styling."}),(0,o.jsx)("meta",{name:"twitter:image",content:"https://block.github.io/goose/assets/images/screenshot-driven-development-4ed1beaa10c6062c0bf87e2d27590ad6.png"})]})]})}function p(e={}){const{wrapper:n}={...(0,i.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(c,{...e})}):c(e)}function h(e,n){throw new Error("Expected "+(n?"component":"object")+" `"+e+"` to be defined: you likely forgot to import, pass, or provide it.")}},2999:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/brutalism-calendar-df62e8a95de08f35dabfa2552f937af0.png"},2220:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/claymorphism-calendar-5b17a874ff38f094d6e4ea180c56c551.png"},6401:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/glassmorphism-calendar-43f14e391e706ee2254d8289a9ba90f1.png"},3181:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/goose-prototypes-calendar-8ff9719a4fa565eab4fcdb2389fdde56.png"},2469:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/neumorphism-calendar-50b2536c0ecf7410320e7ad81005d1b7.png"},5657:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/screenshot-calendar-og-fbaf4b2c4843cce8e2f0ea735c04676b.png"},1956:(e,n,s)=>{s.d(n,{A:()=>t});const t=s.p+"assets/images/screenshot-driven-development-4ed1beaa10c6062c0bf87e2d27590ad6.png"},8453:(e,n,s)=>{s.d(n,{R:()=>a,x:()=>r});var t=s(6540);const o={},i=t.createContext(o);function a(e){const n=t.useContext(i);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:a(e.components),t.createElement(i.Provider,{value:n},e.children)}},615:e=>{e.exports=JSON.parse('{"permalink":"/goose/pr-preview/pr-1420/blog/2024/11/22/screenshot-driven-development","source":"@site/blog/2024-11-22-screenshot-driven-development/index.md","title":"Screenshot-Driven Development","description":"AI Agent uses screenshots to assist in styling.","date":"2024-11-22T00:00:00.000Z","tags":[],"readingTime":4.485,"hasTruncateMarker":true,"authors":[{"name":"Rizel Scarlett","title":"Staff Developer Advocate","page":{"permalink":"/goose/pr-preview/pr-1420/blog/authors/rizel"},"socials":{"x":"https://x.com/blackgirlbytes","github":"https://github.com/blackgirlbytes"},"imageURL":"https://avatars.githubusercontent.com/u/22990146?v=4","key":"rizel"}],"frontMatter":{"draft":false,"title":"Screenshot-Driven Development","description":"AI Agent uses screenshots to assist in styling.","date":"2024-11-22T00:00:00.000Z","authors":["rizel"]},"unlisted":false,"prevItem":{"title":"Previewing Goose v1.0 Beta","permalink":"/goose/pr-preview/pr-1420/blog/2024/12/06/previewing-goose-v10-beta"}}')}}]);