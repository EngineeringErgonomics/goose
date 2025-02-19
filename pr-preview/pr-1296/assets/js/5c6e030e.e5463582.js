"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[6262],{1898:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>h,contentTitle:()=>c,default:()=>p,frontMatter:()=>d,metadata:()=>t,toc:()=>m});const t=JSON.parse('{"id":"tutorials/figma-mcp","title":"Figma Extension","description":"Add Figma MCP Server as a Goose Extension","source":"@site/docs/tutorials/figma-mcp.md","sourceDirName":"tutorials","slug":"/tutorials/figma-mcp","permalink":"/goose/pr-preview/pr-1296/docs/tutorials/figma-mcp","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Figma Extension","description":"Add Figma MCP Server as a Goose Extension"},"sidebar":"tutorialSidebar","previous":{"title":"Fetch Extension","permalink":"/goose/pr-preview/pr-1296/docs/tutorials/fetch-mcp"},"next":{"title":"GitHub Extension","permalink":"/goose/pr-preview/pr-1296/docs/tutorials/github-mcp"}}');var s=i(4848),o=i(8453),a=i(5537),r=i(9329),l=i(5887);const d={title:"Figma Extension",description:"Add Figma MCP Server as a Goose Extension"},c=void 0,h={},m=[{value:"Configuration",id:"configuration",level:2},{value:"Example Usage",id:"example-usage",level:2},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3},{value:"Result",id:"result",level:3}];function g(e){const n={a:"a",admonition:"admonition",code:"code",em:"em",h2:"h2",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.R)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(l.A,{videoUrl:"https://www.youtube.com/embed/vHK9Xg_d6Sk"}),"\n",(0,s.jsxs)(n.p,{children:["This tutorial covers how to add the ",(0,s.jsx)(n.a,{href:"https://github.com/hapins/figma-mcp",children:"Figma MCP Server"})," as a Goose extension to enable interaction with Figma files, designs, and components."]}),"\n",(0,s.jsxs)(n.admonition,{title:"TLDR",type:"tip",children:[(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Command"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"npx @hapins/figma-mcp\n"})}),(0,s.jsx)(n.p,{children:(0,s.jsx)(n.strong,{children:"Environment Variable"})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"FIGMA_ACCESS_TOKEN: <YOUR_TOKEN>\n"})})]}),"\n",(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsxs)(n.p,{children:["Note that you'll need ",(0,s.jsx)(n.a,{href:"https://nodejs.org/",children:"Node.js"})," installed on your system to run this command, as it uses npx."]})}),"\n",(0,s.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,s.jsxs)(a.A,{groupId:"interface",children:[(0,s.jsxs)(r.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Run the ",(0,s.jsx)(n.code,{children:"configure"})," command:"]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,s.jsxs)(n.ol,{start:"2",children:["\n",(0,s.jsxs)(n.li,{children:["Choose to add a ",(0,s.jsx)(n.code,{children:"Command-line Extension"})]}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c6  What type of extension would you like to add?\n  \u2502  \u25cb Built-in Extension \n  // highlight-start    \n  \u2502  \u25cf Command-line Extension (Run a local command or script)\n  // highlight-end    \n  \u2502  \u25cb Remote Extension \n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"3",children:["\n",(0,s.jsx)(n.li,{children:"Give your extension a name"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  // highlight-start\n  \u25c6  What would you like to call this extension?\n  \u2502  figma\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"4",children:["\n",(0,s.jsx)(n.li,{children:"Enter the command"}),"\n"]}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  \u25c7  What would you like to call this extension?\n  \u2502  figma\n  \u2502\n  // highlight-start\n  \u25c6  What command should be run?\n  \u2502  npx @hapins/figma-mcp\n  // highlight-end\n  \u2514 \n"})}),(0,s.jsxs)(n.ol,{start:"5",children:["\n",(0,s.jsxs)(n.li,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://www.figma.com/developers/api#access-tokens",children:"Figma Access Token"})," and paste it in."]}),"\n"]}),(0,s.jsx)(n.admonition,{type:"info",children:(0,s.jsx)(n.p,{children:"You can generate an access token from your Figma account settings under the Personal access tokens section."})}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-sh",children:" \u250c   goose-configure \n \u2502\n \u25c7  What would you like to configure?\n \u2502  Add Extension \n \u2502\n \u25c7  What type of extension would you like to add?\n \u2502  Command-line Extension \n \u2502\n \u25c7  What would you like to call this extension?\n \u2502  figma\n \u2502\n \u25c7  What command should be run?\n \u2502  npx @hapins/figma-mcp\n // highlight-start\n \u25c6  Would you like to add environment variables?\n \u2502  Yes \n \u2502\n \u25c7  Environment variable name:\n \u2502  FIGMA_ACCESS_TOKEN\n \u2502\n \u25c7  Environment variable value:\n \u2502  \u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\u25aa\n \u2502\n \u25c7  Add another environment variable?\n \u2502  No \n // highlight-end\n \u2514  Added figma extension\n"})})]}),(0,s.jsx)(r.A,{value:"ui",label:"Goose Desktop",children:(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Settings"})]}),"\n",(0,s.jsxs)(n.li,{children:["Under ",(0,s.jsx)(n.code,{children:"Extensions"}),", click the ",(0,s.jsx)(n.code,{children:"Add"})," link"]}),"\n",(0,s.jsxs)(n.li,{children:["On the ",(0,s.jsx)(n.code,{children:"Add Extension Manually"})," modal, enter the following:","\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Type"}),": ",(0,s.jsx)(n.code,{children:"Standard IO"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"ID"}),": ",(0,s.jsx)(n.code,{children:"figma-mcp"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"Figma"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Description"}),": ",(0,s.jsx)(n.code,{children:"Figma MCP Server"})," (",(0,s.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Command"}),": ",(0,s.jsx)(n.code,{children:"npx @hapins/figma-mcp"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Environment Variables"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Name"}),": ",(0,s.jsx)(n.code,{children:"FIGMA_ACCESS_TOKEN"})]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.strong,{children:"Value"}),": (",(0,s.jsxs)(n.em,{children:["Obtain a ",(0,s.jsx)(n.a,{href:"https://www.figma.com/developers/api#access-tokens",children:"Figma Access Token"})," and paste it in."]}),")"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add"})," button"]}),"\n"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["Click ",(0,s.jsx)(n.code,{children:"Add Extension"})," button"]}),"\n"]})})]}),"\n",(0,s.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,s.jsx)(n.p,{children:"The Figma MCP server allows you to interact with your Figma files and designs programmatically. When creating your access token, make sure you give it the appropriate access and scope:"}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsx)(n.li,{children:"Read access for files"}),"\n",(0,s.jsx)(n.li,{children:"Write access for Comments"}),"\n",(0,s.jsx)(n.li,{children:"And any other scopes of your choice."}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:["Given this ",(0,s.jsx)(n.a,{href:"https://www.figma.com/community/file/1037030891378807455",children:"Community Figma design"}),", we'll have Goose build a webpage."]}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{alt:"Figma Design",src:i(403).A+"",width:"3600",height:"2062"})}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsxs)(n.p,{children:["Get the Figma File ID from the URL. For example, ",(0,s.jsx)(n.code,{children:"https://www.figma.com/file/KvR8Wpe6ITaCD3et3H7l6k/AI-Landing-Page"})," has a file ID of ",(0,s.jsx)(n.code,{children:"KvR8Wpe6ITaCD3et3H7l6k"}),"."]})}),"\n",(0,s.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"( O)> Convert my figma design with file ID 7r0mLHWx2rrjwn79mcNVNE into a webpage. Get specific detailed information and styling about each element and components within each frame. Make sure to implement any interactions or animations and use a placeholder div with a gray background for images. \n\nSome elements are centralized, some are duplicated, make sure to get them correct. And ensure correct alignment and spacing between elements.\n"})}),"\n",(0,s.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{children:"I'll help you convert your Figma design into a webpage. Let's break this down into steps and\n gather all the necessary information from your Figma file.\n\n1. First, let's get the file information to understand the overall structure:\n\n\u2500\u2500\u2500 get_file_info | figma \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\ndepth: 4\nfile_key: 7r0mLHWx2rrjwn79mcNVNE\n\n\n\nLet me help you convert this Figma design into a webpage. I'll create the HTML and CSS based\n on the design specifications from the Figma file. I'll break it down into sections and impl\nement all the styling and interactions.\n\nLet's create the files:\n\n\u2500\u2500\u2500 text_editor | developer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\npath: ~/D/w/l/m/figma-example/index.html\ncommand: write\nfile_text: ...\n\n\n### .../figma-example/index.html\n\n--- HTML content hidden ---\n\nNow, let's create the CSS file with all the styling:\n\n\u2500\u2500\u2500 text_editor | developer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\npath: ~/D/w/l/m/figma-example/styles.css\ncommand: write\nfile_text: ...\n\n\n### .../figma-example/styles.css\n\n--- CSS content hidden ---\n\nI've created a complete implementation of the Figma design with the following features and c\nonsiderations:\n\n1. **Typography**:\n- Used the exact fonts from the design: Poppins, Overpass, Nunito Sans, Public Sans, and Ral\neway\n- Implemented correct font weights, sizes, and letter spacing\n- Maintained the text hierarchy as shown in the design\n\n2. **Colors**:\n- Used the exact color codes from the Figma file\n- Implemented the proper background colors (#250b71 for header and footer)\n- Used correct text colors for different sections\n\n3. **Layout**:\n- Created a responsive layout that matches the design\n- Implemented proper spacing and alignment\n- Used CSS Grid for the book cards\n- Used Flexbox for navigation and content alignment\n\n4. **Components**:\n- Navigation bar with social media links\n- About Me section with placeholder for image\n- Education section with ordered list\n- Book cards with proper styling\n- Mood section with image placeholders\n- Footer with copyright text\n\n5. **Interactive Elements**:\n- Hover effects on navigation links\n- \"Next Section\" buttons with icons\n- \"Go on Top\" button at the bottom\n\n6. **Responsive Design**:\n- Added media queries for different screen sizes\n- Responsive adjustments for mobile and tablet views\n- Flexible layouts that adapt to different screen sizes\n\n7. **Images**:\n- Used placeholder divs with background colors for images\n- Maintained aspect ratios from the original design\n\n8. **Accessibility**:\n- Semantic HTML structure\n- Proper heading hierarchy\n- Alt text placeholders for images\n- Readable text contrast\n\nTo use this implementation:\n\n1. Make sure all the fonts are properly loaded (they're being loaded from Google Fonts)\n2. Replace the image placeholders with actual images\n3. Add proper links to the social media navigation\n4. Add any additional interactive features or animations as needed\n\nThe implementation maintains the visual hierarchy and styling of the original Figma design while \nensuring it's responsive and accessible. You can further customize the colors, spacing, \nand other properties by modifying the CSS variables if needed.\n\nWould you like me to add any specific interactions or make any adjustments to the implementation?\n"})}),"\n",(0,s.jsx)(n.h3,{id:"result",children:"Result"}),"\n",(0,s.jsx)(n.p,{children:"The Figma design has been successfully converted into a webpage with the styling, content and components from the original design."}),"\n",(0,s.jsx)(n.p,{children:(0,s.jsx)(n.img,{alt:"Figma MCP Output",src:i(5478).A+"",width:"3600",height:"2068"})}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsx)(n.p,{children:"In cases where you need to make additional changes, or the final output is not as expected - you can continue to interact with Goose to make adjustments."})})]})}function p(e={}){const{wrapper:n}={...(0,o.R)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(g,{...e})}):g(e)}},5887:(e,n,i)=>{i.d(n,{A:()=>o});i(6540);var t=i(2362),s=i(4848);const o=e=>{let{videoUrl:n}=e;return(0,s.jsxs)("div",{children:[(0,s.jsx)(t.A,{type:"info",icon:"\ud83c\udfa5",title:"Plug & Play",className:"alert--video",children:(0,s.jsxs)("details",{children:[(0,s.jsx)("summary",{children:"Watch the demo"}),(0,s.jsx)("div",{style:{textAlign:"center",margin:"20px 0"},children:(0,s.jsx)("iframe",{width:"100%",height:"540",src:n,title:"YouTube Short",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})]})}),(0,s.jsx)("hr",{})]})}},403:(e,n,i)=>{i.d(n,{A:()=>t});const t=i.p+"assets/images/figma-mcp-design-dac4a1618225011a385bb592858e3b3b.png"},5478:(e,n,i)=>{i.d(n,{A:()=>t});const t=i.p+"assets/images/figma-mcp-output-3fa01318977684f5673d44f5b9ad7a19.png"}}]);