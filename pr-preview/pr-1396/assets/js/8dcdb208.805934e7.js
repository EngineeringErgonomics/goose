"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[2606],{5801:(e,n,s)=>{s.r(n),s.d(n,{assets:()=>d,contentTitle:()=>a,default:()=>u,frontMatter:()=>c,metadata:()=>i,toc:()=>h});const i=JSON.parse('{"id":"tutorials/puppeteer-mcp","title":"Puppeteer Extension","description":"Add Puppeteer MCP Server as a Goose Extension","source":"@site/docs/tutorials/puppeteer-mcp.md","sourceDirName":"tutorials","slug":"/tutorials/puppeteer-mcp","permalink":"/goose/pr-preview/pr-1396/docs/tutorials/puppeteer-mcp","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Puppeteer Extension","description":"Add Puppeteer MCP Server as a Goose Extension"},"sidebar":"tutorialSidebar","previous":{"title":"Memory Extension","permalink":"/goose/pr-preview/pr-1396/docs/tutorials/memory-mcp"},"next":{"title":"Tavily Web Search Extension","permalink":"/goose/pr-preview/pr-1396/docs/tutorials/tavily-mcp"}}');var t=s(4848),r=s(8453),o=s(5537),l=s(9329);s(5887);const c={title:"Puppeteer Extension",description:"Add Puppeteer MCP Server as a Goose Extension"},a=void 0,d={},h=[{value:"Configuration",id:"configuration",level:2},{value:"Example Usage",id:"example-usage",level:2},{value:"Goose Prompt",id:"goose-prompt",level:3},{value:"Goose Output",id:"goose-output",level:3},{value:"Further Automation with GitHub Extension",id:"further-automation-with-github-extension",level:3},{value:"Goose prompt:",id:"goose-prompt-1",level:3}];function p(e){const n={a:"a",admonition:"admonition",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",mdxAdmonitionTitle:"mdxAdmonitionTitle",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)(n.p,{children:["This tutorial covers how to add the ",(0,t.jsx)(n.a,{href:"https://github.com/modelcontextprotocol/servers/tree/HEAD/src/puppeteer",children:"Puppeteer MCP Server"})," as a Goose extension, enabling Goose to interact with websites - navigating pages, filling forms, clicking buttons, taking screenshots, and executing JavaScript in a real browser environment."]}),"\n",(0,t.jsxs)(n.admonition,{title:"TLDR",type:"tip",children:[(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Command"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"npx -y @modelcontextprotocol/server-puppeteer\n"})})]}),"\n",(0,t.jsx)(n.h2,{id:"configuration",children:"Configuration"}),"\n",(0,t.jsx)(n.admonition,{type:"info",children:(0,t.jsxs)(n.p,{children:["Note that you'll need ",(0,t.jsx)(n.a,{href:"https://nodejs.org/",children:"Node.js"})," installed on your system to run this command, as it uses ",(0,t.jsx)(n.code,{children:"npx"}),"."]})}),"\n",(0,t.jsxs)(o.A,{groupId:"interface",children:[(0,t.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Run the ",(0,t.jsx)(n.code,{children:"configure"})," command:"]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose configure\n"})}),(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["Choose to add a ",(0,t.jsx)(n.code,{children:"Command-line Extension"})]}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c6  What type of extension would you like to add?\n  \u2502  \u25cb Built-in Extension \n  // highlight-start    \n  \u2502  \u25cf Command-line Extension (Run a local command or script)\n  // highlight-end    \n  \u2502  \u25cb Remote Extension \n  \u2514 \n"})}),(0,t.jsxs)(n.ol,{start:"3",children:["\n",(0,t.jsx)(n.li,{children:"Give your extension a name"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  // highlight-start\n  \u25c6  What would you like to call this extension?\n  \u2502  Puppeteer\n  // highlight-end\n  \u2514 \n"})}),(0,t.jsxs)(n.ol,{start:"4",children:["\n",(0,t.jsx)(n.li,{children:"Enter the command"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"  \u250c   goose-configure \n  \u2502\n  \u25c7  What would you like to configure?\n  \u2502  Add Extension \n  \u2502\n  \u25c7  What type of extension would you like to add?\n  \u2502  Command-line Extension \n  \u2502\n  \u25c7  What would you like to call this extension?\n  \u2502  Puppeteer\n  \u2502\n  // highlight-start\n  \u25c6  What command should be run?\n  \u2502  npx -y @modelcontextprotocol/server-puppeteer\n  // highlight-end\n  \u2514 \n"})}),(0,t.jsxs)(n.ol,{start:"5",children:["\n",(0,t.jsx)(n.li,{children:"Add Extension"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:" \u250c   goose-configure \n \u2502\n \u25c7  What would you like to configure?\n \u2502  Add Extension \n \u2502\n \u25c7  What type of extension would you like to add?\n \u2502  Command-line Extension \n \u2502\n \u25c7  What would you like to call this extension?\n \u2502  Puppeteer\n \u2502\n \u25c7  What command should be run?\n \u2502  npx -y @modelcontextprotocol/server-puppeteer\n // highlight-start\n \u25c6  Would you like to add environment variables?\n \u2502  No \n \u2502\n // highlight-end\n \u2514  Added puppeteer extension\n"})})]}),(0,t.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Click ",(0,t.jsx)(n.code,{children:"..."})," in the upper right corner"]}),"\n",(0,t.jsxs)(n.li,{children:["Click ",(0,t.jsx)(n.code,{children:"Settings"})]}),"\n",(0,t.jsxs)(n.li,{children:["Under ",(0,t.jsx)(n.code,{children:"Extensions"}),", click the ",(0,t.jsx)(n.code,{children:"Add"})," link"]}),"\n",(0,t.jsxs)(n.li,{children:["On the ",(0,t.jsx)(n.code,{children:"Add Extension Manually"})," modal, enter the following:","\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Type"}),": ",(0,t.jsx)(n.code,{children:"Standard IO"})]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"ID"}),": ",(0,t.jsx)(n.code,{children:"puppeteer-mcp"})," (",(0,t.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Name"}),": ",(0,t.jsx)(n.code,{children:"Puppeteer"})," (",(0,t.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Description"}),": ",(0,t.jsx)(n.code,{children:"Puppeteer MCP Server"})," (",(0,t.jsx)(n.em,{children:"set this to whatever you want"}),")"]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Command"}),": ",(0,t.jsx)(n.code,{children:"npx -y @modelcontextprotocol/server-puppeteer"})]}),"\n"]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["Click ",(0,t.jsx)(n.code,{children:"Add Extension"})," button"]}),"\n"]})})]}),"\n",(0,t.jsx)(n.h2,{id:"example-usage",children:"Example Usage"}),"\n",(0,t.jsx)(n.p,{children:"In this example, I\u2019ll show you how to use Goose with the Puppeteer Extension to perform an accessibility audit on a website. By automating browser interactions, Goose can navigate the site, test keyboard navigation, check color contrast, analyze ARIA attributes, and generate a detailed report with recommendations and code fixes."}),"\n",(0,t.jsx)(n.p,{children:"This allows you to quickly identify and resolve accessibility issues without manually inspecting each page."}),"\n",(0,t.jsxs)(o.A,{groupId:"interface",children:[(0,t.jsxs)(l.A,{value:"cli",label:"Goose CLI",default:!0,children:[(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Open a terminal and start a new Goose session:"}),"\n"]}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-sh",children:"goose session\n"})})]}),(0,t.jsx)(l.A,{value:"ui",label:"Goose Desktop",children:(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"Open a new session in Goose Desktop"}),"\n"]})})]}),"\n",(0,t.jsx)(n.h3,{id:"goose-prompt",children:"Goose Prompt"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"Can you check if our website, primarily the docs section, is accessible? Please conduct a full accessibility audit, focusing on the following:\n\n1. Keyboard Navigation: Test all interactive elements (links, buttons, dropdowns, etc.) to ensure they are accessible without a mouse.\n\n2. Color Contrast Analysis: Verify that all text and UI elements meet WCAG contrast ratio standards.\n\n3. Screen Reader Compatibility: Test whether a screen reader can properly interpret content and navigation.\n\n4. ARIA & Semantic HTML Validation: Identify any missing or misused ARIA attributes and ensure proper semantic structure.\n\n5. Error Identification & Screenshots: Provide a comprehensive report with identified issues, along with screenshots if necessary.\n\n6. Fixes & Code Recommendations: Suggest HTML, CSS, or JavaScript changes to resolve any compliance issues.\n\nHere is the website to test: https://broken-workshop.dequelabs.com/. I want to ensure that our documentation meets accessibility standards. Please generate a detailed report with steps to fix any issues found.\n\n"})}),"\n",(0,t.jsx)(n.h3,{id:"goose-output",children:"Goose Output"}),"\n",(0,t.jsxs)(n.admonition,{title:"Desktop",type:"note",children:[(0,t.jsx)(n.p,{children:"I'll help you conduct a comprehensive accessibility audit of the website. I'll use a combination of automated and manual testing approaches to evaluate different accessibility aspects."}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"First, let's navigate to the site"}),"\n",(0,t.jsx)(n.li,{children:"Take screenshots and perform keyboard navigation testing"}),"\n",(0,t.jsx)(n.li,{children:"Run accessibility audits including color contrast analysis"}),"\n",(0,t.jsx)(n.li,{children:"Document findings"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Let's begin:"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.code,{children:"\u2500\u2500\u2500 Puppeteer Navigate \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Let's take a screenshot of the homepage for reference:"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.code,{children:"\u2500\u2500\u2500 Puppeteer Screenshot \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"})}),(0,t.jsx)(n.p,{children:"I'll conduct a detailed accessibility audit of the recipe dashboard website. From the initial view, I can already identify several accessibility concerns. Let me break down the findings by category:"}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"1. Keyboard Navigation Issues:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:'The "COOK" buttons for recipes appear to lack proper focus indicators'}),"\n",(0,t.jsx)(n.li,{children:"The edit icons (pencil icons) on recipe cards don't seem to have keyboard focus states"}),"\n",(0,t.jsx)(n.li,{children:"The navigation sequence may not be logical as there's no clear tab order"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"2. Color Contrast Issues:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:'The light gray text used for "Difficulty" labels may not meet WCAG contrast requirements'}),"\n",(0,t.jsx)(n.li,{children:'The green "Beginner" text appears to have insufficient contrast against the white background'}),"\n",(0,t.jsx)(n.li,{children:'The bar chart in the "YUMMINESS" section uses colors that may be difficult to distinguish'}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"3. Semantic Structure Problems:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:'The heading hierarchy is unclear - "RECIPE DASHBOARD" appears to be an h1, but recipe titles don\'t seem to use proper heading tags'}),"\n",(0,t.jsx)(n.li,{children:"The statistics (9 EGGS USED, 10 RECIPES MADE, etc.) don't appear to use semantic list structures"}),"\n",(0,t.jsx)(n.li,{children:"The chart lacks proper ARIA labels for screen readers"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"4. ARIA and Screen Reader Issues:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"The recipe cards don't appear to have proper aria-labelledby associations"}),"\n",(0,t.jsx)(n.li,{children:"The edit buttons lack aria-labels describing their purpose"}),"\n",(0,t.jsx)(n.li,{children:"The chart data isn't properly accessible to screen readers"}),"\n",(0,t.jsx)(n.li,{children:"No ARIA landmarks are visible for main navigation areas"}),"\n"]}),(0,t.jsx)(n.p,{children:"Let me run some additional tests to verify these issues:"}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.code,{children:"\u2500\u2500\u2500 Puppeteer Evaluate \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"})}),(0,t.jsx)(n.p,{children:"Let me capture some specific sections for detailed analysis:"}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.code,{children:"\u2500\u2500\u2500 Puppeteer Screenshot \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"})}),(0,t.jsx)(n.p,{children:"Based on my analysis, here's a detailed report with recommendations:"}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"ACCESSIBILITY AUDIT REPORT"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"1. Structural Issues:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'\x3c!-- Current --\x3e\n<div class="recipe-title">Chocolate Cake</div>\n\n\x3c!-- Recommended --\x3e\n<h2 class="recipe-title">Chocolate Cake</h2>\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"2. Interactive Elements:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'\x3c!-- Current --\x3e\n<button class="cook-button">COOK CHOCOLATE CAKE</button>\n\n\x3c!-- Recommended --\x3e\n<button \n  class="cook-button"\n  aria-label="Cook Chocolate Cake recipe"\n  tabindex="0">\n  COOK CHOCOLATE CAKE\n</button>\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"3. Image Accessibility:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'\x3c!-- Current --\x3e\n<img src="chocolate-cake.jpg">\n\n\x3c!-- Recommended --\x3e\n<img \n  src="chocolate-cake.jpg" \n  alt="Chocolate cake with dark chocolate frosting"\n  role="img">\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"4. Chart Accessibility:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'\x3c!-- Recommended addition --\x3e\n<div class="yumminess-chart" \n     role="region" \n     aria-label="Monthly yumminess ratings">\n  <table class="sr-only">\n    \x3c!-- Add hidden table for screen readers --\x3e\n  </table>\n</div>\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"5. Color Contrast Fixes:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-css",children:"/* Current */\n.difficulty-beginner {\n  color: #90EE90;\n}\n\n/* Recommended */\n.difficulty-beginner {\n  color: #2E7D32; /* Darker green for better contrast */\n}\n"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"6. Focus Management:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-css",children:"/* Add to CSS */\n:focus {\n  outline: 3px solid #2196F3;\n  outline-offset: 2px;\n}\n\n.cook-button:focus {\n  box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.4);\n}\n"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Priority Fixes Needed:"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"1. High Priority:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Add proper heading structure using h1-h6 tags"}),"\n",(0,t.jsx)(n.li,{children:"Implement keyboard focus indicators"}),"\n",(0,t.jsx)(n.li,{children:"Add alt text to all recipe images"}),"\n",(0,t.jsx)(n.li,{children:"Fix color contrast for difficulty labels"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"2. Medium Priority:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Add ARIA labels to interactive elements"}),"\n",(0,t.jsx)(n.li,{children:"Implement proper list structure for statistics"}),"\n",(0,t.jsx)(n.li,{children:"Add skip links for keyboard navigation"}),"\n",(0,t.jsx)(n.li,{children:"Make chart data accessible via screen reader"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"3. Low Priority:"})}),(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Enhance focus order"}),"\n",(0,t.jsx)(n.li,{children:"Add ARIA landmarks"}),"\n",(0,t.jsx)(n.li,{children:"Implement announcement of dynamic content changes"}),"\n"]}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Additional Recommendations:"})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"1. Add a skip link:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'<a href="#main-content" class="skip-link">\n  Skip to main content\n</a>\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"2. Implement proper form labels:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'<label for="prep-time">Preparation Time</label>\n<input id="prep-time" type="text" value="20 min">\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"3. Add ARIA live regions for dynamic updates:"})}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-html",children:'<div aria-live="polite" role="status">\n  \x3c!-- Dynamic content updates --\x3e\n</div>\n'})}),(0,t.jsx)(n.p,{children:(0,t.jsx)(n.strong,{children:"Testing Tools Recommended:"})}),(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsx)(n.li,{children:"WAVE Web Accessibility Evaluation Tool"}),"\n",(0,t.jsx)(n.li,{children:"axe DevTools"}),"\n",(0,t.jsx)(n.li,{children:"Keyboard-only navigation testing"}),"\n",(0,t.jsx)(n.li,{children:"Screen reader testing with VoiceOver (Mac)"}),"\n"]}),(0,t.jsx)(n.p,{children:"This audit reveals several critical accessibility issues that should be addressed to ensure WCAG 2.1 compliance. The most urgent concerns are the lack of proper semantic structure, insufficient color contrast, and missing ARIA attributes. Would you like me to provide more specific details about implementing any of these fixes?"})]}),"\n",(0,t.jsx)(n.h3,{id:"further-automation-with-github-extension",children:"Further Automation with GitHub Extension"}),"\n",(0,t.jsxs)(n.p,{children:["You can take this accessibility audit a step further by combining the ",(0,t.jsx)(n.a,{href:"/docs/tutorials/github-mcp",children:"GitHub Extension"})," with the Puppeteer Extension. With this setup, Goose doesn\u2019t just find issues and apply fixes, it can also handle the entire Git workflow for you. The GitHub Extension allows Goose to commit changes, create a pull request, and even generate a PR description, so all you have to do is review and merge."]}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["Enable the GitHub extension by following the steps in the ",(0,t.jsx)(n.strong,{children:(0,t.jsx)(n.a,{href:"/docs/tutorials/github-mcp#configuration",children:"GitHub Extension Tutorial"})}),"."]}),"\n"]}),"\n",(0,t.jsxs)(n.admonition,{type:"tip",children:[(0,t.jsx)(n.mdxAdmonitionTitle,{}),(0,t.jsx)(n.p,{children:"Ensure your GitHub Personal Access Token has the necessary permissions for repository access and pull request creation when using this combined approach."})]}),"\n",(0,t.jsxs)(n.ol,{start:"2",children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsx)(n.p,{children:"Ask Goose to:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Create a new branch"}),"\n",(0,t.jsx)(n.li,{children:"Commit the accessibility improvements"}),"\n",(0,t.jsx)(n.li,{children:"Open a pull request"}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.h3,{id:"goose-prompt-1",children:"Goose prompt:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{children:"Can you create a new branch called 'accessibility-improvements', apply the accessibility fixes you suggested, and open a pull request with these changes?\n"})}),"\n",(0,t.jsx)(n.p,{children:"Goose will then:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["\u2705 Create a branch: ",(0,t.jsx)(n.code,{children:"accessibility-improvements"})]}),"\n",(0,t.jsx)(n.li,{children:"\u2705 Apply the recommended accessibility fixes"}),"\n",(0,t.jsx)(n.li,{children:"\u2705 Commit the changes with a descriptive message"}),"\n",(0,t.jsx)(n.li,{children:"\u2705 Open a pull request with a summary of improvements"}),"\n"]})]})}function u(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(p,{...e})}):p(e)}},5887:(e,n,s)=>{s.d(n,{A:()=>r});s(6540);var i=s(2362),t=s(4848);const r=e=>{let{videoUrl:n}=e;return(0,t.jsxs)("div",{children:[(0,t.jsx)(i.A,{type:"info",icon:"\ud83c\udfa5",title:"Plug & Play",className:"alert--video",children:(0,t.jsxs)("details",{children:[(0,t.jsx)("summary",{children:"Watch the demo"}),(0,t.jsx)("div",{style:{textAlign:"center",margin:"20px 0"},children:(0,t.jsx)("iframe",{width:"100%",height:"540",src:n,title:"YouTube Short",frameBorder:"0",allow:"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",allowFullScreen:!0})})]})}),(0,t.jsx)("hr",{})]})}}}]);