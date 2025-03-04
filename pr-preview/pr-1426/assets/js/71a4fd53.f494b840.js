"use strict";(self.webpackChunkgoose=self.webpackChunkgoose||[]).push([[5342],{2165:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>l,contentTitle:()=>a,default:()=>d,frontMatter:()=>i,metadata:()=>s,toc:()=>c});const s=JSON.parse('{"id":"tutorials/running-goose-cicd","title":"Running Goose in CI/CD Environments","description":"Learn how to set up Goose in your CI/CD pipeline. Automate Goose interactions for tasks like code review, documentation checks, and other automated workflows.","source":"@site/docs/tutorials/running-goose-cicd.md","sourceDirName":"tutorials","slug":"/tutorials/running-goose-cicd","permalink":"/goose/pr-preview/pr-1426/docs/tutorials/running-goose-cicd","draft":false,"unlisted":false,"tags":[],"version":"current","frontMatter":{"title":"Running Goose in CI/CD Environments","description":"Learn how to set up Goose in your CI/CD pipeline. Automate Goose interactions for tasks like code review, documentation checks, and other automated workflows."},"sidebar":"tutorialSidebar","previous":{"title":"Puppeteer Extension","permalink":"/goose/pr-preview/pr-1426/docs/tutorials/puppeteer-mcp"},"next":{"title":"Tavily Web Search Extension","permalink":"/goose/pr-preview/pr-1426/docs/tutorials/tavily-mcp"}}');var t=o(4848),r=o(8453);o(5537),o(9329);const i={title:"Running Goose in CI/CD Environments",description:"Learn how to set up Goose in your CI/CD pipeline. Automate Goose interactions for tasks like code review, documentation checks, and other automated workflows."},a=void 0,l={},c=[{value:"Common Use Cases",id:"common-use-cases",level:2},{value:"Using Goose with GitHub Actions",id:"using-goose-with-github-actions",level:2},{value:"Create the Workflow File",id:"create-the-workflow-file",level:4},{value:"Configure Basic Workflow Structure",id:"configure-basic-workflow-structure",level:4},{value:"Install and Configure Goose",id:"install-and-configure-goose",level:4},{value:"Prepare Instructions for Goose",id:"prepare-instructions-for-goose",level:4},{value:"Run Goose and Filter Output",id:"run-goose-and-filter-output",level:4},{value:"Post Comment to PR",id:"post-comment-to-pr",level:4},{value:"Using CI specific MCP servers as Goose extensions",id:"using-ci-specific-mcp-servers-as-goose-extensions",level:2},{value:"Security Considerations",id:"security-considerations",level:2}];function u(e){const n={a:"a",admonition:"admonition",code:"code",h2:"h2",h4:"h4",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.R)(),...e.components},{Details:o}=n;return o||function(e,n){throw new Error("Expected "+(n?"component":"object")+" `"+e+"` to be defined: you likely forgot to import, pass, or provide it.")}("Details",!0),(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.p,{children:"With the same way we use Goose to resolve issues on our local machine, we can also use Goose in CI/CD environments to automate tasks like code review, documentation checks, and other automated workflows. This tutorial will guide you through setting up Goose in your CI/CD pipeline."}),"\n",(0,t.jsx)(n.h2,{id:"common-use-cases",children:"Common Use Cases"}),"\n",(0,t.jsx)(n.p,{children:"Here are some common ways to use Goose in your CI/CD pipeline:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Automating Build and Deployment Tasks"}),"\n",(0,t.jsx)(n.li,{children:"Infrastructure and Environment Management"}),"\n",(0,t.jsx)(n.li,{children:"Automating Rollbacks and Recovery"}),"\n",(0,t.jsx)(n.li,{children:"Intelligent Test Execution"}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"using-goose-with-github-actions",children:"Using Goose with GitHub Actions"}),"\n",(0,t.jsx)(n.p,{children:"You can also use Goose directly in your GitHub Actions workflow, follow these steps:"}),"\n",(0,t.jsx)(n.admonition,{title:"TLDR",type:"info",children:(0,t.jsxs)(o,{children:[(0,t.jsx)("summary",{children:"Copy the GitHub Workflow"}),(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:'\nname: Goose\n\non:\n   pull_request:\n      types: [opened, synchronize, reopened, labeled]\n\npermissions:\n   contents: write\n   pull-requests: write\n   issues: write\n\nenv:\n   PROVIDER_API_KEY: ${{ secrets.REPLACE_WITH_PROVIDER_API_KEY }}\n   PR_NUMBER: ${{ github.event.pull_request.number }}\n\njobs:\n   goose-comment:\n      runs-on: ubuntu-latest\n\n      steps:\n            - name: Check out repository\n            uses: actions/checkout@v4\n            with:\n                  fetch-depth: 0\n\n            - name: Gather PR information\n            run: |\n                  {\n                  echo "# Files Changed"\n                  gh pr view $PR_NUMBER --json files \\\n                     -q \'.files[] | "* " + .path + " (" + (.additions|tostring) + " additions, " + (.deletions|tostring) + " deletions)"\'\n                  echo ""\n                  echo "# Changes Summary"\n                  gh pr diff $PR_NUMBER\n                  } > changes.txt\n\n            - name: Install Goose CLI\n            run: |\n                  mkdir -p /home/runner/.local/bin\n                  curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh \\\n                  | CONFIGURE=false INSTALL_PATH=/home/runner/.local/bin bash\n                  echo "/home/runner/.local/bin" >> $GITHUB_PATH\n\n            - name: Configure Goose\n            run: |\n                  mkdir -p ~/.config/goose\n                  cat <<EOF > ~/.config/goose/config.yaml\n                  GOOSE_PROVIDER: REPLACE_WITH_PROVIDER\n                  GOOSE_MODEL: REPLACE_WITH_MODEL\n                  keyring: false\n                  EOF\n\n            - name: Create instructions for Goose\n            run: |\n                  cat <<EOF > instructions.txt\n                  Create a summary of the changes provided. Don\'t provide any session or logging details.\n                  The summary for each file should be brief and structured as:\n                  <filename/path (wrapped in backticks)>\n                     - dot points of changes\n                  You don\'t need any extensions, don\'t mention extensions at all.\n                  The changes to summarise are:\n                  $(cat changes.txt)\n                  EOF\n\n            - name: Test\n            run: cat instructions.txt\n\n            - name: Run Goose and filter output\n            run: |\n                  goose run --instructions instructions.txt | \\\n                  # Remove ANSI color codes\n                  sed -E \'s/\\x1B\\[[0-9;]*[mK]//g\' | \\\n                  # Remove session/logging lines\n                  grep -v "logging to /home/runner/.config/goose/sessions/" | \\\n                  grep -v "^starting session" | \\\n                  grep -v "^Closing session" | \\\n                  # Trim trailing whitespace\n                  sed \'s/[[:space:]]*$//\' \\\n                  > pr_comment.txt\n\n            - name: Post comment to PR\n            run: |\n                  cat -A pr_comment.txt\n                  gh pr comment $PR_NUMBER --body-file pr_comment.txt\n'})})]})}),"\n",(0,t.jsx)(n.h4,{id:"create-the-workflow-file",children:"Create the Workflow File"}),"\n",(0,t.jsxs)(n.p,{children:["Create a new file in your repository at ",(0,t.jsx)(n.code,{children:".github/workflows/goose.yml"}),". This will contain your GitHub Actions workflow configuration."]}),"\n",(0,t.jsx)(n.h4,{id:"configure-basic-workflow-structure",children:"Configure Basic Workflow Structure"}),"\n",(0,t.jsx)(n.p,{children:"Here's a basic workflow structure that triggers Goose on pull requests:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:"name: Goose\n\non:\n    pull_request:\n        types: [opened, synchronize, reopened, labeled]\n\npermissions:\n    contents: write\n    pull-requests: write\n    issues: write\n\nenv:\n   PROVIDER_API_KEY: ${{ secrets.REPLACE_WITH_PROVIDER_API_KEY }}\n   PR_NUMBER: ${{ github.event.pull_request.number }}\n"})}),"\n",(0,t.jsx)(n.p,{children:"This configuration:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsx)(n.li,{children:"Triggers the workflow on pull request events"}),"\n",(0,t.jsx)(n.li,{children:"Sets necessary permissions for GitHub Actions"}),"\n",(0,t.jsx)(n.li,{children:"Configures environment variables for your chosen Goose provider"}),"\n"]}),"\n",(0,t.jsx)(n.h4,{id:"install-and-configure-goose",children:"Install and Configure Goose"}),"\n",(0,t.jsx)(n.p,{children:"The workflow needs to install and configure Goose in the CI environment. Here's how to do it:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:'steps:\n    - name: Install Goose CLI\n      run: |\n          mkdir -p /home/runner/.local/bin\n          curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh \\\n            | CONFIGURE=false INSTALL_PATH=/home/runner/.local/bin bash\n          echo "/home/runner/.local/bin" >> $GITHUB_PATH\n\n    - name: Configure Goose\n      run: |\n          mkdir -p ~/.config/goose\n          cat <<EOF > ~/.config/goose/config.yaml\n          GOOSE_PROVIDER: REPLACE_WITH_PROVIDER\n          GOOSE_MODEL: REPLACE_WITH_MODEL\n          keyring: false\n          EOF\n'})}),"\n",(0,t.jsxs)(n.p,{children:["Replace ",(0,t.jsx)(n.code,{children:"REPLACE_WITH_PROVIDER"})," and ",(0,t.jsx)(n.code,{children:"REPLACE_WITH_MODEL"})," with your Goose provider and model names and add any other necessary configuration required."]}),"\n",(0,t.jsx)(n.h4,{id:"prepare-instructions-for-goose",children:"Prepare Instructions for Goose"}),"\n",(0,t.jsx)(n.p,{children:"Create instructions for Goose to follow based on the PR changes:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:"    - name: Create instructions for Goose\n      run: |\n          cat <<EOF > instructions.txt\n          Create a summary of the changes provided. Don't provide any session or logging details.\n          The summary for each file should be brief and structured as:\n            <filename/path (wrapped in backticks)>\n              - dot points of changes\n          You don't need any extensions, don't mention extensions at all.\n          The changes to summarise are:\n          $(cat changes.txt)\n          EOF\n"})}),"\n",(0,t.jsx)(n.h4,{id:"run-goose-and-filter-output",children:"Run Goose and Filter Output"}),"\n",(0,t.jsx)(n.p,{children:"Run Goose with the prepared instructions and filter the output for clean results:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:'    - name: Run Goose and filter output\n      run: |\n          goose run --instructions instructions.txt | \\\n            # Remove ANSI color codes\n            sed -E \'s/\\x1B\\[[0-9;]*[mK]//g\' | \\\n            # Remove session/logging lines\n            grep -v "logging to /home/runner/.config/goose/sessions/" | \\\n            grep -v "^starting session" | \\\n            grep -v "^Closing session" | \\\n            # Trim trailing whitespace\n            sed \'s/[[:space:]]*$//\' \\\n            > pr_comment.txt\n'})}),"\n",(0,t.jsx)(n.h4,{id:"post-comment-to-pr",children:"Post Comment to PR"}),"\n",(0,t.jsx)(n.p,{children:"Finally, post the Goose output as a comment on the pull request:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:"    - name: Post comment to PR\n      run: |\n          cat -A pr_comment.txt\n          gh pr comment $PR_NUMBER --body-file pr_comment.txt\n"})}),"\n",(0,t.jsx)(n.p,{children:"With this workflow, Goose will run on pull requests, analyze the changes, and post a summary as a comment on the PR."}),"\n",(0,t.jsx)(n.h2,{id:"using-ci-specific-mcp-servers-as-goose-extensions",children:"Using CI specific MCP servers as Goose extensions"}),"\n",(0,t.jsx)(n.p,{children:"There might also be cases where you want to use Goose with other environments, custom setups etc. In such cases, you can use Goose extensions to interact with these environments."}),"\n",(0,t.jsxs)(n.p,{children:["You can find related extensions as MCP Servers on ",(0,t.jsx)(n.a,{href:"https://www.pulsemcp.com/servers",children:"PulseMCP"})," and interact with them using Goose."]}),"\n",(0,t.jsx)(n.p,{children:"Process Goose's output to ensure it's clean and useful:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-yaml",children:'    - name: Run Goose and filter output\n      run: |\n          goose run --instructions instructions.txt | \\\n            # Remove ANSI color codes\n            sed -E \'s/\\x1B\\[[0-9;]*[mK]//g\' | \\\n            # Remove session/logging lines\n            grep -v "logging to /home/runner/.config/goose/sessions/" | \\\n            grep -v "^starting session" | \\\n            grep -v "^Closing session" | \\\n            # Trim trailing whitespace\n            sed \'s/[[:space:]]*$//\' \\\n            > pr_comment.txt\n'})}),"\n",(0,t.jsx)(n.h2,{id:"security-considerations",children:"Security Considerations"}),"\n",(0,t.jsx)(n.p,{children:"When running Goose in CI/CD, keep these security practices in mind:"}),"\n",(0,t.jsxs)(n.ol,{children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Secret Management"}),": Store your sensitive credentials (like API tokens) as 'Secrets' that you can pass to GOose as environment variables. Never expose these credentials in logs or PR comments"]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Permissions"}),": When using a script or workflow, ensure you follow the principle of least privilege. Only grant necessary permissions in the workflow and regularly audit workflow permissions."]}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:[(0,t.jsx)(n.strong,{children:"Input Validation"}),": Validate and sanitize inputs before passing to Goose. Consider using action inputs with specific types and implement appropriate error handling."]}),"\n"]}),"\n"]})]})}function d(e={}){const{wrapper:n}={...(0,r.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(u,{...e})}):u(e)}},9329:(e,n,o)=>{o.d(n,{A:()=>i});o(6540);var s=o(4164);const t={tabItem:"tabItem_Ymn6"};var r=o(4848);function i(e){let{children:n,hidden:o,className:i}=e;return(0,r.jsx)("div",{role:"tabpanel",className:(0,s.A)(t.tabItem,i),hidden:o,children:n})}},5537:(e,n,o)=>{o.d(n,{A:()=>j});var s=o(6540),t=o(4164),r=o(5627),i=o(6347),a=o(372),l=o(604),c=o(1861),u=o(8749);function d(e){return s.Children.toArray(e).filter((e=>"\n"!==e)).map((e=>{if(!e||(0,s.isValidElement)(e)&&function(e){const{props:n}=e;return!!n&&"object"==typeof n&&"value"in n}(e))return e;throw new Error(`Docusaurus error: Bad <Tabs> child <${"string"==typeof e.type?e.type:e.type.name}>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.`)}))?.filter(Boolean)??[]}function h(e){const{values:n,children:o}=e;return(0,s.useMemo)((()=>{const e=n??function(e){return d(e).map((e=>{let{props:{value:n,label:o,attributes:s,default:t}}=e;return{value:n,label:o,attributes:s,default:t}}))}(o);return function(e){const n=(0,c.XI)(e,((e,n)=>e.value===n.value));if(n.length>0)throw new Error(`Docusaurus error: Duplicate values "${n.map((e=>e.value)).join(", ")}" found in <Tabs>. Every value needs to be unique.`)}(e),e}),[n,o])}function p(e){let{value:n,tabValues:o}=e;return o.some((e=>e.value===n))}function m(e){let{queryString:n=!1,groupId:o}=e;const t=(0,i.W6)(),r=function(e){let{queryString:n=!1,groupId:o}=e;if("string"==typeof n)return n;if(!1===n)return null;if(!0===n&&!o)throw new Error('Docusaurus error: The <Tabs> component groupId prop is required if queryString=true, because this value is used as the search param name. You can also provide an explicit value such as queryString="my-search-param".');return o??null}({queryString:n,groupId:o});return[(0,l.aZ)(r),(0,s.useCallback)((e=>{if(!r)return;const n=new URLSearchParams(t.location.search);n.set(r,e),t.replace({...t.location,search:n.toString()})}),[r,t])]}function g(e){const{defaultValue:n,queryString:o=!1,groupId:t}=e,r=h(e),[i,l]=(0,s.useState)((()=>function(e){let{defaultValue:n,tabValues:o}=e;if(0===o.length)throw new Error("Docusaurus error: the <Tabs> component requires at least one <TabItem> children component");if(n){if(!p({value:n,tabValues:o}))throw new Error(`Docusaurus error: The <Tabs> has a defaultValue "${n}" but none of its children has the corresponding value. Available values are: ${o.map((e=>e.value)).join(", ")}. If you intend to show no default tab, use defaultValue={null} instead.`);return n}const s=o.find((e=>e.default))??o[0];if(!s)throw new Error("Unexpected error: 0 tabValues");return s.value}({defaultValue:n,tabValues:r}))),[c,d]=m({queryString:o,groupId:t}),[g,f]=function(e){let{groupId:n}=e;const o=function(e){return e?`docusaurus.tab.${e}`:null}(n),[t,r]=(0,u.Dv)(o);return[t,(0,s.useCallback)((e=>{o&&r.set(e)}),[o,r])]}({groupId:t}),v=(()=>{const e=c??g;return p({value:e,tabValues:r})?e:null})();(0,a.A)((()=>{v&&l(v)}),[v]);return{selectedValue:i,selectValue:(0,s.useCallback)((e=>{if(!p({value:e,tabValues:r}))throw new Error(`Can't select invalid tab value=${e}`);l(e),d(e),f(e)}),[d,f,r]),tabValues:r}}var f=o(9136);const v={tabList:"tabList__CuJ",tabItem:"tabItem_LNqP"};var x=o(4848);function w(e){let{className:n,block:o,selectedValue:s,selectValue:i,tabValues:a}=e;const l=[],{blockElementScrollPositionUntilNextRender:c}=(0,r.a_)(),u=e=>{const n=e.currentTarget,o=l.indexOf(n),t=a[o].value;t!==s&&(c(n),i(t))},d=e=>{let n=null;switch(e.key){case"Enter":u(e);break;case"ArrowRight":{const o=l.indexOf(e.currentTarget)+1;n=l[o]??l[0];break}case"ArrowLeft":{const o=l.indexOf(e.currentTarget)-1;n=l[o]??l[l.length-1];break}}n?.focus()};return(0,x.jsx)("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,t.A)("tabs",{"tabs--block":o},n),children:a.map((e=>{let{value:n,label:o,attributes:r}=e;return(0,x.jsx)("li",{role:"tab",tabIndex:s===n?0:-1,"aria-selected":s===n,ref:e=>{l.push(e)},onKeyDown:d,onClick:u,...r,className:(0,t.A)("tabs__item",v.tabItem,r?.className,{"tabs__item--active":s===n}),children:o??n},n)}))})}function b(e){let{lazy:n,children:o,selectedValue:r}=e;const i=(Array.isArray(o)?o:[o]).filter(Boolean);if(n){const e=i.find((e=>e.props.value===r));return e?(0,s.cloneElement)(e,{className:(0,t.A)("margin-top--md",e.props.className)}):null}return(0,x.jsx)("div",{className:"margin-top--md",children:i.map(((e,n)=>(0,s.cloneElement)(e,{key:n,hidden:e.props.value!==r})))})}function y(e){const n=g(e);return(0,x.jsxs)("div",{className:(0,t.A)("tabs-container",v.tabList),children:[(0,x.jsx)(w,{...n,...e}),(0,x.jsx)(b,{...n,...e})]})}function j(e){const n=(0,f.A)();return(0,x.jsx)(y,{...e,children:d(e.children)},String(n))}},8453:(e,n,o)=>{o.d(n,{R:()=>i,x:()=>a});var s=o(6540);const t={},r=s.createContext(t);function i(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:i(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);