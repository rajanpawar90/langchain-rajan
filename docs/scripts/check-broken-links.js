// Sorry py folks, gotta be js for this one
const { checkBrokenLinks } = require("@langchain/scripts/check_broken_links");
const { exec } = require("child_process");

const command = `node -e "const { checkBrokenLinks } = require('@langchain/scripts/check_broken_links'); checkBrokenLinks('docs', { timeout: 10000, retryFailed: true })" `;

exec(command, (error, stdout, stderr) => {
  if (error) {
    console.error(`exec error: ${error}`);
    return;
  }
  console.log(`stdout: ${stdout}`);
  console.error(`stderr: ${stderr}`);
});
