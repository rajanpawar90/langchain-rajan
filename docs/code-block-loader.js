#!/usr/bin/env node

/* eslint-disable prefer-template */
/* eslint-disable no-param-reassign */
// eslint-disable-next-line import/no-extraneous-dependencies
const babel = require("@babel/core");
const path = require("path");
const fs = require("fs");

async function webpackLoader(content, map, meta) {
  const cb = this.async();

  if (!this.resourcePath.endsWith(".ts")) {
    cb(null, JSON.stringify({ content, imports: [] }), map, meta);
    return;
  }

  try {
    const result = await babel.parseAsync(content, {
      sourceType: "module",
      filename: this.resourcePath,
    });

    const imports = [];

    result.program.body.forEach((node) => {
      if (node.type === "ImportDeclaration") {
        const source = node.source.value;

        if (!source.startsWith("langchain")) {
          return;
        }

        node.specifiers.forEach((specifier) => {
          if (specifier.type === "ImportSpecifier") {
            const local = specifier.local.name;
            const imported = specifier.imported.name;
            imports.push({ local, imported, source });
          } else {
            throw new Error("Unsupported import type");
          }
        });
      }
    });

    for (const imp of imports) {
      const { imported, source } = imp;
      const moduleName = source.split("/").slice(1).join("_");
      const docsPath = path.resolve(__dirname, "docs", "api", moduleName);
      const available = fs.readdirSync(docsPath, { withFileTypes: true });

      let found;
      try {
        found = (await fs.promises.readdir(docsPath)).find((file) => {
          const filePath = path.resolve(docsPath, file);
          return (
            fs.lstatSync(filePath).isDirectory() &&
            fs.readFileSync(path.resolve(filePath, imported + ".md"), "utf8")
          );
        });
      } catch (err) {
        const errMsg = `Error reading directory: ${err.message}`;
        cb(new Error(errMsg));
        return;
      }

      if (found) {
        imp.docs = `/${path.join("docs", "api", moduleName, found, imported)}`;
      } else {
        throw new Error(
          `Could not find docs for ${source}.${imported} in docs/api/`
        );
      }
    }

    cb(null, JSON.stringify({ content, imports }), map, meta);
  } catch (err) {
    cb(err);
  }
}

module.exports = webpackLoader;
