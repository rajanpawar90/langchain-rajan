#!/bin/bash

# Update and install required packages
set -ex
yum -y update
yum install -y gcc bzip2-devel libffi-devel zlib-devel wget tar gzip

# Install Quarto
QUARTO_VERSION="1.3.450"
QUARTO_DOWNLOAD_URL="https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.tar.gz"
wget -q "${QUARTO_DOWNLOAD_URL}"
tar -xzf "quarto-${QUARTO_VERSION}-linux-amd64.tar.gz"
export PATH="$PATH:$(pwd)/quarto-${QUARTO_VERSION}/bin/"

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Install required Python packages
python_packages=("uv" "vercel_requirements.txt" "-r")
libs_partners=("$(ls ../libs/partners | grep -vE 'airbyte|ibm|.md' | xargs -I {} echo '../libs/partners/{}')")
for lib in "${libs_partners[@]}"; do
    python3 -m uv pip install -e "${lib}"
done

# Autogenerate integration tables
python3 scripts/model_feat_table.py

# Copy external files
mkdir -p docs/templates
cp ../templates/docs/INDEX.md docs/templates/index.md
python3 scripts/copy_templates.py

# Copy cookbook README
cp ../cookbook/README.md src/pages/cookbook.mdx

# Download and process READMEs
for project in "langserve" "langgraph"; do
    REPO_README_URL="https://raw.githubusercontent.com/langchain-ai/${project}/main/README.md"
    OUTPUT_FILE="docs/${project}.md"
    wget -q "${REPO_README_URL}" -O "${OUTPUT_FILE}"
    python3 scripts/resolve_local_links.py "${OUTPUT_FILE}" "${REPO_README_URL}/tree/main/"
done

# Render the documentation
quarto render docs/
python3 scripts/generate_api_reference_links.py --docs_dir docs
