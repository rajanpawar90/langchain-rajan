#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "${SCRIPT_DIR}"

# Create a temporary directory for storing the built files
mkdir -p ../_dist
cd ../_dist

# Define a function to copy files from the source directory to the current directory
copy_files() {
  local src_dir="$1"
  local dest_dir="$2"
  local exclude_patterns=("$@" --exclude node_modules --exclude api_reference --exclude .venv --exclude .docusaurus)
  rsync -ruv "${exclude_patterns[@]}" "${src_dir}/" .
}

# Define a function to run a poetry command
get_poetry_run_cmd() {
  if command -v poetry > /dev/null; then
    echo "poetry run $@"
  else
    echo "Poetry not found. Please install poetry and try again."
    exit 1
  fi
}

# Copy the files from the source directory to the current directory
copy_files "../" .

# Run the poetry commands
$(get_poetry_run_cmd python scripts/model_feat_table.py)
cp ../cookbook/README.md src/pages/cookbook.mdx

# Define a function to download a file using wget
download_file() {
  local url="$1"
  local output_file="$2"
  wget -q "$url" -O "$output_file" || { echo "Failed to download $url"; exit 1; }
}

# Download the README files
download_file "https://raw.githubusercontent.com/langchain-ai/langserve/main/README.md" docs/langserve.md
download_file "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md" docs/langgraph.md

# Create the directories for the Quarto renderer
mkdir -p docs/templates
cp ../templates/docs/INDEX.md docs/templates/index.md

# Run the Quarto renderer
$(get_poetry_run_cmd quarto render docs)

# Generate the API reference links
$(get_poetry_run_cmd python scripts/generate_api_reference_links.py --docs_dir docs)

# Install the dependencies using yarn
yarn

# Start the development server using yarn
yarn start

# Remove the temporary directory
trap "cd ..; rm -rf _dist" EXIT
