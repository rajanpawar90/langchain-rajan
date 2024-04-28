#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# Check for imports from langchain, langchain_experimental, or langchain_community
if git --no-pager grep -rE '^from (langchain|langchain_experimental|langchain_community)\.' . > /dev/null; then
    errors=$((errors+1))
fi

# Exit with a status code of 1 if any errors were found
if [ "$errors" -ne 0 ]; then
    exit 1
fi

# Exit with a status code of 0 if no errors were found
exit 0
