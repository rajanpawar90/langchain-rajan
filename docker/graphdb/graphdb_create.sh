#! /bin/bash

REPOSITORY_ID="langchain"
GRAPHDB_URI="http://localhost:7200/"
GRAPHDB_BIN="/opt/graphdb/dist/bin/graphdb"

echo -e "\nUsing GraphDB: ${GRAPHDB_URI}"

start_graphdb() {
  echo -e "\nStarting GraphDB..."
  $GRAPHDB_BIN &
}

check_graphdb_status() {
  echo -e "\nWaiting GraphDB to start..."
  for i in {1..30}; do
    curl --silent --output /dev/null --fail --head --write-out '%{http_code}' ${GRAPHDB_URI}/rest/repositories || continue
    if [ "$(curl --silent --output /dev/null --head --write-out '%{http_code}' ${GRAPHDB_URI}/rest/repositories)" = '200' ]; then
      echo -e "\nUp and running"
      return
    fi
    sleep 1
  done
  echo "GraphDB failed to start within the allotted time"
  exit 1
}

start_graphdb
check_graphdb_status
wait
