# Use the latest version of the official GraphDB image
FROM ontotext/graphdb:latest

# Create the required directory for the repository
RUN mkdir -p /data/repositories/langchain

# Copy the configuration file to the appropriate location
COPY config.ttl /data/repositories/langchain/

# Copy the script to run the container
COPY graphdb_create.sh /graphdb_create.sh

# Make the script executable
RUN chmod +x /graphdb_create.sh

# Set the entrypoint to run the script
ENTRYPOINT ["bash", "/graphdb_create.sh"]
