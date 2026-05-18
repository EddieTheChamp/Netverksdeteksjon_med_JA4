FROM zeek/zeek:latest

USER root

# 1. Install git and ca-certificates so zkg can pull from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN zkg install --force zeek/foxio/ja4

# 2. Explicitly tell Zeek to load the package globally so it's always active
RUN echo "@load packages" >> $(zeek-config --site_dir)/local.zeek

# 3. Modify the config.zeek file to enable JA4 fullstring and JA4S fullstring (local config.zeek must be in same directory as this Dockerfile)
COPY config.zeek /tmp/config.zeek
RUN cp /tmp/config.zeek $(zeek-config --site_dir)/packages/ja4/config.zeek

ENTRYPOINT ["zeek"]