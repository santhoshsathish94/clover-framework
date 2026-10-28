FROM alpine:3.20

RUN addgroup -S agent && adduser -S agent -G agent
WORKDIR /workspace
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown -R agent:agent /workspace
USER agent

ENTRYPOINT ["/entrypoint.sh"]
