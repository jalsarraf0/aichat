FROM dart:stable AS build
WORKDIR /app
COPY pubspec.yaml pubspec.lock ./
RUN dart pub get
COPY lib/ lib/
COPY bin/ bin/
RUN dart compile exe bin/server.dart -o bin/dartboard

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libsqlite3-0 \
    && ln -sf /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 /usr/lib/x86_64-linux-gnu/libsqlite3.so \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r dartboard && useradd -r -g dartboard -d /app -s /sbin/nologin dartboard
WORKDIR /app
RUN mkdir -p /data && chown dartboard:dartboard /data
COPY --from=build /app/bin/dartboard /app/bin/dartboard
COPY docker/web/web/ /app/web/
USER dartboard
ENV PORT=8200 DB_PATH=/data/dartboard.db WEB_DIR=/app/web
EXPOSE 8200
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -sf http://localhost:8200/health || exit 1
CMD ["/app/bin/dartboard"]
