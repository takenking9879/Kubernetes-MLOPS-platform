# ── Build stage ────────────────────────────────────────────────────────────────
FROM golang:1.24 AS builder

WORKDIR /app

# Download dependencies first (layer cache friendly).
COPY go.mod go.sum ./
RUN go mod download

# Build a statically linked binary (CGO disabled for distroless compatibility).
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o vk-ml ./cmd/vk-ml

# ── Runtime stage ──────────────────────────────────────────────────────────────
# distroless/static: no shell, no package manager, ~2MB base.
# The binary runs as non-root user 65532 (distroless "nobody").
FROM gcr.io/distroless/static:nonroot

WORKDIR /
COPY --from=builder /app/vk-ml /vk-ml

USER 65532:65532

ENTRYPOINT ["/vk-ml"]
