FROM golang:alpine AS build
RUN go install github.com/pressly/goose/v3/cmd/goose@latest

FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=build /go/bin/goose /usr/local/bin/goose
WORKDIR /app
ENTRYPOINT ["goose"]
