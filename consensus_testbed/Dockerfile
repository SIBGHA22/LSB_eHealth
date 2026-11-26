# Static Go binary on an empty base image: no image download needed.
FROM scratch
COPY bin/node /node
EXPOSE 8080
ENTRYPOINT ["/node"]
