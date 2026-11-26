package main

import (
	"log"
	"net/http"
	"os"

	storageService "lsb-ehealth/storage-service/service"
)

func main() {

	addr := os.Getenv("LSB_STORAGE_ADDR")

	if addr == "" {
		addr = "127.0.0.1:18084"
	}

	svc := storageService.NewService()

	log.Println("LSB Storage Service listening on", addr)

	if err := http.ListenAndServe(
		addr,
		svc.Handler(),
	); err != nil {
		log.Fatal(err)
	}
}
