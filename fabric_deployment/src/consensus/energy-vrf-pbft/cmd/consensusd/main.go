package main

import (
	"log"
	"net/http"
	"os"
	"time"

	consensus "lsb-ehealth/energy-vrf-pbft"
)

func main() {
	address := os.Getenv("LSB_CONSENSUS_ADDR")
	if address == "" {
		address = "127.0.0.1:18081"
	}

	service, err := consensus.NewRuntimeService()
	if err != nil {
		log.Fatalf(
			"initialize consensus runtime: %v",
			err,
		)
	}

	server := &http.Server{
		Addr:              address,
		Handler:           service.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf(
		"LSB Energy-VRF-PBFT runtime listening on %s",
		address,
	)

	if err := server.ListenAndServe(); err != nil &&
		err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
