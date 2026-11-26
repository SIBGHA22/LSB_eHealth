package main

import (
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	consensus "lsb-ehealth/energy-vrf-pbft"
)

func main() {
	address := os.Getenv("LSB_NETWORK_CONSENSUS_ADDR")

	if address == "" {
		address = "127.0.0.1:18082"
	}

	rawURLs := os.Getenv("LSB_VALIDATOR_URLS")

	if rawURLs == "" {
		rawURLs = strings.Join(
			[]string{
				"http://127.0.0.1:18101",
				"http://127.0.0.1:18102",
				"http://127.0.0.1:18103",
				"http://127.0.0.1:18104",
				"http://127.0.0.1:18105",
				"http://127.0.0.1:18106",
				"http://127.0.0.1:18107",
			},
			",",
		)
	}

	urls := strings.Split(rawURLs, ",")

	service, err :=
		consensus.NewNetworkRuntimeService(urls)

	if err != nil {
		log.Fatalf(
			"initialize network consensus: %v",
			err,
		)
	}

	server := &http.Server{
		Addr:              address,
		Handler:           service.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf(
		"LSB network Energy-VRF-PBFT coordinator listening on %s",
		address,
	)

	if err := server.ListenAndServe(); err != nil &&
		err != http.ErrServerClosed {

		log.Fatal(err)
	}
}
