package main

import (
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	consensus "lsb-ehealth/energy-vrf-pbft"
)

func envFloat(
	name string,
	defaultValue float64,
) float64 {

	value := os.Getenv(name)
	if value == "" {
		return defaultValue
	}

	parsed, err := strconv.ParseFloat(
		value,
		64,
	)
	if err != nil {
		log.Fatalf(
			"invalid %s: %v",
			name,
			err,
		)
	}

	return parsed
}

func main() {
	id := os.Getenv("LSB_VALIDATOR_ID")
	if id == "" {
		id = "validator-01"
	}

	address := os.Getenv("LSB_VALIDATOR_ADDR")
	if address == "" {
		address = "127.0.0.1:18101"
	}

	battery := envFloat(
		"LSB_VALIDATOR_BATTERY",
		0.90,
	)

	trust := envFloat(
		"LSB_VALIDATOR_TRUST",
		0.95,
	)

	service, err := consensus.NewValidatorRuntime(
		id,
		battery,
		trust,
	)
	if err != nil {
		log.Fatalf(
			"initialize validator: %v",
			err,
		)
	}

	server := &http.Server{
		Addr:              address,
		Handler:           service.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf(
		"LSB validator %s listening on %s",
		id,
		address,
	)

	if err := server.ListenAndServe(); err != nil &&
		err != http.ErrServerClosed {

		log.Fatal(err)
	}
}
