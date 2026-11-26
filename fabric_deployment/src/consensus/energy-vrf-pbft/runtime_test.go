package consensus

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func runtimeHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func TestRuntimeHealth(t *testing.T) {
	service, err := NewRuntimeService()
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(
		http.MethodGet,
		"/health",
		nil,
	)

	response := httptest.NewRecorder()

	service.Handler().ServeHTTP(
		response,
		request,
	)

	if response.Code != http.StatusOK {
		t.Fatalf(
			"health status=%d",
			response.Code,
		)
	}
}

func TestRuntimeConsensus(t *testing.T) {
	service, err := NewRuntimeService()
	if err != nil {
		t.Fatal(err)
	}

	payload := ConsensusRequest{
		Epoch:             1,
		ShardID:           0,
		SeedHex:           runtimeHash("runtime-seed"),
		PayloadHash:       runtimeHash("synthetic-event"),
		PreviousBlockHash: runtimeHash("previous-block"),
		Timestamp:         "2026-08-17T00:00:00Z",
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/consensus",
		bytes.NewReader(encoded),
	)

	response := httptest.NewRecorder()

	service.Handler().ServeHTTP(
		response,
		request,
	)

	if response.Code != http.StatusOK {
		t.Fatalf(
			"consensus status=%d body=%s",
			response.Code,
			response.Body.String(),
		)
	}

	var result ConsensusResponse

	if err := json.Unmarshal(
		response.Body.Bytes(),
		&result,
	); err != nil {
		t.Fatal(err)
	}

	if result.Certificate == nil {
		t.Fatal("certificate missing")
	}

	if result.Certificate.VoteCount < CertificateThreshold {
		t.Fatalf(
			"vote count=%d",
			result.Certificate.VoteCount,
		)
	}

	if result.Certificate.LogicalMessageCount != 8 {
		t.Fatalf(
			"logical messages=%d want=8",
			result.Certificate.LogicalMessageCount,
		)
	}

	if len(result.CertificateHash) != 64 {
		t.Fatalf(
			"certificate hash length=%d",
			len(result.CertificateHash),
		)
	}
}

func TestRuntimeRejectsInvalidShard(t *testing.T) {
	service, err := NewRuntimeService()
	if err != nil {
		t.Fatal(err)
	}

	payload := ConsensusRequest{
		Epoch:             1,
		ShardID:           9,
		SeedHex:           runtimeHash("runtime-seed"),
		PayloadHash:       runtimeHash("event"),
		PreviousBlockHash: runtimeHash("block"),
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/consensus",
		bytes.NewReader(encoded),
	)

	response := httptest.NewRecorder()

	service.Handler().ServeHTTP(
		response,
		request,
	)

	if response.Code == http.StatusOK {
		t.Fatal("invalid shard accepted")
	}
}
