package service

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"sync"

	storage "lsb-ehealth/storage-core"
)

type StoreRequest struct {
	ID      string `json:"id"`
	Payload string `json:"payload"`
	AAD     string `json:"aad"`
}

type StoreResponse struct {
	ID        string           `json:"id"`
	Status    string           `json:"status"`
	Fragments int              `json:"fragments"`
	Manifest  storage.Manifest `json:"manifest"`
}

type RetrieveResponse struct {
	ID       string `json:"id"`
	Payload  string `json:"payload"`
	Verified bool   `json:"verified"`
}

type StoredRecord struct {
	Payload *storage.EncodedPayload
	AAD     []byte
}

type Service struct {
	mu      sync.Mutex
	records map[string]*StoredRecord
	key     []byte
}

func NewService() *Service {
	return &Service{
		records: make(map[string]*StoredRecord),
		key:     make([]byte, 32),
	}
}

func (s *Service) Handler() http.Handler {

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {

		json.NewEncoder(w).Encode(
			map[string]string{
				"service": "lsb-storage-service",
				"status":  "ok",
			},
		)
	})

	mux.HandleFunc(
		"/v1/storage/store",
		s.store,
	)

	mux.HandleFunc(
		"/v1/storage/retrieve/",
		s.retrieve,
	)

	return mux
}

func (s *Service) store(
	w http.ResponseWriter,
	r *http.Request,
) {

	var req StoreRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	payload, err := base64.StdEncoding.DecodeString(req.Payload)

	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	encoded, err := storage.Encode(
		payload,
		s.key,
		[]byte(req.AAD),
	)

	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	s.mu.Lock()

	s.records[req.ID] = &StoredRecord{
		Payload: encoded,
		AAD:     []byte(req.AAD),
	}

	s.mu.Unlock()

	json.NewEncoder(w).Encode(
		StoreResponse{
			ID:        req.ID,
			Status:    "stored",
			Fragments: len(encoded.Fragments),
			Manifest:  encoded.Manifest,
		},
	)
}

func (s *Service) retrieve(
	w http.ResponseWriter,
	r *http.Request,
) {

	id := strings.TrimPrefix(
		r.URL.Path,
		"/v1/storage/retrieve/",
	)

	s.mu.Lock()

	record := s.records[id]

	s.mu.Unlock()

	if record == nil {
		http.Error(
			w,
			"not found",
			404,
		)
		return
	}

	fragments := make([][]byte, storage.TotalShards)

	for i, f := range record.Payload.Fragments {

		fragments[i] = f.Data
	}

	data, err := storage.Restore(
		record.Payload.Manifest,
		fragments,
		s.key,
		record.AAD,
	)

	if err != nil {

		http.Error(
			w,
			err.Error(),
			500,
		)

		return
	}

	json.NewEncoder(w).Encode(
		RetrieveResponse{
			ID:       id,
			Payload:  base64.StdEncoding.EncodeToString(data),
			Verified: true,
		},
	)
}
