package consensus

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type RuntimeMetrics struct {
	ConsensusRequests      uint64  `json:"consensus_requests"`
	CertificatesIssued     uint64  `json:"certificates_issued"`
	ConsensusFailures      uint64  `json:"consensus_failures"`
	RejectedVotes          uint64  `json:"rejected_votes"`
	LogicalMessages        uint64  `json:"logical_messages"`
	LastSelectionLatencyMS float64 `json:"last_selection_latency_ms"`
	LastConsensusLatencyMS float64 `json:"last_consensus_latency_ms"`
	LastTotalLatencyMS     float64 `json:"last_total_latency_ms"`
}

type RuntimeService struct {
	validators []*ValidatorNode

	mu      sync.RWMutex
	metrics RuntimeMetrics
}

type ConsensusRequest struct {
	Epoch             uint64 `json:"epoch"`
	ShardID           int    `json:"shard_id"`
	SeedHex           string `json:"seed_hex"`
	PayloadHash       string `json:"payload_hash"`
	PreviousBlockHash string `json:"previous_block_hash"`
	Timestamp         string `json:"timestamp,omitempty"`
}

type ConsensusResponse struct {
	Committee          *Committee         `json:"committee"`
	Certificate        *CommitCertificate `json:"certificate"`
	CertificateHash    string             `json:"certificate_hash"`
	RejectedVotes      []string           `json:"rejected_votes"`
	SelectionLatencyMS float64            `json:"selection_latency_ms"`
	ConsensusLatencyMS float64            `json:"consensus_latency_ms"`
	TotalLatencyMS     float64            `json:"total_latency_ms"`
}

func NewRuntimeService() (*RuntimeService, error) {
	profiles := []struct {
		ID      string
		Battery float64
		Trust   float64
	}{
		{"validator-01", 0.92, 0.96},
		{"validator-02", 0.87, 0.94},
		{"validator-03", 0.81, 0.93},
		{"validator-04", 0.74, 0.91},
		{"validator-05", 0.68, 0.90},
		{"validator-06", 0.56, 0.88},
		{"validator-07", 0.44, 0.86},
	}

	validators := make([]*ValidatorNode, 0, len(profiles))

	for _, profile := range profiles {
		node, err := NewValidatorNode(
			profile.ID,
			profile.Battery,
			profile.Trust,
		)
		if err != nil {
			return nil, err
		}

		validators = append(validators, node)
	}

	return &RuntimeService{
		validators: validators,
	}, nil
}

func (s *RuntimeService) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/v1/validators", s.handleValidators)
	mux.HandleFunc("/v1/consensus", s.handleConsensus)
	mux.HandleFunc("/metrics", s.handleMetrics)

	return mux
}

func writeJSON(
	w http.ResponseWriter,
	status int,
	value interface{},
) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	_ = json.NewEncoder(w).Encode(value)
}

func writeError(
	w http.ResponseWriter,
	status int,
	err error,
) {
	writeJSON(
		w,
		status,
		map[string]string{
			"error": err.Error(),
		},
	)
}

func (s *RuntimeService) handleHealth(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodGet {
		writeError(
			w,
			http.StatusMethodNotAllowed,
			errors.New("method not allowed"),
		)
		return
	}

	writeJSON(
		w,
		http.StatusOK,
		map[string]interface{}{
			"status":                "ok",
			"service":               "lsb-energy-vrf-pbft",
			"validators":            len(s.validators),
			"committee_size":        CommitteeSize,
			"byzantine_tolerance":   ByzantineTolerance,
			"certificate_threshold": CertificateThreshold,
			"shards":                ShardCount,
			"reselection_interval":  ReselectionInterval,
		},
	)
}

func (s *RuntimeService) handleValidators(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodGet {
		writeError(
			w,
			http.StatusMethodNotAllowed,
			errors.New("method not allowed"),
		)
		return
	}

	type validatorResponse struct {
		ID         string  `json:"id"`
		Battery    float64 `json:"battery"`
		Trust      float64 `json:"trust"`
		Active     bool    `json:"active"`
		VRFKey     string  `json:"vrf_public_key"`
		SigningKey string  `json:"signing_public_key"`
	}

	out := make([]validatorResponse, 0, len(s.validators))

	for _, validator := range s.validators {
		out = append(
			out,
			validatorResponse{
				ID:         validator.Info.ID,
				Battery:    validator.Info.BatteryFraction,
				Trust:      validator.Info.TrustScore,
				Active:     validator.Info.Active,
				VRFKey:     hex.EncodeToString(validator.Info.PublicKey),
				SigningKey: hex.EncodeToString(validator.Info.SigningPublicKey),
			},
		)
	}

	writeJSON(w, http.StatusOK, out)
}

func (s *RuntimeService) handleMetrics(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodGet {
		writeError(
			w,
			http.StatusMethodNotAllowed,
			errors.New("method not allowed"),
		)
		return
	}

	s.mu.RLock()
	metrics := s.metrics
	s.mu.RUnlock()

	writeJSON(w, http.StatusOK, metrics)
}

func decodeSeed(value string) ([]byte, error) {
	value = strings.TrimSpace(value)

	if len(value) != 64 {
		return nil, errors.New(
			"seed_hex must contain exactly 64 hexadecimal characters",
		)
	}

	seed, err := hex.DecodeString(value)
	if err != nil {
		return nil, errors.New("seed_hex is not valid hexadecimal")
	}

	return seed, nil
}

func (s *RuntimeService) handleConsensus(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodPost {
		writeError(
			w,
			http.StatusMethodNotAllowed,
			errors.New("method not allowed"),
		)
		return
	}

	totalStart := time.Now()

	var request ConsensusRequest

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&request); err != nil {
		s.recordFailure()
		writeError(
			w,
			http.StatusBadRequest,
			fmt.Errorf("invalid request: %w", err),
		)
		return
	}

	seed, err := decodeSeed(request.SeedHex)
	if err != nil {
		s.recordFailure()
		writeError(w, http.StatusBadRequest, err)
		return
	}

	if request.Timestamp == "" {
		request.Timestamp = time.Now().
			UTC().
			Format(time.RFC3339Nano)
	}

	selectionStart := time.Now()

	committee, err := SelectCommittee(
		s.validators,
		request.Epoch,
		request.ShardID,
		seed,
	)
	if err != nil {
		s.recordFailure()
		writeError(w, http.StatusBadRequest, err)
		return
	}

	selectionLatency :=
		float64(time.Since(selectionStart).Nanoseconds()) / 1e6

	consensusStart := time.Now()

	registry := ValidatorRegistry(s.validators)

	var leader *ValidatorNode

	for _, validator := range s.validators {
		if validator.Info.ID == committee.Leader {
			leader = validator
			break
		}
	}

	if leader == nil {
		s.recordFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			errors.New("committee leader not found"),
		)
		return
	}

	proposal, err := CreateProposal(
		leader,
		committee,
		request.PayloadHash,
		request.PreviousBlockHash,
		request.Timestamp,
	)
	if err != nil {
		s.recordFailure()
		writeError(w, http.StatusBadRequest, err)
		return
	}

	votes := make([]Vote, 0, len(committee.Members))

	for _, member := range committee.Members {
		var node *ValidatorNode

		for _, validator := range s.validators {
			if validator.Info.ID == member.ValidatorID {
				node = validator
				break
			}
		}

		if node == nil {
			continue
		}

		vote, err := CreateVote(
			node,
			committee,
			*proposal,
			registry,
		)
		if err != nil {
			continue
		}

		votes = append(votes, *vote)
	}

	certificate, rejected, err :=
		BuildCommitCertificate(
			committee,
			*proposal,
			votes,
			registry,
		)

	if err != nil {
		s.recordFailure()
		writeError(
			w,
			http.StatusConflict,
			err,
		)
		return
	}

	if err := VerifyCommitCertificate(
		*certificate,
		committee,
		registry,
	); err != nil {
		s.recordFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			err,
		)
		return
	}

	certificateHash, err := CertificateHash(*certificate)
	if err != nil {
		s.recordFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			err,
		)
		return
	}

	consensusLatency :=
		float64(time.Since(consensusStart).Nanoseconds()) / 1e6

	totalLatency :=
		float64(time.Since(totalStart).Nanoseconds()) / 1e6

	s.recordSuccess(
		selectionLatency,
		consensusLatency,
		totalLatency,
		uint64(len(rejected)),
		uint64(certificate.LogicalMessageCount),
	)

	writeJSON(
		w,
		http.StatusOK,
		ConsensusResponse{
			Committee:          committee,
			Certificate:        certificate,
			CertificateHash:    certificateHash,
			RejectedVotes:      rejected,
			SelectionLatencyMS: selectionLatency,
			ConsensusLatencyMS: consensusLatency,
			TotalLatencyMS:     totalLatency,
		},
	)
}

func (s *RuntimeService) recordFailure() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.metrics.ConsensusRequests++
	s.metrics.ConsensusFailures++
}

func (s *RuntimeService) recordSuccess(
	selectionLatency float64,
	consensusLatency float64,
	totalLatency float64,
	rejected uint64,
	logicalMessages uint64,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.metrics.ConsensusRequests++
	s.metrics.CertificatesIssued++
	s.metrics.RejectedVotes += rejected
	s.metrics.LogicalMessages += logicalMessages
	s.metrics.LastSelectionLatencyMS = selectionLatency
	s.metrics.LastConsensusLatencyMS = consensusLatency
	s.metrics.LastTotalLatencyMS = totalLatency
}
