package consensus

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

type NetworkRuntimeMetrics struct {
	ConsensusRequests      uint64  `json:"consensus_requests"`
	CertificatesIssued     uint64  `json:"certificates_issued"`
	ConsensusFailures      uint64  `json:"consensus_failures"`
	RejectedVotes          uint64  `json:"rejected_votes"`
	LogicalMessages        uint64  `json:"logical_messages"`
	ValidatorHTTPRequests  uint64  `json:"validator_http_requests"`
	LastSelectionLatencyMS float64 `json:"last_selection_latency_ms"`
	LastConsensusLatencyMS float64 `json:"last_consensus_latency_ms"`
	LastTotalLatencyMS     float64 `json:"last_total_latency_ms"`
}

type NetworkConsensusResponse struct {
	Committee             *Committee         `json:"committee"`
	Certificate           *CommitCertificate `json:"certificate"`
	CertificateHash       string             `json:"certificate_hash"`
	RejectedVotes         []string           `json:"rejected_votes"`
	ValidatorHTTPRequests int                `json:"validator_http_requests"`
	SelectionLatencyMS    float64            `json:"selection_latency_ms"`
	ConsensusLatencyMS    float64            `json:"consensus_latency_ms"`
	TotalLatencyMS        float64            `json:"total_latency_ms"`
}

type NetworkRuntimeService struct {
	client *http.Client

	endpoints map[string]string
	registry  map[string]ValidatorInfo

	mu      sync.RWMutex
	metrics NetworkRuntimeMetrics
}

func NewNetworkRuntimeService(
	urls []string,
) (*NetworkRuntimeService, error) {

	if len(urls) < CommitteeSize {
		return nil, fmt.Errorf(
			"need at least %d validator endpoints",
			CommitteeSize,
		)
	}

	service := &NetworkRuntimeService{
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
		endpoints: make(map[string]string),
		registry:  make(map[string]ValidatorInfo),
	}

	for _, baseURL := range urls {
		baseURL = strings.TrimRight(
			strings.TrimSpace(baseURL),
			"/",
		)

		var info ValidatorRuntimeInfo

		if err := service.getJSON(
			baseURL+"/v1/info",
			&info,
		); err != nil {
			return nil, fmt.Errorf(
				"discover validator %s: %w",
				baseURL,
				err,
			)
		}

		if _, exists := service.registry[info.ID]; exists {
			return nil, fmt.Errorf(
				"duplicate validator ID %s",
				info.ID,
			)
		}

		vrfKey, err := hex.DecodeString(info.VRFPublicKey)
		if err != nil || len(vrfKey) != ed25519.PublicKeySize {
			return nil, fmt.Errorf(
				"validator %s has invalid VRF key",
				info.ID,
			)
		}

		signingKey, err := hex.DecodeString(
			info.SigningPublicKey,
		)
		if err != nil ||
			len(signingKey) != ed25519.PublicKeySize {
			return nil, fmt.Errorf(
				"validator %s has invalid signing key",
				info.ID,
			)
		}

		service.registry[info.ID] = ValidatorInfo{
			ID:               info.ID,
			BatteryFraction:  info.Battery,
			TrustScore:       info.Trust,
			Active:           info.Active,
			PublicKey:        ed25519.PublicKey(vrfKey),
			SigningPublicKey: ed25519.PublicKey(signingKey),
		}

		service.endpoints[info.ID] = baseURL
	}

	return service, nil
}

func (s *NetworkRuntimeService) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/v1/consensus", s.handleConsensus)
	mux.HandleFunc("/metrics", s.handleMetrics)

	return mux
}

func (s *NetworkRuntimeService) getJSON(
	url string,
	output interface{},
) error {

	response, err := s.client.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 ||
		response.StatusCode >= 300 {

		body, _ := io.ReadAll(response.Body)

		return fmt.Errorf(
			"HTTP %d: %s",
			response.StatusCode,
			strings.TrimSpace(string(body)),
		)
	}

	return json.NewDecoder(response.Body).Decode(output)
}

func (s *NetworkRuntimeService) postJSON(
	url string,
	input interface{},
	output interface{},
) error {

	payload, err := json.Marshal(input)
	if err != nil {
		return err
	}

	response, err := s.client.Post(
		url,
		"application/json",
		bytes.NewReader(payload),
	)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 ||
		response.StatusCode >= 300 {

		body, _ := io.ReadAll(response.Body)

		return fmt.Errorf(
			"HTTP %d: %s",
			response.StatusCode,
			strings.TrimSpace(string(body)),
		)
	}

	return json.NewDecoder(response.Body).Decode(output)
}

func (s *NetworkRuntimeService) handleHealth(
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
			"service":               "lsb-network-energy-vrf-pbft",
			"mode":                  "networked",
			"registered_validators": len(s.registry),
			"committee_size":        CommitteeSize,
			"byzantine_tolerance":   ByzantineTolerance,
			"certificate_threshold": CertificateThreshold,
			"shards":                ShardCount,
		},
	)
}

func (s *NetworkRuntimeService) handleMetrics(
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

func (s *NetworkRuntimeService) selectNetworkCommittee(
	epoch uint64,
	shardID int,
	seed []byte,
	seedHex string,
) (*Committee, int, error) {

	candidates := make(
		[]CommitteeMember,
		0,
		len(s.registry),
	)

	httpRequests := 0

	ids := make([]string, 0, len(s.registry))
	for id := range s.registry {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		info := s.registry[id]

		if !info.Active {
			continue
		}

		endpoint := s.endpoints[id]

		var proof SelectionProof

		httpRequests++

		err := s.postJSON(
			endpoint+"/v1/vrf",
			VRFProofRequest{
				Epoch:   epoch,
				ShardID: shardID,
				SeedHex: seedHex,
			},
			&proof,
		)
		if err != nil {
			return nil, httpRequests, fmt.Errorf(
				"VRF request %s: %w",
				id,
				err,
			)
		}

		if proof.Epoch != epoch {
			return nil, httpRequests, fmt.Errorf(
				"validator %s returned wrong epoch",
				id,
			)
		}

		if proof.ShardID != shardID {
			return nil, httpRequests, fmt.Errorf(
				"validator %s returned wrong shard",
				id,
			)
		}

		member, err := VerifyAndScoreCandidate(
			info,
			proof,
			seed,
		)
		if err != nil {
			return nil, httpRequests, fmt.Errorf(
				"verify VRF proof from %s: %w",
				id,
				err,
			)
		}

		candidates = append(candidates, *member)
	}

	if len(candidates) < CommitteeSize {
		return nil, httpRequests, fmt.Errorf(
			"only %d valid candidates; require %d",
			len(candidates),
			CommitteeSize,
		)
	}

	sort.SliceStable(
		candidates,
		func(i, j int) bool {
			if candidates[i].Score ==
				candidates[j].Score {

				return candidates[i].ValidatorID <
					candidates[j].ValidatorID
			}

			return candidates[i].Score >
				candidates[j].Score
		},
	)

	selected := append(
		[]CommitteeMember(nil),
		candidates[:CommitteeSize]...,
	)

	return &Committee{
		Epoch:   epoch,
		ShardID: shardID,
		Leader:  selected[0].ValidatorID,
		Members: selected,
	}, httpRequests, nil
}

func committeeIDs(
	committee *Committee,
) []string {

	ids := make(
		[]string,
		0,
		len(committee.Members),
	)

	for _, member := range committee.Members {
		ids = append(ids, member.ValidatorID)
	}

	return ids
}

func (s *NetworkRuntimeService) handleConsensus(
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
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusBadRequest,
			fmt.Errorf("invalid request: %w", err),
		)
		return
	}

	seed, err := decodeSeed(request.SeedHex)
	if err != nil {
		s.recordNetworkFailure()
		writeError(w, http.StatusBadRequest, err)
		return
	}

	if request.Timestamp == "" {
		request.Timestamp = time.Now().
			UTC().
			Format(time.RFC3339Nano)
	}

	selectionStart := time.Now()

	committee, httpRequests, err :=
		s.selectNetworkCommittee(
			request.Epoch,
			request.ShardID,
			seed,
			request.SeedHex,
		)

	if err != nil {
		s.recordNetworkFailure()
		writeError(w, http.StatusBadRequest, err)
		return
	}

	selectionLatency :=
		float64(time.Since(selectionStart).Nanoseconds()) /
			1e6

	consensusStart := time.Now()

	memberIDs := committeeIDs(committee)

	leaderEndpoint, ok :=
		s.endpoints[committee.Leader]

	if !ok {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			errors.New("leader endpoint missing"),
		)
		return
	}

	var proposal Proposal

	httpRequests++

	err = s.postJSON(
		leaderEndpoint+"/v1/proposal",
		ProposalRequest{
			Epoch:              committee.Epoch,
			ShardID:            committee.ShardID,
			PayloadHash:        request.PayloadHash,
			PreviousBlockHash:  request.PreviousBlockHash,
			Timestamp:          request.Timestamp,
			ExpectedLeaderID:   committee.Leader,
			CommitteeMemberIDs: memberIDs,
		},
		&proposal,
	)
	if err != nil {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusBadGateway,
			fmt.Errorf(
				"leader proposal failed: %w",
				err,
			),
		)
		return
	}

	if err := VerifyProposal(
		proposal,
		committee,
		s.registry,
	); err != nil {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusBadGateway,
			fmt.Errorf(
				"remote proposal verification failed: %w",
				err,
			),
		)
		return
	}

	leaderInfo := s.registry[committee.Leader]

	leaderPublicKey :=
		hex.EncodeToString(
			leaderInfo.SigningPublicKey,
		)

	votes := make([]Vote, 0, CommitteeSize)

	rejected := make([]string, 0)

	for _, member := range committee.Members {
		endpoint := s.endpoints[member.ValidatorID]

		var vote Vote

		httpRequests++

		err := s.postJSON(
			endpoint+"/v1/vote",
			VoteRequest{
				Proposal:               proposal,
				CommitteeLeaderID:      committee.Leader,
				CommitteeMemberIDs:     memberIDs,
				LeaderSigningPublicKey: leaderPublicKey,
			},
			&vote,
		)

		if err != nil {
			rejected = append(
				rejected,
				fmt.Sprintf(
					"%s:network:%s",
					member.ValidatorID,
					err.Error(),
				),
			)
			continue
		}

		if err := VerifyVote(
			vote,
			proposal,
			committee,
			s.registry,
		); err != nil {
			rejected = append(
				rejected,
				fmt.Sprintf(
					"%s:%s",
					member.ValidatorID,
					err.Error(),
				),
			)
			continue
		}

		votes = append(votes, vote)
	}

	certificate, buildRejected, err :=
		BuildCommitCertificate(
			committee,
			proposal,
			votes,
			s.registry,
		)

	rejected = append(
		rejected,
		buildRejected...,
	)

	if err != nil {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusConflict,
			fmt.Errorf(
				"certificate failed: %w",
				err,
			),
		)
		return
	}

	if err := VerifyCommitCertificate(
		*certificate,
		committee,
		s.registry,
	); err != nil {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			err,
		)
		return
	}

	certificateHash, err :=
		CertificateHash(*certificate)

	if err != nil {
		s.recordNetworkFailure()
		writeError(
			w,
			http.StatusInternalServerError,
			err,
		)
		return
	}

	consensusLatency :=
		float64(time.Since(consensusStart).Nanoseconds()) /
			1e6

	totalLatency :=
		float64(time.Since(totalStart).Nanoseconds()) /
			1e6

	s.recordNetworkSuccess(
		selectionLatency,
		consensusLatency,
		totalLatency,
		uint64(len(rejected)),
		uint64(certificate.LogicalMessageCount),
		uint64(httpRequests),
	)

	writeJSON(
		w,
		http.StatusOK,
		NetworkConsensusResponse{
			Committee:             committee,
			Certificate:           certificate,
			CertificateHash:       certificateHash,
			RejectedVotes:         rejected,
			ValidatorHTTPRequests: httpRequests,
			SelectionLatencyMS:    selectionLatency,
			ConsensusLatencyMS:    consensusLatency,
			TotalLatencyMS:        totalLatency,
		},
	)
}

func (s *NetworkRuntimeService) recordNetworkFailure() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.metrics.ConsensusRequests++
	s.metrics.ConsensusFailures++
}

func (s *NetworkRuntimeService) recordNetworkSuccess(
	selectionLatency float64,
	consensusLatency float64,
	totalLatency float64,
	rejected uint64,
	logicalMessages uint64,
	httpRequests uint64,
) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.metrics.ConsensusRequests++
	s.metrics.CertificatesIssued++
	s.metrics.RejectedVotes += rejected
	s.metrics.LogicalMessages += logicalMessages
	s.metrics.ValidatorHTTPRequests += httpRequests
	s.metrics.LastSelectionLatencyMS = selectionLatency
	s.metrics.LastConsensusLatencyMS = consensusLatency
	s.metrics.LastTotalLatencyMS = totalLatency
}
