package consensus

import (
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

type ValidatorRuntime struct {
	Node *ValidatorNode
}

type ValidatorRuntimeInfo struct {
	ID               string  `json:"id"`
	Battery          float64 `json:"battery"`
	Trust            float64 `json:"trust"`
	Active           bool    `json:"active"`
	VRFPublicKey     string  `json:"vrf_public_key"`
	SigningPublicKey string  `json:"signing_public_key"`
}

type VRFProofRequest struct {
	Epoch   uint64 `json:"epoch"`
	ShardID int    `json:"shard_id"`
	SeedHex string `json:"seed_hex"`
}

type ProposalRequest struct {
	Epoch              uint64   `json:"epoch"`
	ShardID            int      `json:"shard_id"`
	PayloadHash        string   `json:"payload_hash"`
	PreviousBlockHash  string   `json:"previous_block_hash"`
	Timestamp          string   `json:"timestamp"`
	ExpectedLeaderID   string   `json:"expected_leader_id"`
	CommitteeMemberIDs []string `json:"committee_member_ids"`
}

type VoteRequest struct {
	Proposal               Proposal `json:"proposal"`
	CommitteeLeaderID      string   `json:"committee_leader_id"`
	CommitteeMemberIDs     []string `json:"committee_member_ids"`
	LeaderSigningPublicKey string   `json:"leader_signing_public_key"`
}

func NewValidatorRuntime(
	id string,
	battery float64,
	trust float64,
) (*ValidatorRuntime, error) {

	node, err := NewValidatorNode(id, battery, trust)
	if err != nil {
		return nil, err
	}

	return &ValidatorRuntime{Node: node}, nil
}

func (v *ValidatorRuntime) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", v.handleHealth)
	mux.HandleFunc("/v1/info", v.handleInfo)
	mux.HandleFunc("/v1/vrf", v.handleVRF)
	mux.HandleFunc("/v1/proposal", v.handleProposal)
	mux.HandleFunc("/v1/vote", v.handleVote)

	return mux
}

func (v *ValidatorRuntime) info() ValidatorRuntimeInfo {
	return ValidatorRuntimeInfo{
		ID:               v.Node.Info.ID,
		Battery:          v.Node.Info.BatteryFraction,
		Trust:            v.Node.Info.TrustScore,
		Active:           v.Node.Info.Active,
		VRFPublicKey:     hex.EncodeToString(v.Node.Info.PublicKey),
		SigningPublicKey: hex.EncodeToString(v.Node.Info.SigningPublicKey),
	}
}

func (v *ValidatorRuntime) handleHealth(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "ok",
		"service":   "lsb-validator",
		"validator": v.Node.Info.ID,
		"active":    v.Node.Info.Active,
	})
}

func (v *ValidatorRuntime) handleInfo(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}

	writeJSON(w, http.StatusOK, v.info())
}

func (v *ValidatorRuntime) handleVRF(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}

	var request VRFProofRequest

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request: %w", err))
		return
	}

	seed, err := decodeSeed(request.SeedHex)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	proof, err := v.Node.GenerateSelectionProof(
		request.Epoch,
		request.ShardID,
		seed,
	)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	writeJSON(w, http.StatusOK, proof)
}

func containsValidatorID(ids []string, target string) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}

	return false
}

func (v *ValidatorRuntime) handleProposal(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}

	var request ProposalRequest

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request: %w", err))
		return
	}

	if request.ExpectedLeaderID != v.Node.Info.ID {
		writeError(
			w,
			http.StatusForbidden,
			errors.New("validator is not selected leader"),
		)
		return
	}

	if !containsValidatorID(
		request.CommitteeMemberIDs,
		v.Node.Info.ID,
	) {
		writeError(
			w,
			http.StatusForbidden,
			errors.New("leader is not in committee"),
		)
		return
	}

	if len(request.CommitteeMemberIDs) != CommitteeSize {
		writeError(
			w,
			http.StatusBadRequest,
			fmt.Errorf(
				"committee must contain %d members",
				CommitteeSize,
			),
		)
		return
	}

	members := make(
		[]CommitteeMember,
		0,
		len(request.CommitteeMemberIDs),
	)

	for _, id := range request.CommitteeMemberIDs {
		members = append(
			members,
			CommitteeMember{
				ValidatorID: id,
			},
		)
	}

	committee := &Committee{
		Epoch:   request.Epoch,
		ShardID: request.ShardID,
		Leader:  v.Node.Info.ID,
		Members: members,
	}

	proposal, err := CreateProposal(
		v.Node,
		committee,
		request.PayloadHash,
		request.PreviousBlockHash,
		request.Timestamp,
	)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	writeJSON(w, http.StatusOK, proposal)
}

func (v *ValidatorRuntime) handleVote(
	w http.ResponseWriter,
	r *http.Request,
) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}

	var request VoteRequest

	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request: %w", err))
		return
	}

	if !v.Node.Info.Active {
		writeError(w, http.StatusForbidden, errors.New("validator is inactive"))
		return
	}

	if request.CommitteeLeaderID != request.Proposal.LeaderID {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("committee leader mismatch"),
		)
		return
	}

	if !containsValidatorID(
		request.CommitteeMemberIDs,
		request.CommitteeLeaderID,
	) {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("committee leader is not committee member"),
		)
		return
	}

	if !containsValidatorID(
		request.CommitteeMemberIDs,
		v.Node.Info.ID,
	) {
		writeError(
			w,
			http.StatusForbidden,
			errors.New("validator is not in committee"),
		)
		return
	}

	expectedProposalID := proposalDigest(
		request.Proposal.Epoch,
		request.Proposal.ShardID,
		request.Proposal.LeaderID,
		request.Proposal.PayloadHash,
		request.Proposal.PreviousBlockHash,
		request.Proposal.Timestamp,
	)

	if expectedProposalID != request.Proposal.ProposalID {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("proposal digest mismatch"),
		)
		return
	}

	publicKeyBytes, err := hex.DecodeString(
		strings.TrimSpace(request.LeaderSigningPublicKey),
	)

	if err != nil ||
		len(publicKeyBytes) != ed25519.PublicKeySize {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("invalid leader signing public key"),
		)
		return
	}

	signature, err := hex.DecodeString(request.Proposal.Signature)

	if err != nil ||
		len(signature) != ed25519.SignatureSize {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("invalid proposal signature"),
		)
		return
	}

	if !ed25519.Verify(
		ed25519.PublicKey(publicKeyBytes),
		[]byte(request.Proposal.ProposalID),
		signature,
	) {
		writeError(
			w,
			http.StatusBadRequest,
			errors.New("leader proposal signature verification failed"),
		)
		return
	}

	vote := Vote{
		ValidatorID: v.Node.Info.ID,
		ProposalID:  request.Proposal.ProposalID,
		Epoch:       request.Proposal.Epoch,
		ShardID:     request.Proposal.ShardID,
		Decision:    CommitDecision,
	}

	voteSignature := ed25519.Sign(
		v.Node.SigningPrivateKey,
		voteSigningBytes(vote),
	)

	vote.Signature = hex.EncodeToString(voteSignature)

	writeJSON(w, http.StatusOK, vote)
}
