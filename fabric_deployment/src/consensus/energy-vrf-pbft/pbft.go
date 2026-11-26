package consensus

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

const CommitDecision = "COMMIT"

var sha256Pattern = regexp.MustCompile(`^[a-fA-F0-9]{64}$`)

type Proposal struct {
	ProposalID        string `json:"proposal_id"`
	Epoch             uint64 `json:"epoch"`
	ShardID           int    `json:"shard_id"`
	LeaderID          string `json:"leader_id"`
	PayloadHash       string `json:"payload_hash"`
	PreviousBlockHash string `json:"previous_block_hash"`
	Timestamp         string `json:"timestamp"`
	Signature         string `json:"signature"`
}

type Vote struct {
	ValidatorID string `json:"validator_id"`
	ProposalID  string `json:"proposal_id"`
	Epoch       uint64 `json:"epoch"`
	ShardID     int    `json:"shard_id"`
	Decision    string `json:"decision"`
	Signature   string `json:"signature"`
}

type CommitCertificate struct {
	Proposal            Proposal `json:"proposal"`
	Votes               []Vote   `json:"votes"`
	VoteCount           int      `json:"vote_count"`
	Threshold           int      `json:"threshold"`
	LogicalMessageCount int      `json:"logical_message_count"`
}

func proposalSigningBytes(
	epoch uint64,
	shardID int,
	leaderID string,
	payloadHash string,
	previousBlockHash string,
	timestamp string,
) []byte {

	return []byte(fmt.Sprintf(
		"LSB-EHEALTH|PROPOSAL|epoch=%d|shard=%d|leader=%s|payload=%s|previous=%s|timestamp=%s",
		epoch,
		shardID,
		leaderID,
		strings.ToLower(payloadHash),
		strings.ToLower(previousBlockHash),
		timestamp,
	))
}

func proposalDigest(
	epoch uint64,
	shardID int,
	leaderID string,
	payloadHash string,
	previousBlockHash string,
	timestamp string,
) string {

	sum := sha256.Sum256(
		proposalSigningBytes(
			epoch,
			shardID,
			leaderID,
			payloadHash,
			previousBlockHash,
			timestamp,
		),
	)

	return hex.EncodeToString(sum[:])
}

func voteSigningBytes(vote Vote) []byte {
	return []byte(fmt.Sprintf(
		"LSB-EHEALTH|VOTE|validator=%s|proposal=%s|epoch=%d|shard=%d|decision=%s",
		vote.ValidatorID,
		vote.ProposalID,
		vote.Epoch,
		vote.ShardID,
		vote.Decision,
	))
}

func committeeContains(
	committee *Committee,
	validatorID string,
) bool {

	if committee == nil {
		return false
	}

	for _, member := range committee.Members {
		if member.ValidatorID == validatorID {
			return true
		}
	}

	return false
}

func CreateProposal(
	leader *ValidatorNode,
	committee *Committee,
	payloadHash string,
	previousBlockHash string,
	timestamp string,
) (*Proposal, error) {

	if leader == nil {
		return nil, errors.New("leader is required")
	}

	if committee == nil {
		return nil, errors.New("committee is required")
	}

	if committee.Leader != leader.Info.ID {
		return nil, fmt.Errorf(
			"validator %s is not committee leader %s",
			leader.Info.ID,
			committee.Leader,
		)
	}

	if !sha256Pattern.MatchString(payloadHash) {
		return nil, errors.New("payload hash must be SHA-256 hex")
	}

	if !sha256Pattern.MatchString(previousBlockHash) {
		return nil, errors.New(
			"previous block hash must be SHA-256 hex",
		)
	}

	if _, err := time.Parse(time.RFC3339Nano, timestamp); err != nil {
		return nil, fmt.Errorf("invalid timestamp: %w", err)
	}

	proposalID := proposalDigest(
		committee.Epoch,
		committee.ShardID,
		leader.Info.ID,
		payloadHash,
		previousBlockHash,
		timestamp,
	)

	signature := ed25519.Sign(
		leader.SigningPrivateKey,
		[]byte(proposalID),
	)

	return &Proposal{
		ProposalID:        proposalID,
		Epoch:             committee.Epoch,
		ShardID:           committee.ShardID,
		LeaderID:          leader.Info.ID,
		PayloadHash:       strings.ToLower(payloadHash),
		PreviousBlockHash: strings.ToLower(previousBlockHash),
		Timestamp:         timestamp,
		Signature:         hex.EncodeToString(signature),
	}, nil
}

func VerifyProposal(
	proposal Proposal,
	committee *Committee,
	registry map[string]ValidatorInfo,
) error {

	if committee == nil {
		return errors.New("committee is required")
	}

	if proposal.Epoch != committee.Epoch {
		return errors.New("proposal epoch does not match committee")
	}

	if proposal.ShardID != committee.ShardID {
		return errors.New("proposal shard does not match committee")
	}

	if proposal.LeaderID != committee.Leader {
		return errors.New("proposal signer is not committee leader")
	}

	info, ok := registry[proposal.LeaderID]
	if !ok {
		return errors.New("proposal leader not found in registry")
	}

	if !info.Active {
		return errors.New("proposal leader is inactive")
	}

	expectedID := proposalDigest(
		proposal.Epoch,
		proposal.ShardID,
		proposal.LeaderID,
		proposal.PayloadHash,
		proposal.PreviousBlockHash,
		proposal.Timestamp,
	)

	if expectedID != proposal.ProposalID {
		return errors.New("proposal digest mismatch")
	}

	signature, err := hex.DecodeString(proposal.Signature)
	if err != nil {
		return fmt.Errorf("decode proposal signature: %w", err)
	}

	if len(signature) != ed25519.SignatureSize {
		return errors.New("invalid proposal signature size")
	}

	if !ed25519.Verify(
		info.SigningPublicKey,
		[]byte(proposal.ProposalID),
		signature,
	) {
		return errors.New("proposal signature verification failed")
	}

	return nil
}

func CreateVote(
	validator *ValidatorNode,
	committee *Committee,
	proposal Proposal,
	registry map[string]ValidatorInfo,
) (*Vote, error) {

	if validator == nil {
		return nil, errors.New("validator is required")
	}

	if !validator.Info.Active {
		return nil, errors.New("validator is inactive")
	}

	if !committeeContains(committee, validator.Info.ID) {
		return nil, errors.New(
			"validator is not a committee member",
		)
	}

	if err := VerifyProposal(
		proposal,
		committee,
		registry,
	); err != nil {
		return nil, fmt.Errorf(
			"proposal verification failed: %w",
			err,
		)
	}

	vote := Vote{
		ValidatorID: validator.Info.ID,
		ProposalID:  proposal.ProposalID,
		Epoch:       proposal.Epoch,
		ShardID:     proposal.ShardID,
		Decision:    CommitDecision,
	}

	signature := ed25519.Sign(
		validator.SigningPrivateKey,
		voteSigningBytes(vote),
	)

	vote.Signature = hex.EncodeToString(signature)

	return &vote, nil
}

func VerifyVote(
	vote Vote,
	proposal Proposal,
	committee *Committee,
	registry map[string]ValidatorInfo,
) error {

	if vote.ProposalID != proposal.ProposalID {
		return errors.New("vote proposal ID mismatch")
	}

	if vote.Epoch != proposal.Epoch {
		return errors.New("vote epoch mismatch")
	}

	if vote.ShardID != proposal.ShardID {
		return errors.New("vote shard mismatch")
	}

	if vote.Decision != CommitDecision {
		return errors.New("unsupported vote decision")
	}

	if !committeeContains(committee, vote.ValidatorID) {
		return errors.New(
			"vote signer is not a committee member",
		)
	}

	info, ok := registry[vote.ValidatorID]
	if !ok {
		return errors.New("vote signer not found in registry")
	}

	if !info.Active {
		return errors.New("vote signer is inactive")
	}

	signature, err := hex.DecodeString(vote.Signature)
	if err != nil {
		return fmt.Errorf("decode vote signature: %w", err)
	}

	if len(signature) != ed25519.SignatureSize {
		return errors.New("invalid vote signature size")
	}

	if !ed25519.Verify(
		info.SigningPublicKey,
		voteSigningBytes(vote),
		signature,
	) {
		return errors.New("vote signature verification failed")
	}

	return nil
}

func BuildCommitCertificate(
	committee *Committee,
	proposal Proposal,
	votes []Vote,
	registry map[string]ValidatorInfo,
) (*CommitCertificate, []string, error) {

	if err := VerifyProposal(
		proposal,
		committee,
		registry,
	); err != nil {
		return nil, nil, err
	}

	validVotes := make([]Vote, 0, len(votes))
	rejected := make([]string, 0)
	seen := make(map[string]bool)

	for _, vote := range votes {
		if seen[vote.ValidatorID] {
			rejected = append(
				rejected,
				fmt.Sprintf(
					"%s:duplicate-vote",
					vote.ValidatorID,
				),
			)
			continue
		}

		seen[vote.ValidatorID] = true

		if err := VerifyVote(
			vote,
			proposal,
			committee,
			registry,
		); err != nil {
			rejected = append(
				rejected,
				fmt.Sprintf(
					"%s:%s",
					vote.ValidatorID,
					err.Error(),
				),
			)
			continue
		}

		validVotes = append(validVotes, vote)
	}

	if len(validVotes) < CertificateThreshold {
		return nil, rejected, fmt.Errorf(
			"insufficient valid votes: have %d require %d",
			len(validVotes),
			CertificateThreshold,
		)
	}

	sort.Slice(
		validVotes,
		func(i, j int) bool {
			return validVotes[i].ValidatorID <
				validVotes[j].ValidatorID
		},
	)

	certificate := &CommitCertificate{
		Proposal:            proposal,
		Votes:               validVotes,
		VoteCount:           len(validVotes),
		Threshold:           CertificateThreshold,
		LogicalMessageCount: 1 + len(committee.Members),
	}

	return certificate, rejected, nil
}

func VerifyCommitCertificate(
	certificate CommitCertificate,
	committee *Committee,
	registry map[string]ValidatorInfo,
) error {

	if certificate.Threshold != CertificateThreshold {
		return errors.New("certificate threshold mismatch")
	}

	if certificate.VoteCount != len(certificate.Votes) {
		return errors.New("certificate vote count mismatch")
	}

	if certificate.VoteCount < CertificateThreshold {
		return errors.New("certificate below quorum threshold")
	}

	if certificate.LogicalMessageCount !=
		1+len(committee.Members) {

		return errors.New(
			"certificate logical message count mismatch",
		)
	}

	if err := VerifyProposal(
		certificate.Proposal,
		committee,
		registry,
	); err != nil {
		return err
	}

	seen := make(map[string]bool)

	for _, vote := range certificate.Votes {
		if seen[vote.ValidatorID] {
			return errors.New(
				"duplicate validator in certificate",
			)
		}

		seen[vote.ValidatorID] = true

		if err := VerifyVote(
			vote,
			certificate.Proposal,
			committee,
			registry,
		); err != nil {
			return err
		}
	}

	return nil
}

func CertificateHash(
	certificate CommitCertificate,
) (string, error) {

	copyCertificate := certificate
	copyCertificate.Votes = append(
		[]Vote(nil),
		certificate.Votes...,
	)

	sort.Slice(
		copyCertificate.Votes,
		func(i, j int) bool {
			return copyCertificate.Votes[i].ValidatorID <
				copyCertificate.Votes[j].ValidatorID
		},
	)

	payload, err := json.Marshal(copyCertificate)
	if err != nil {
		return "", fmt.Errorf(
			"marshal certificate: %w",
			err,
		)
	}

	sum := sha256.Sum256(payload)

	return hex.EncodeToString(sum[:]), nil
}
