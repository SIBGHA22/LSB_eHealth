package consensus

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func testHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func findValidator(
	t *testing.T,
	validators []*ValidatorNode,
	id string,
) *ValidatorNode {

	t.Helper()

	for _, validator := range validators {
		if validator.Info.ID == id {
			return validator
		}
	}

	t.Fatalf("validator %s not found", id)
	return nil
}

func createConsensusFixture(
	t *testing.T,
) (
	[]*ValidatorNode,
	*Committee,
	Proposal,
	map[string]ValidatorInfo,
) {

	t.Helper()

	validators := makeValidators(t, 10)

	committee, err := SelectCommittee(
		validators,
		7,
		0,
		[]byte("pbft-test-seed"),
	)
	if err != nil {
		t.Fatal(err)
	}

	registry := ValidatorRegistry(validators)

	leader := findValidator(
		t,
		validators,
		committee.Leader,
	)

	proposal, err := CreateProposal(
		leader,
		committee,
		testHash("synthetic-health-event"),
		testHash("previous-fabric-block"),
		"2026-08-17T00:00:00Z",
	)
	if err != nil {
		t.Fatal(err)
	}

	return validators, committee, *proposal, registry
}

func createAllVotes(
	t *testing.T,
	validators []*ValidatorNode,
	committee *Committee,
	proposal Proposal,
	registry map[string]ValidatorInfo,
) []Vote {

	t.Helper()

	votes := make([]Vote, 0, len(committee.Members))

	for _, member := range committee.Members {
		node := findValidator(
			t,
			validators,
			member.ValidatorID,
		)

		vote, err := CreateVote(
			node,
			committee,
			proposal,
			registry,
		)
		if err != nil {
			t.Fatal(err)
		}

		votes = append(votes, *vote)
	}

	return votes
}

func TestProposalVerification(t *testing.T) {
	_, committee, proposal, registry :=
		createConsensusFixture(t)

	if err := VerifyProposal(
		proposal,
		committee,
		registry,
	); err != nil {
		t.Fatalf(
			"valid proposal rejected: %v",
			err,
		)
	}
}

func TestFiveOfSevenCertificate(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	certificate, rejected, err :=
		BuildCommitCertificate(
			committee,
			proposal,
			votes[:CertificateThreshold],
			registry,
		)

	if err != nil {
		t.Fatal(err)
	}

	if len(rejected) != 0 {
		t.Fatalf(
			"unexpected rejected votes: %v",
			rejected,
		)
	}

	if certificate.VoteCount != 5 {
		t.Fatalf(
			"vote count=%d want=5",
			certificate.VoteCount,
		)
	}

	if certificate.LogicalMessageCount != 8 {
		t.Fatalf(
			"logical messages=%d want=8",
			certificate.LogicalMessageCount,
		)
	}

	if err := VerifyCommitCertificate(
		*certificate,
		committee,
		registry,
	); err != nil {
		t.Fatalf(
			"certificate verification failed: %v",
			err,
		)
	}
}

func TestTwoByzantineVotesTolerated(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	votes[0].Signature = strings.Repeat(
		"00",
		ed25519.SignatureSize,
	)

	votes[1].Signature = strings.Repeat(
		"00",
		ed25519.SignatureSize,
	)

	certificate, rejected, err :=
		BuildCommitCertificate(
			committee,
			proposal,
			votes,
			registry,
		)

	if err != nil {
		t.Fatalf(
			"2 Byzantine validators should be tolerated: %v",
			err,
		)
	}

	if certificate.VoteCount != 5 {
		t.Fatalf(
			"expected 5 valid votes, got %d",
			certificate.VoteCount,
		)
	}

	if len(rejected) != 2 {
		t.Fatalf(
			"expected 2 rejected votes, got %d",
			len(rejected),
		)
	}
}

func TestThreeByzantineVotesFailThreshold(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	for i := 0; i < 3; i++ {
		votes[i].Signature = strings.Repeat(
			"00",
			ed25519.SignatureSize,
		)
	}

	if _, _, err := BuildCommitCertificate(
		committee,
		proposal,
		votes,
		registry,
	); err == nil {
		t.Fatal(
			"certificate accepted with only 4 valid votes",
		)
	}
}

func TestDuplicateVoteRejected(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	votes = append(votes, votes[0])

	certificate, rejected, err :=
		BuildCommitCertificate(
			committee,
			proposal,
			votes,
			registry,
		)

	if err != nil {
		t.Fatal(err)
	}

	if certificate.VoteCount != 7 {
		t.Fatalf(
			"duplicate vote changed quorum count: %d",
			certificate.VoteCount,
		)
	}

	if len(rejected) != 1 {
		t.Fatalf(
			"expected one duplicate rejection, got %d",
			len(rejected),
		)
	}
}

func TestWrongShardVoteRejected(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	votes[0].ShardID = 1

	if err := VerifyVote(
		votes[0],
		proposal,
		committee,
		registry,
	); err == nil {
		t.Fatal("wrong-shard vote accepted")
	}
}

func TestWrongEpochVoteRejected(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	votes[0].Epoch++

	if err := VerifyVote(
		votes[0],
		proposal,
		committee,
		registry,
	); err == nil {
		t.Fatal("wrong-epoch vote accepted")
	}
}

func TestTamperedProposalRejected(t *testing.T) {
	_, committee, proposal, registry :=
		createConsensusFixture(t)

	proposal.PayloadHash =
		testHash("tampered-payload")

	if err := VerifyProposal(
		proposal,
		committee,
		registry,
	); err == nil {
		t.Fatal("tampered proposal accepted")
	}
}

func TestCertificateHashDeterministic(t *testing.T) {
	validators, committee, proposal, registry :=
		createConsensusFixture(t)

	votes := createAllVotes(
		t,
		validators,
		committee,
		proposal,
		registry,
	)

	certificate, _, err :=
		BuildCommitCertificate(
			committee,
			proposal,
			votes,
			registry,
		)
	if err != nil {
		t.Fatal(err)
	}

	first, err := CertificateHash(*certificate)
	if err != nil {
		t.Fatal(err)
	}

	second, err := CertificateHash(*certificate)
	if err != nil {
		t.Fatal(err)
	}

	if first != second {
		t.Fatalf(
			"certificate hash changed: %s != %s",
			first,
			second,
		)
	}

	if len(first) != 64 {
		t.Fatalf(
			"certificate hash length=%d want=64",
			len(first),
		)
	}
}
