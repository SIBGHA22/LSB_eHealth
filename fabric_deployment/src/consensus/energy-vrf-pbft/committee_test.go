package consensus

import (
	"testing"
)

func makeValidators(
	t *testing.T,
	count int,
) []*ValidatorNode {

	t.Helper()

	validators := make([]*ValidatorNode, 0, count)

	for i := 0; i < count; i++ {
		/*
			Intentional heterogeneous energy/trust state.

			Some validators are below the 60% battery preference
			threshold. They remain valid candidates.
		*/
		battery := 0.35 + float64(i%6)*0.10
		trust := 0.70 + float64(i%4)*0.05

		node, err := NewValidatorNode(
			"validator-"+string(rune('A'+i)),
			battery,
			trust,
		)
		if err != nil {
			t.Fatalf(
				"create validator %d: %v",
				i,
				err,
			)
		}

		validators = append(validators, node)
	}

	return validators
}

func TestShardLanes(t *testing.T) {
	expected := []string{
		"lsb-shard-0",
		"lsb-shard-1",
		"lsb-shard-2",
		"lsb-shard-3",
	}

	for shardID, want := range expected {
		got, err := ShardLaneName(shardID)
		if err != nil {
			t.Fatalf(
				"ShardLaneName(%d): %v",
				shardID,
				err,
			)
		}

		if got != want {
			t.Fatalf(
				"got %s want %s",
				got,
				want,
			)
		}
	}

	if _, err := ShardLaneName(4); err == nil {
		t.Fatal("invalid shard 4 accepted")
	}
}

func TestEpochRotation(t *testing.T) {
	tests := []struct {
		block uint64
		epoch uint64
	}{
		{0, 0},
		{1, 0},
		{99, 0},
		{100, 1},
		{199, 1},
		{200, 2},
	}

	for _, tc := range tests {
		if got := EpochForBlock(tc.block); got != tc.epoch {
			t.Fatalf(
				"block %d: epoch=%d want=%d",
				tc.block,
				got,
				tc.epoch,
			)
		}
	}

	if ShouldReselect(98, 99) {
		t.Fatal("unexpected reselection before boundary")
	}

	if !ShouldReselect(99, 100) {
		t.Fatal("expected reselection at block 100")
	}
}

func TestSelectCommittee(t *testing.T) {
	validators := makeValidators(t, 10)

	committee, err := SelectCommittee(
		validators,
		1,
		0,
		[]byte("block-hash-seed-001"),
	)
	if err != nil {
		t.Fatalf("SelectCommittee failed: %v", err)
	}

	if len(committee.Members) != CommitteeSize {
		t.Fatalf(
			"committee size=%d want=%d",
			len(committee.Members),
			CommitteeSize,
		)
	}

	if committee.Leader == "" {
		t.Fatal("committee leader not selected")
	}

	seen := map[string]bool{}

	for _, member := range committee.Members {
		if seen[member.ValidatorID] {
			t.Fatalf(
				"duplicate validator %s",
				member.ValidatorID,
			)
		}

		seen[member.ValidatorID] = true
	}
}

func TestSelectionDeterministicForFixedState(t *testing.T) {
	validators := makeValidators(t, 10)

	seed := []byte("deterministic-selection-seed")

	first, err := SelectCommittee(
		validators,
		3,
		2,
		seed,
	)
	if err != nil {
		t.Fatal(err)
	}

	second, err := SelectCommittee(
		validators,
		3,
		2,
		seed,
	)
	if err != nil {
		t.Fatal(err)
	}

	if first.Leader != second.Leader {
		t.Fatalf(
			"leader changed: %s != %s",
			first.Leader,
			second.Leader,
		)
	}

	for i := range first.Members {
		if first.Members[i].ValidatorID !=
			second.Members[i].ValidatorID {

			t.Fatalf(
				"committee changed at position %d",
				i,
			)
		}
	}
}

func TestLowBatteryValidatorNotHardRejected(t *testing.T) {
	node, err := NewValidatorNode(
		"low-energy-validator",
		0.40,
		0.95,
	)
	if err != nil {
		t.Fatal(err)
	}

	seed := []byte("energy-soft-preference")

	proof, err := node.GenerateSelectionProof(
		1,
		0,
		seed,
	)
	if err != nil {
		t.Fatal(err)
	}

	member, err := VerifyAndScoreCandidate(
		node.Info,
		*proof,
		seed,
	)
	if err != nil {
		t.Fatalf(
			"low battery validator was hard rejected: %v",
			err,
		)
	}

	if member.Score <= 0 {
		t.Fatal("candidate score must be positive")
	}
}

func TestInactiveValidatorExcluded(t *testing.T) {
	validators := makeValidators(t, 8)

	validators[0].SetActive(false)
	validators[1].SetActive(false)

	_, err := SelectCommittee(
		validators,
		1,
		1,
		[]byte("inactive-validator-test"),
	)

	if err == nil {
		t.Fatal(
			"selection should fail with fewer than 7 active validators",
		)
	}
}

func TestTamperedSelectionProofRejected(t *testing.T) {
	node, err := NewValidatorNode(
		"validator-tamper-test",
		0.80,
		0.90,
	)
	if err != nil {
		t.Fatal(err)
	}

	seed := []byte("tamper-selection-proof")

	proof, err := node.GenerateSelectionProof(
		5,
		3,
		seed,
	)
	if err != nil {
		t.Fatal(err)
	}

	originalProof := proof.VRFProof

	if proof.VRFProof[0] == '0' {
		proof.VRFProof = "1" + proof.VRFProof[1:]
	} else {
		proof.VRFProof = "0" + proof.VRFProof[1:]
	}

	if proof.VRFProof == originalProof {
		t.Fatal("proof mutation did not change proof")
	}

	if _, err := VerifyAndScoreCandidate(
		node.Info,
		*proof,
		seed,
	); err == nil {
		t.Fatal("tampered selection proof accepted")
	}
}
