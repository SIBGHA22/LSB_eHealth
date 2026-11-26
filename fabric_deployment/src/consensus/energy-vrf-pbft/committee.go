package consensus

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
)

type SelectionProof struct {
	ValidatorID string  `json:"validator_id"`
	Epoch       uint64  `json:"epoch"`
	ShardID     int     `json:"shard_id"`
	Battery     float64 `json:"battery"`
	Trust       float64 `json:"trust"`
	VRFOutput   string  `json:"vrf_output"`
	VRFProof    string  `json:"vrf_proof"`
}

type CommitteeMember struct {
	ValidatorID string  `json:"validator_id"`
	Battery     float64 `json:"battery"`
	Trust       float64 `json:"trust"`
	VRFOutput   string  `json:"vrf_output"`
	VRFProof    string  `json:"vrf_proof"`
	Score       float64 `json:"score"`
}

type Committee struct {
	Epoch   uint64            `json:"epoch"`
	ShardID int               `json:"shard_id"`
	Leader  string            `json:"leader"`
	Members []CommitteeMember `json:"members"`
}

func ValidateShardID(shardID int) error {
	if shardID < 0 || shardID >= ShardCount {
		return fmt.Errorf(
			"invalid shard ID %d; expected 0-%d",
			shardID,
			ShardCount-1,
		)
	}

	return nil
}

func ShardLaneName(shardID int) (string, error) {
	if err := ValidateShardID(shardID); err != nil {
		return "", err
	}

	return fmt.Sprintf("lsb-shard-%d", shardID), nil
}

func EpochForBlock(blockNumber uint64) uint64 {
	return blockNumber / ReselectionInterval
}

func ShouldReselect(
	previousBlock uint64,
	currentBlock uint64,
) bool {

	if currentBlock <= previousBlock {
		return false
	}

	return EpochForBlock(previousBlock) != EpochForBlock(currentBlock)
}

func selectionInput(
	epoch uint64,
	shardID int,
	seed []byte,
) ([]byte, error) {

	if err := ValidateShardID(shardID); err != nil {
		return nil, err
	}

	if len(seed) == 0 {
		return nil, errors.New("committee seed cannot be empty")
	}

	return []byte(fmt.Sprintf(
		"lsb-ehealth|energy-vrf-pbft|epoch=%d|shard=%d|seed=%x",
		epoch,
		shardID,
		seed,
	)), nil
}

func (v *ValidatorNode) GenerateSelectionProof(
	epoch uint64,
	shardID int,
	seed []byte,
) (*SelectionProof, error) {

	if !v.Info.Active {
		return nil, fmt.Errorf(
			"validator %s is inactive",
			v.Info.ID,
		)
	}

	input, err := selectionInput(epoch, shardID, seed)
	if err != nil {
		return nil, err
	}

	result, err := ProveVRF(v.PrivateKey, input)
	if err != nil {
		return nil, fmt.Errorf(
			"validator %s VRF proof: %w",
			v.Info.ID,
			err,
		)
	}

	return &SelectionProof{
		ValidatorID: v.Info.ID,
		Epoch:       epoch,
		ShardID:     shardID,
		Battery:     v.Info.BatteryFraction,
		Trust:       v.Info.TrustScore,
		VRFOutput:   hex.EncodeToString(result.Output),
		VRFProof:    hex.EncodeToString(result.Proof),
	}, nil
}

func vrfScore(output []byte) (float64, error) {
	if len(output) < 8 {
		return 0, errors.New("VRF output is too short")
	}

	value := binary.BigEndian.Uint64(output[:8])
	maxUint64 := ^uint64(0)

	return float64(value) / float64(maxUint64), nil
}

func VerifyAndScoreCandidate(
	info ValidatorInfo,
	proof SelectionProof,
	seed []byte,
) (*CommitteeMember, error) {

	if !info.Active {
		return nil, fmt.Errorf(
			"validator %s is inactive",
			info.ID,
		)
	}

	if proof.ValidatorID != info.ID {
		return nil, errors.New("validator ID mismatch")
	}

	if proof.Battery < 0 || proof.Battery > 1 {
		return nil, errors.New("invalid reported battery value")
	}

	if proof.Trust < 0 || proof.Trust > 1 {
		return nil, errors.New("invalid reported trust value")
	}

	/*
		For the prototype, battery/trust values in the proof must match
		the coordinator's current validator registry.

		Later the runtime service will update these values through
		authenticated telemetry.
	*/
	if proof.Battery != info.BatteryFraction {
		return nil, errors.New("battery state mismatch")
	}

	if proof.Trust != info.TrustScore {
		return nil, errors.New("trust state mismatch")
	}

	input, err := selectionInput(
		proof.Epoch,
		proof.ShardID,
		seed,
	)
	if err != nil {
		return nil, err
	}

	decodedProof, err := hex.DecodeString(proof.VRFProof)
	if err != nil {
		return nil, fmt.Errorf(
			"decode VRF proof: %w",
			err,
		)
	}

	claimedOutput, err := hex.DecodeString(proof.VRFOutput)
	if err != nil {
		return nil, fmt.Errorf(
			"decode VRF output: %w",
			err,
		)
	}

	verifiedOutput, err := VerifyVRF(
		info.PublicKey,
		input,
		decodedProof,
	)
	if err != nil {
		return nil, err
	}

	if !bytes.Equal(
		claimedOutput,
		verifiedOutput,
	) {
		return nil, errors.New(
			"claimed VRF output does not match verified output",
		)
	}

	randomComponent, err := vrfScore(verifiedOutput)
	if err != nil {
		return nil, err
	}

	score :=
		VRFWeight*randomComponent +
			BatteryWeight*info.BatteryFraction +
			TrustWeight*info.TrustScore

	if info.BatteryFraction >= BatteryPreferenceThreshold {
		score += HighBatteryBonus
	}

	return &CommitteeMember{
		ValidatorID: info.ID,
		Battery:     info.BatteryFraction,
		Trust:       info.TrustScore,
		VRFOutput:   proof.VRFOutput,
		VRFProof:    proof.VRFProof,
		Score:       score,
	}, nil
}

func SelectCommittee(
	validators []*ValidatorNode,
	epoch uint64,
	shardID int,
	seed []byte,
) (*Committee, error) {

	if err := ValidateShardID(shardID); err != nil {
		return nil, err
	}

	var candidates []CommitteeMember

	for _, validator := range validators {
		if validator == nil || !validator.Info.Active {
			continue
		}

		proof, err := validator.GenerateSelectionProof(
			epoch,
			shardID,
			seed,
		)
		if err != nil {
			return nil, err
		}

		member, err := VerifyAndScoreCandidate(
			validator.Info,
			*proof,
			seed,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"verify validator %s: %w",
				validator.Info.ID,
				err,
			)
		}

		candidates = append(candidates, *member)
	}

	if len(candidates) < CommitteeSize {
		return nil, fmt.Errorf(
			"not enough active validators: have %d require %d",
			len(candidates),
			CommitteeSize,
		)
	}

	sort.SliceStable(
		candidates,
		func(i, j int) bool {
			if candidates[i].Score == candidates[j].Score {
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
	}, nil
}
