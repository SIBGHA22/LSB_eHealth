package consensus

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"
	"strings"
)

type ValidatorInfo struct {
	ID               string
	BatteryFraction  float64
	TrustScore       float64
	Active           bool
	PublicKey        ed25519.PublicKey
	SigningPublicKey ed25519.PublicKey
}

type ValidatorNode struct {
	Info              ValidatorInfo
	PrivateKey        ed25519.PrivateKey
	SigningPrivateKey ed25519.PrivateKey
}

func NewValidatorNode(
	id string,
	batteryFraction float64,
	trustScore float64,
) (*ValidatorNode, error) {

	id = strings.TrimSpace(id)
	if id == "" {
		return nil, errors.New("validator ID is required")
	}

	if batteryFraction < 0 || batteryFraction > 1 {
		return nil, errors.New("battery fraction must be between 0 and 1")
	}

	if trustScore < 0 || trustScore > 1 {
		return nil, errors.New("trust score must be between 0 and 1")
	}

	vrfKeys, err := GenerateVRFKeyPair()
	if err != nil {
		return nil, fmt.Errorf("generate validator VRF keys: %w", err)
	}

	signingPublic, signingPrivate, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate signing key: %w", err)
	}

	return &ValidatorNode{
		Info: ValidatorInfo{
			ID:               id,
			BatteryFraction:  batteryFraction,
			TrustScore:       trustScore,
			Active:           true,
			PublicKey:        vrfKeys.PublicKey,
			SigningPublicKey: signingPublic,
		},
		PrivateKey:        vrfKeys.PrivateKey,
		SigningPrivateKey: signingPrivate,
	}, nil
}

func (v *ValidatorNode) SetBatteryFraction(value float64) error {
	if value < 0 || value > 1 {
		return errors.New("battery fraction must be between 0 and 1")
	}

	v.Info.BatteryFraction = value
	return nil
}

func (v *ValidatorNode) SetTrustScore(value float64) error {
	if value < 0 || value > 1 {
		return errors.New("trust score must be between 0 and 1")
	}

	v.Info.TrustScore = value
	return nil
}

func (v *ValidatorNode) SetActive(active bool) {
	v.Info.Active = active
}

func ValidatorRegistry(
	validators []*ValidatorNode,
) map[string]ValidatorInfo {

	registry := make(map[string]ValidatorInfo)

	for _, validator := range validators {
		if validator == nil {
			continue
		}

		registry[validator.Info.ID] = validator.Info
	}

	return registry
}
