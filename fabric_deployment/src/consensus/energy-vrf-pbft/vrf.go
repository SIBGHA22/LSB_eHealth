package consensus

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"

	vrf "gitlab.com/yawning/edwards25519-extra/vrf"
)

type VRFKeyPair struct {
	PublicKey  ed25519.PublicKey
	PrivateKey ed25519.PrivateKey
}

type VRFResult struct {
	Output []byte
	Proof  []byte
}

func GenerateVRFKeyPair() (*VRFKeyPair, error) {
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate VRF key: %w", err)
	}

	return &VRFKeyPair{
		PublicKey:  publicKey,
		PrivateKey: privateKey,
	}, nil
}

func ProveVRF(privateKey ed25519.PrivateKey, input []byte) (*VRFResult, error) {
	if len(privateKey) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid VRF private key size")
	}

	if len(input) == 0 {
		return nil, errors.New("VRF input cannot be empty")
	}

	proof := vrf.Prove(privateKey, input)

	publicKey, ok := privateKey.Public().(ed25519.PublicKey)
	if !ok {
		return nil, errors.New("unable to derive VRF public key")
	}

	valid, output := vrf.Verify(publicKey, proof, input)
	if !valid {
		return nil, errors.New("locally generated VRF proof failed verification")
	}

	return &VRFResult{
		Output: output,
		Proof:  proof,
	}, nil
}

func VerifyVRF(
	publicKey ed25519.PublicKey,
	input []byte,
	proof []byte,
) ([]byte, error) {

	if len(publicKey) != ed25519.PublicKeySize {
		return nil, errors.New("invalid VRF public key size")
	}

	if len(input) == 0 {
		return nil, errors.New("VRF input cannot be empty")
	}

	if len(proof) != vrf.ProofSize {
		return nil, fmt.Errorf(
			"invalid VRF proof size: got %d expected %d",
			len(proof),
			vrf.ProofSize,
		)
	}

	valid, output := vrf.Verify(publicKey, proof, input)
	if !valid {
		return nil, errors.New("VRF proof verification failed")
	}

	return output, nil
}
