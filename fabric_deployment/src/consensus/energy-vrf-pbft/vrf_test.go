package consensus

import (
	"bytes"
	"testing"

	vrf "gitlab.com/yawning/edwards25519-extra/vrf"
)

func TestConsensusParameters(t *testing.T) {
	if CommitteeSize != 7 {
		t.Fatalf("committee size must be 7")
	}

	if ByzantineTolerance != 2 {
		t.Fatalf("Byzantine tolerance must be 2")
	}

	if CertificateThreshold != 2*ByzantineTolerance+1 {
		t.Fatalf(
			"certificate threshold must equal 2f+1: got %d",
			CertificateThreshold,
		)
	}

	if ShardCount != 4 {
		t.Fatalf("shard count must be 4")
	}
}

func TestVRFProveVerify(t *testing.T) {
	keyPair, err := GenerateVRFKeyPair()
	if err != nil {
		t.Fatalf("GenerateVRFKeyPair failed: %v", err)
	}

	input := []byte("lsb-ehealth:epoch:1:shard:0")

	result, err := ProveVRF(keyPair.PrivateKey, input)
	if err != nil {
		t.Fatalf("ProveVRF failed: %v", err)
	}

	if len(result.Proof) != vrf.ProofSize {
		t.Fatalf(
			"unexpected proof size: got %d expected %d",
			len(result.Proof),
			vrf.ProofSize,
		)
	}

	if len(result.Output) != vrf.OutputSize {
		t.Fatalf(
			"unexpected output size: got %d expected %d",
			len(result.Output),
			vrf.OutputSize,
		)
	}

	verifiedOutput, err := VerifyVRF(
		keyPair.PublicKey,
		input,
		result.Proof,
	)
	if err != nil {
		t.Fatalf("VerifyVRF failed: %v", err)
	}

	if !bytes.Equal(result.Output, verifiedOutput) {
		t.Fatal("VRF prover and verifier outputs differ")
	}
}

func TestVRFWrongInputRejected(t *testing.T) {
	keyPair, err := GenerateVRFKeyPair()
	if err != nil {
		t.Fatal(err)
	}

	result, err := ProveVRF(
		keyPair.PrivateKey,
		[]byte("epoch-1"),
	)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := VerifyVRF(
		keyPair.PublicKey,
		[]byte("epoch-2"),
		result.Proof,
	); err == nil {
		t.Fatal("VRF proof accepted for wrong input")
	}
}

func TestVRFTamperedProofRejected(t *testing.T) {
	keyPair, err := GenerateVRFKeyPair()
	if err != nil {
		t.Fatal(err)
	}

	input := []byte("lsb-ehealth:committee-selection")

	result, err := ProveVRF(keyPair.PrivateKey, input)
	if err != nil {
		t.Fatal(err)
	}

	tampered := append([]byte(nil), result.Proof...)
	tampered[len(tampered)-1] ^= 0x01

	if _, err := VerifyVRF(
		keyPair.PublicKey,
		input,
		tampered,
	); err == nil {
		t.Fatal("tampered VRF proof was accepted")
	}
}
