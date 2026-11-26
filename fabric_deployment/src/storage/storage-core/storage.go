package storage

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"

	"github.com/klauspost/compress/zstd"
	"github.com/klauspost/reedsolomon"
)

const (
	DataShards   = 7
	ParityShards = 3
	TotalShards  = DataShards + ParityShards
	KeySize      = 32
)

type Fragment struct {
	Index  int    `json:"index"`
	SHA256 string `json:"sha256"`
	Data   []byte `json:"-"`
}

type Manifest struct {
	Version          string   `json:"version"`
	DataShards       int      `json:"data_shards"`
	ParityShards     int      `json:"parity_shards"`
	TotalShards      int      `json:"total_shards"`
	OriginalSize     int      `json:"original_size"`
	CompressedSize   int      `json:"compressed_size"`
	CiphertextSize   int      `json:"ciphertext_size"`
	NonceHex         string   `json:"nonce_hex"`
	AAD              string   `json:"aad"`
	PlaintextSHA256  string   `json:"plaintext_sha256"`
	CiphertextSHA256 string   `json:"ciphertext_sha256"`
	FragmentSHA256   []string `json:"fragment_sha256"`
	ErasureOverhead  float64  `json:"erasure_overhead_ratio"`
	CompressionRatio float64  `json:"compression_ratio"`
}

type EncodedPayload struct {
	Manifest  Manifest
	Fragments []Fragment
}

func hashHex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func validateKey(key []byte) error {
	if len(key) != KeySize {
		return fmt.Errorf(
			"AES-256 key must be exactly %d bytes",
			KeySize,
		)
	}
	return nil
}

func compressPayload(data []byte) ([]byte, error) {
	encoder, err := zstd.NewWriter(nil)
	if err != nil {
		return nil, fmt.Errorf("create zstd encoder: %w", err)
	}
	defer encoder.Close()

	return encoder.EncodeAll(data, nil), nil
}

func decompressPayload(data []byte) ([]byte, error) {
	decoder, err := zstd.NewReader(nil)
	if err != nil {
		return nil, fmt.Errorf("create zstd decoder: %w", err)
	}
	defer decoder.Close()

	out, err := decoder.DecodeAll(data, nil)
	if err != nil {
		return nil, fmt.Errorf("zstd decompress: %w", err)
	}

	return out, nil
}

func encryptAESGCM(
	key []byte,
	plaintext []byte,
	aad []byte,
) ([]byte, []byte, error) {

	if err := validateKey(key); err != nil {
		return nil, nil, err
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, err
	}

	nonce := make([]byte, gcm.NonceSize())

	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, nil, fmt.Errorf(
			"generate AES-GCM nonce: %w",
			err,
		)
	}

	ciphertext := gcm.Seal(
		nil,
		nonce,
		plaintext,
		aad,
	)

	return ciphertext, nonce, nil
}

func decryptAESGCM(
	key []byte,
	nonce []byte,
	ciphertext []byte,
	aad []byte,
) ([]byte, error) {

	if err := validateKey(key); err != nil {
		return nil, err
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	if len(nonce) != gcm.NonceSize() {
		return nil, errors.New(
			"invalid AES-GCM nonce length",
		)
	}

	plaintext, err := gcm.Open(
		nil,
		nonce,
		ciphertext,
		aad,
	)
	if err != nil {
		return nil, errors.New(
			"AES-GCM authentication failed",
		)
	}

	return plaintext, nil
}

func Encode(
	plaintext []byte,
	key []byte,
	aad []byte,
) (*EncodedPayload, error) {

	if len(plaintext) == 0 {
		return nil, errors.New(
			"payload cannot be empty",
		)
	}

	if err := validateKey(key); err != nil {
		return nil, err
	}

	compressed, err := compressPayload(plaintext)
	if err != nil {
		return nil, err
	}

	ciphertext, nonce, err := encryptAESGCM(
		key,
		compressed,
		aad,
	)
	if err != nil {
		return nil, err
	}

	encoder, err := reedsolomon.New(
		DataShards,
		ParityShards,
	)
	if err != nil {
		return nil, err
	}

	shards, err := encoder.Split(ciphertext)
	if err != nil {
		return nil, fmt.Errorf(
			"split ciphertext: %w",
			err,
		)
	}

	if err := encoder.Encode(shards); err != nil {
		return nil, fmt.Errorf(
			"encode parity shards: %w",
			err,
		)
	}

	fragments := make(
		[]Fragment,
		TotalShards,
	)

	hashes := make(
		[]string,
		TotalShards,
	)

	totalFragmentBytes := 0

	for i, shard := range shards {
		copyShard := append(
			[]byte(nil),
			shard...,
		)

		hash := hashHex(copyShard)

		fragments[i] = Fragment{
			Index:  i,
			SHA256: hash,
			Data:   copyShard,
		}

		hashes[i] = hash
		totalFragmentBytes += len(copyShard)
	}

	compressionRatio := float64(
		len(compressed),
	) / float64(len(plaintext))

	erasureOverhead := float64(
		totalFragmentBytes,
	) / float64(len(ciphertext))

	return &EncodedPayload{
		Manifest: Manifest{
			Version:          "LSB-STORAGE-V1",
			DataShards:       DataShards,
			ParityShards:     ParityShards,
			TotalShards:      TotalShards,
			OriginalSize:     len(plaintext),
			CompressedSize:   len(compressed),
			CiphertextSize:   len(ciphertext),
			NonceHex:         hex.EncodeToString(nonce),
			AAD:              string(aad),
			PlaintextSHA256:  hashHex(plaintext),
			CiphertextSHA256: hashHex(ciphertext),
			FragmentSHA256:   hashes,
			ErasureOverhead:  erasureOverhead,
			CompressionRatio: compressionRatio,
		},
		Fragments: fragments,
	}, nil
}

func Restore(
	manifest Manifest,
	fragments [][]byte,
	key []byte,
	aad []byte,
) ([]byte, error) {

	if err := validateKey(key); err != nil {
		return nil, err
	}

	if manifest.DataShards != DataShards ||
		manifest.ParityShards != ParityShards ||
		manifest.TotalShards != TotalShards {

		return nil, errors.New(
			"unsupported erasure-code configuration",
		)
	}

	if len(fragments) != TotalShards {
		return nil, fmt.Errorf(
			"expected %d fragment positions",
			TotalShards,
		)
	}

	if len(manifest.FragmentSHA256) != TotalShards {
		return nil, errors.New(
			"invalid fragment hash manifest",
		)
	}

	shards := make(
		[][]byte,
		TotalShards,
	)

	available := 0

	for i := 0; i < TotalShards; i++ {
		if len(fragments[i]) == 0 {
			continue
		}

		if hashHex(fragments[i]) !=
			manifest.FragmentSHA256[i] {

			// Treat corrupted fragment as erased.
			continue
		}

		shards[i] = append(
			[]byte(nil),
			fragments[i]...,
		)

		available++
	}

	if available < DataShards {
		return nil, fmt.Errorf(
			"insufficient valid fragments: have %d require %d",
			available,
			DataShards,
		)
	}

	encoder, err := reedsolomon.New(
		DataShards,
		ParityShards,
	)
	if err != nil {
		return nil, err
	}

	if err := encoder.Reconstruct(shards); err != nil {
		return nil, fmt.Errorf(
			"reconstruct shards: %w",
			err,
		)
	}

	ok, err := encoder.Verify(shards)
	if err != nil {
		return nil, fmt.Errorf(
			"verify reconstructed shards: %w",
			err,
		)
	}

	if !ok {
		return nil, errors.New(
			"reconstructed Reed-Solomon set is invalid",
		)
	}

	ciphertext := make(
		[]byte,
		0,
		manifest.CiphertextSize,
	)

	for i := 0; i < DataShards; i++ {
		ciphertext = append(
			ciphertext,
			shards[i]...,
		)
	}

	if len(ciphertext) <
		manifest.CiphertextSize {

		return nil, errors.New(
			"reconstructed ciphertext is too short",
		)
	}

	ciphertext = ciphertext[:manifest.CiphertextSize]

	if hashHex(ciphertext) !=
		manifest.CiphertextSHA256 {

		return nil, errors.New(
			"ciphertext SHA-256 mismatch",
		)
	}

	nonce, err := hex.DecodeString(
		manifest.NonceHex,
	)
	if err != nil {
		return nil, errors.New(
			"invalid manifest nonce",
		)
	}

	compressed, err := decryptAESGCM(
		key,
		nonce,
		ciphertext,
		aad,
	)
	if err != nil {
		return nil, err
	}

	plaintext, err := decompressPayload(
		compressed,
	)
	if err != nil {
		return nil, err
	}

	if hashHex(plaintext) !=
		manifest.PlaintextSHA256 {

		return nil, errors.New(
			"plaintext SHA-256 mismatch",
		)
	}

	return plaintext, nil
}
