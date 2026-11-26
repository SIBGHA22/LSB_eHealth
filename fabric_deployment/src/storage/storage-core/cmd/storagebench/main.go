package main

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"time"

	storage "lsb-ehealth/storage-core"
)

type Result struct {
	PayloadBytes       int     `json:"payload_bytes"`
	EncodeMs           float64 `json:"encode_ms"`
	RestoreMs          float64 `json:"restore_ms"`
	OriginalSize       int     `json:"original_size"`
	CompressedSize     int     `json:"compressed_size"`
	CiphertextSize     int     `json:"ciphertext_size"`
	TotalFragments     int     `json:"total_fragments"`
	RecoveredFragments int     `json:"recovered_fragments"`
	RecoverySuccessful bool    `json:"recovery_successful"`
	CompressionRatio   float64 `json:"compression_ratio"`
	ErasureOverhead    float64 `json:"erasure_overhead"`
}

func main() {

	payload := make([]byte, 1024*1024)

	for i := range payload {
		payload[i] = byte(i % 251)
	}

	key := make([]byte, 32)
	_, _ = rand.Read(key)

	aad := []byte("lsb-ehealth-storage-benchmark")

	start := time.Now()

	encoded, err := storage.Encode(
		payload,
		key,
		aad,
	)

	if err != nil {
		panic(err)
	}

	encodeMs := float64(time.Since(start).Microseconds()) / 1000

	// Prepare all fragment positions required by Restore()
	fragments := make([][]byte, storage.TotalShards)

	for i, fragment := range encoded.Fragments {

		if i < len(fragments) {
			fragments[i] = fragment.Data
		}
	}

	start = time.Now()

	restored, err := storage.Restore(
		encoded.Manifest,
		fragments,
		key,
		aad,
	)

	if err != nil {
		panic(err)
	}

	restoreMs := float64(time.Since(start).Microseconds()) / 1000

	result := Result{
		PayloadBytes:       len(payload),
		EncodeMs:           encodeMs,
		RestoreMs:          restoreMs,
		OriginalSize:       encoded.Manifest.OriginalSize,
		CompressedSize:     encoded.Manifest.CompressedSize,
		CiphertextSize:     encoded.Manifest.CiphertextSize,
		TotalFragments:     len(encoded.Fragments),
		RecoveredFragments: len(fragments),
		RecoverySuccessful: string(restored) == string(payload),
		CompressionRatio:   encoded.Manifest.CompressionRatio,
		ErasureOverhead:    encoded.Manifest.ErasureOverhead,
	}

	out, _ := json.MarshalIndent(
		result,
		"",
		"  ",
	)

	fmt.Println(string(out))

	os.Exit(0)
}
