package main

import "testing"

func TestValidateHash(t *testing.T) {
	valid := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

	if err := validateHash("test_hash", valid); err != nil {
		t.Fatalf("valid SHA256 hash rejected: %v", err)
	}

	if err := validateHash("test_hash", "abc123"); err == nil {
		t.Fatal("invalid hash was accepted")
	}
}

func TestValidateShard(t *testing.T) {
	tests := []struct {
		channel string
		shard   string
		valid   bool
	}{
		{"lsb-registry", "registry", true},
		{"lsb-shard-0", "0", true},
		{"lsb-shard-1", "1", true},
		{"lsb-shard-2", "2", true},
		{"lsb-shard-3", "3", true},

		{"lsb-shard-0", "1", false},
		{"lsb-shard-2", "3", false},
		{"lsb-registry", "0", false},
		{"lsbtest", "0", false},
	}

	for _, tc := range tests {
		err := validateShard(tc.channel, tc.shard)

		if tc.valid && err != nil {
			t.Fatalf(
				"expected valid channel=%s shard=%s: %v",
				tc.channel,
				tc.shard,
				err,
			)
		}

		if !tc.valid && err == nil {
			t.Fatalf(
				"expected invalid channel=%s shard=%s",
				tc.channel,
				tc.shard,
			)
		}
	}
}
