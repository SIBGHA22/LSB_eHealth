// Package proto holds what nodes and the benchmark share: transactions, messages,
// deterministic keys, signatures and the two committee-selection rules.
package proto

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// ---- LSB eHealth parameters (copied from the paper's testbed config.go) ----
const (
	LSBCommitteeSize        = 7
	LSBByzantineTolerance   = 2 // f
	LSBCertificateThreshold = 5 // 2f+1
	ReselectionInterval     = 100
	BatteryPreference       = 0.60
	VRFWeight               = 0.72
	BatteryWeight           = 0.18
	TrustWeight             = 0.10
	HighBatteryBonus        = 0.02
)

type Tx struct {
	ID        string `json:"id"`
	Seq       uint64 `json:"seq"`
	Payload   []byte `json:"payload"`
	Digest    string `json:"digest"` // hex sha256 the client claims for Payload
	Mode      string `json:"mode"`   // "lsb" or "dbh"
	Committee []int  `json:"committee"`
	Leader    int    `json:"leader"`
}

// Msg is one protocol message between nodes.
// Kinds: lsb: PREPREPARE, PREPARE, COMMIT ; dbh: VOTEREQ (reply carries the vote).
type Msg struct {
	Kind   string `json:"kind"`
	From   int    `json:"from"`
	TxID   string `json:"tx_id"`
	Tx     *Tx    `json:"tx,omitempty"`
	Accept bool   `json:"accept"`
	Sig    []byte `json:"sig"`
}

type SubmitResult struct {
	TxID       string  `json:"tx_id"`
	Committed  bool    `json:"committed"`
	Rejected   bool    `json:"rejected"`
	TimedOut   bool    `json:"timed_out"`
	Votes      int     `json:"votes"`
	LeaderMs   float64 `json:"leader_ms"`
	BlockIndex int     `json:"block_index"`
}

type Metrics struct {
	Node        int     `json:"node"`
	MsgsSent    int64   `json:"msgs_sent"`
	MsgsRecv    int64   `json:"msgs_recv"`
	BytesSent   int64   `json:"bytes_sent"`
	SigsMade    int64   `json:"sigs_made"`
	SigsChecked int64   `json:"sigs_checked"`
	CPUUserMs   float64 `json:"cpu_user_ms"`
	CPUSysMs    float64 `json:"cpu_sys_ms"`
	MaxRSSKB    int64   `json:"max_rss_kb"`
	Blocks      int     `json:"blocks"`
	Fault       string  `json:"fault"`
}

// Keys are derived deterministically so every party knows every public key
// (stands in for the PKI / PUF-derived keys of the real system).
func KeyFor(node int) ed25519.PrivateKey {
	seed := sha256.Sum256([]byte(fmt.Sprintf("lsb-dbh-testbed-node-%d", node)))
	return ed25519.NewKeyFromSeed(seed[:])
}

func PubFor(node int) ed25519.PublicKey { return KeyFor(node).Public().(ed25519.PublicKey) }

func (m *Msg) SigningBytes() []byte {
	c := *m
	c.Sig = nil
	c.Tx = nil // the tx body is bound through TxID + digest checks
	b, _ := json.Marshal(c)
	return b
}

func Digest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

// ---- committee selection ----

// VRFOutput: an ed25519 signature over the round is unique per key and verifiable with the
// public key, so hashing it gives a verifiable pseudo-random value in [0,1).
func VRFOutput(node int, round uint64) (float64, []byte) {
	msg := make([]byte, 8)
	binary.BigEndian.PutUint64(msg, round)
	proof := ed25519.Sign(KeyFor(node), msg)
	h := sha256.Sum256(proof)
	return float64(binary.BigEndian.Uint64(h[:8])) / math.Pow(2, 64), proof
}

// LSBCommittee: top-k nodes by 0.72*VRF + 0.18*battery + 0.10*trust + 0.02*[battery>=0.6] (paper Eq. 6).
func LSBCommittee(n int, round uint64, battery, trust []float64) []int {
	type sc struct {
		id int
		s  float64
	}
	all := make([]sc, n)
	for i := 0; i < n; i++ {
		v, _ := VRFOutput(i, round)
		s := VRFWeight*v + BatteryWeight*battery[i] + TrustWeight*trust[i]
		if battery[i] >= BatteryPreference {
			s += HighBatteryBonus
		}
		all[i] = sc{i, s}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	k := LSBCommitteeSize
	if k > n {
		k = n
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].id // out[0] = highest score = leader
	}
	return out
}

// DBHValidatorCount: DBH-Chain PoV uses about log2(N) validators (min 4), as in the reconstruction.
func DBHValidatorCount(n int) int {
	v := int(math.Ceil(math.Log2(math.Max(float64(n), 2))))
	if v < 4 {
		v = 4
	}
	if v > n {
		v = n
	}
	return v
}

// DBHCommittee: v validators drawn uniformly by the same VRF output (no energy or trust terms).
func DBHCommittee(n int, round uint64) []int {
	type sc struct {
		id int
		s  float64
	}
	all := make([]sc, n)
	for i := 0; i < n; i++ {
		v, _ := VRFOutput(i, round^0x9e3779b97f4a7c15)
		all[i] = sc{i, v}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	v := DBHValidatorCount(n)
	out := make([]int, v)
	for i := 0; i < v; i++ {
		out[i] = all[i].id
	}
	return out
}

func DBHMajority(v int) int { return v/2 + 1 }
