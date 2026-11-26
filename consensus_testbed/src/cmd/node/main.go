// node: one validator. The same binary runs both consensus protocols:
//
//	lsb      : LSB eHealth as specified in the paper (Sec. 7.4, Algorithm 3): the proposer of a
//	           k=7 committee chosen by the energy-aware VRF score (Eq. 6) collects signed Commit
//	           votes and forms a quorum certificate from Q = 2f+1 = 5 valid votes.
//	lsb-pbft : the same committee running full three-phase PBFT (extra variant).
//	dbh : DBH-Chain Proof-of-Validation = one voting round: the leader asks ~log2(N)
//	      validators to validate and sign; simple majority decides.
//
// Both protocols do the same per-message work: JSON over HTTP, ed25519 sign + verify,
// SHA-256 payload check. Only the protocol (who talks to whom, how many rounds) differs.
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"lsbdbh/internal/proto"
)

var (
	self      int
	nodeCount int
	peerFmt   string // e.g. http://node-%d:8080
	fault     string // none | crash | byzantine
	netDelay  time.Duration
	timeout   = 1500 * time.Millisecond

	priv ed25519.PrivateKey
	pubs []ed25519.PublicKey

	client = &http.Client{
		Timeout: 3 * time.Second,
		Transport: &http.Transport{MaxIdleConns: 4096, MaxIdleConnsPerHost: 512,
			IdleConnTimeout: 90 * time.Second},
	}

	msgsSent, msgsRecv, bytesSent, sigsMade, sigsChecked atomic.Int64

	ledgerMu sync.Mutex
	ledger   []string // block hashes
	lastHash = "genesis"
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func sign(m *proto.Msg) {
	m.From = self
	m.Sig = ed25519.Sign(priv, m.SigningBytes())
	sigsMade.Add(1)
}

func verify(m *proto.Msg) bool {
	if m.From < 0 || m.From >= nodeCount {
		return false
	}
	sigsChecked.Add(1)
	return ed25519.Verify(pubs[m.From], m.SigningBytes(), m.Sig)
}

// validate: the honest verdict on a transaction; a byzantine node inverts it.
func validate(tx *proto.Tx) bool {
	ok := proto.Digest(tx.Payload) == tx.Digest
	if fault == "byzantine" {
		return !ok
	}
	return ok
}

func appendBlock(txid, digest string) int {
	ledgerMu.Lock()
	defer ledgerMu.Unlock()
	h := sha256.Sum256([]byte(lastHash + "|" + txid + "|" + digest))
	lastHash = hex.EncodeToString(h[:])
	ledger = append(ledger, lastHash)
	return len(ledger) - 1
}

func post(to int, path string, body any) (*http.Response, error) {
	b, _ := json.Marshal(body)
	if netDelay > 0 {
		time.Sleep(netDelay)
	}
	msgsSent.Add(1)
	bytesSent.Add(int64(len(b)))
	return client.Post(fmt.Sprintf(peerFmt, to)+path, "application/json", bytes.NewReader(b))
}

func sendAsync(to int, m proto.Msg) {
	go func() {
		resp, err := post(to, "/msg", m)
		if err == nil {
			resp.Body.Close()
		}
	}()
}

// ======================= LSB: PBFT state machine =======================

type pbft struct {
	mu          sync.Mutex
	tx          *proto.Tx
	leader      int
	prePrepared bool
	myVote      bool
	prepares    map[int]bool // node -> accept
	commits     map[int]bool
	sentCommit  bool
	committed   bool
	decided     bool
	done        chan proto.SubmitResult // only on the leader
	start       time.Time
}

var (
	pbftMu sync.Mutex
	rounds = map[string]*pbft{}
)

func getRound(key string) *pbft {
	pbftMu.Lock()
	defer pbftMu.Unlock()
	r, ok := rounds[key]
	if !ok {
		r = &pbft{prepares: map[int]bool{}, commits: map[int]bool{}}
		rounds[key] = r
	}
	return r
}

func key(txid string, leader int) string { return txid + "|" + strconv.Itoa(leader) }

func inCommittee(tx *proto.Tx, id int) bool {
	for _, c := range tx.Committee {
		if c == id {
			return true
		}
	}
	return false
}

func broadcast(tx *proto.Tx, m proto.Msg) {
	for _, c := range tx.Committee {
		if c != self {
			sendAsync(c, m)
		}
	}
}

// progress re-checks quorums after any event. Caller holds r.mu.
func (r *pbft) progress() {
	if r.tx == nil {
		return
	}
	q := proto.LSBCertificateThreshold
	if len(r.tx.Committee) < proto.LSBCommitteeSize { // small networks
		f := (len(r.tx.Committee) - 1) / 3
		q = 2*f + 1
	}
	acc, rej := 0, 0
	for _, v := range r.prepares {
		if v {
			acc++
		} else {
			rej++
		}
	}
	if r.prePrepared && !r.sentCommit && acc >= q {
		r.sentCommit = true
		r.commits[self] = true
		m := proto.Msg{Kind: "COMMIT", TxID: r.tx.ID, Accept: true}
		m.Tx = &proto.Tx{ID: r.tx.ID, Committee: r.tx.Committee, Leader: r.tx.Leader} // routing only
		sign(&m)
		broadcast(r.tx, m)
	}
	if !r.committed && len(r.commits) >= q {
		r.committed = true
		idx := appendBlock(r.tx.ID, r.tx.Digest)
		if r.done != nil && !r.decided {
			r.decided = true
			r.done <- proto.SubmitResult{TxID: r.tx.ID, Committed: true, Votes: len(r.commits),
				LeaderMs: ms(time.Since(r.start)), BlockIndex: idx}
		}
	}
	// a commit quorum can no longer form -> reject early
	if r.done != nil && !r.decided && rej > len(r.tx.Committee)-q {
		r.decided = true
		r.done <- proto.SubmitResult{TxID: r.tx.ID, Rejected: true, Votes: rej,
			LeaderMs: ms(time.Since(r.start))}
	}
}

func onPrePrepare(m *proto.Msg) {
	tx := m.Tx
	if tx == nil || !inCommittee(tx, self) || !inCommittee(tx, m.From) || tx.Leader != m.From {
		return
	}
	r := getRound(key(tx.ID, m.From))
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.prePrepared {
		return
	}
	r.tx, r.leader, r.prePrepared = tx, m.From, true
	r.myVote = validate(tx)
	r.prepares[self] = r.myVote
	p := proto.Msg{Kind: "PREPARE", TxID: tx.ID, Accept: r.myVote}
	p.Tx = &proto.Tx{ID: tx.ID, Committee: tx.Committee, Leader: tx.Leader} // routing only
	sign(&p)
	broadcast(tx, p)
	r.progress()
}

func onPhase(m *proto.Msg) {
	if m.Tx == nil {
		return
	}
	r := getRound(key(m.TxID, m.Tx.Leader))
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.tx == nil { // arrived before our pre-prepare: keep routing info
		r.tx = &proto.Tx{ID: m.TxID, Committee: m.Tx.Committee, Leader: m.Tx.Leader}
	}
	if !inCommittee(r.tx, m.From) {
		return
	}
	if m.Kind == "PREPARE" {
		r.prepares[m.From] = m.Accept
	} else if m.Kind == "COMMIT" && m.Accept {
		r.commits[m.From] = true
	}
	r.progress()
}

func lsbLead(tx *proto.Tx) proto.SubmitResult {
	r := getRound(key(tx.ID, self))
	r.mu.Lock()
	r.done = make(chan proto.SubmitResult, 1)
	r.start = time.Now()
	done := r.done
	r.mu.Unlock()

	pp := proto.Msg{Kind: "PREPREPARE", TxID: tx.ID, Tx: tx}
	sign(&pp)
	broadcast(tx, pp)
	onPrePrepare(&pp) // the leader is also a replica

	select {
	case res := <-done:
		return res
	case <-time.After(timeout):
		r.mu.Lock()
		r.decided = true
		r.mu.Unlock()
		return proto.SubmitResult{TxID: tx.ID, TimedOut: true, LeaderMs: ms(time.Since(r.start))}
	}
}

// ======================= DBH: Proof-of-Validation =======================

// voteLead runs one signed-vote round: the proposer signs the proposal, every committee member
// verifies it and returns a signed vote, and the proposer forms a certificate from `quorum`
// accepting votes. Used by
//   dbh : DBH-Chain PoV, quorum = simple majority of ~log2(N) validators
//   lsb : LSB eHealth Algorithm 3 (paper Sec. 7.4), quorum Q = 2f+1 = 5 of the k = 7 committee
func voteLead(tx *proto.Tx, quorum int) proto.SubmitResult {
	start := time.Now()
	maj := quorum
	rejectLimit := len(tx.Committee) - quorum + 1 // this many rejects make the quorum impossible
	type vote struct{ ok, accept bool }
	ch := make(chan vote, len(tx.Committee))
	for _, c := range tx.Committee {
		if c == self {
			ch <- vote{true, validate(tx)}
			continue
		}
		go func(c int) {
			req := proto.Msg{Kind: "VOTEREQ", TxID: tx.ID, Tx: tx}
			sign(&req)
			resp, err := post(c, "/vote", req)
			if err != nil {
				ch <- vote{}
				return
			}
			defer resp.Body.Close()
			var v proto.Msg
			if json.NewDecoder(resp.Body).Decode(&v) != nil || v.From != c || v.TxID != tx.ID || !verify(&v) {
				ch <- vote{}
				return
			}
			msgsRecv.Add(1)
			ch <- vote{true, v.Accept}
		}(c)
	}
	acc, rej, got := 0, 0, 0
	deadline := time.After(timeout)
	for got < len(tx.Committee) {
		select {
		case v := <-ch:
			got++
			if v.ok && v.accept {
				acc++
			} else if v.ok {
				rej++
			}
			if acc >= maj {
				idx := appendBlock(tx.ID, tx.Digest)
				return proto.SubmitResult{TxID: tx.ID, Committed: true, Votes: acc, LeaderMs: ms(time.Since(start)), BlockIndex: idx}
			}
			if rej >= rejectLimit {
				return proto.SubmitResult{TxID: tx.ID, Rejected: true, Votes: rej, LeaderMs: ms(time.Since(start))}
			}
		case <-deadline:
			return proto.SubmitResult{TxID: tx.ID, TimedOut: true, LeaderMs: ms(time.Since(start))}
		}
	}
	return proto.SubmitResult{TxID: tx.ID, TimedOut: true, LeaderMs: ms(time.Since(start))}
}

// ======================= HTTP handlers =======================

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func handleSubmit(w http.ResponseWriter, req *http.Request) {
	var tx proto.Tx
	if err := json.NewDecoder(req.Body).Decode(&tx); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if fault == "crash" {
		time.Sleep(timeout) // a crashed leader never answers in time
		http.Error(w, "crashed", 503)
		return
	}
	tx.Leader = self
	var res proto.SubmitResult
	switch tx.Mode {
	case "lsb": // paper-faithful: Algorithm 3 signed-vote certificate
		f := (len(tx.Committee) - 1) / 3
		res = voteLead(&tx, 2*f+1)
	case "lsb-pbft": // full three-phase PBFT over the same committee (extra variant)
		res = lsbLead(&tx)
	case "dbh":
		res = voteLead(&tx, proto.DBHMajority(len(tx.Committee)))
	default:
		http.Error(w, "mode must be lsb, lsb-pbft or dbh", 400)
		return
	}
	json.NewEncoder(w).Encode(res)
}

func handleMsg(w http.ResponseWriter, req *http.Request) {
	var m proto.Msg
	if json.NewDecoder(req.Body).Decode(&m) != nil {
		w.WriteHeader(400)
		return
	}
	w.WriteHeader(204)
	if fault == "crash" {
		return
	}
	msgsRecv.Add(1)
	if !verify(&m) {
		return
	}
	switch m.Kind {
	case "PREPREPARE":
		onPrePrepare(&m)
	case "PREPARE", "COMMIT":
		onPhase(&m)
	}
}

func handleVote(w http.ResponseWriter, req *http.Request) {
	var m proto.Msg
	if json.NewDecoder(req.Body).Decode(&m) != nil || m.Tx == nil {
		w.WriteHeader(400)
		return
	}
	if fault == "crash" {
		time.Sleep(timeout)
		w.WriteHeader(503)
		return
	}
	msgsRecv.Add(1)
	if !verify(&m) || !inCommittee(m.Tx, self) || m.Tx.Leader != m.From {
		w.WriteHeader(403)
		return
	}
	v := proto.Msg{Kind: "VOTE", TxID: m.TxID, Accept: validate(m.Tx)}
	sign(&v)
	b, _ := json.Marshal(v)
	if netDelay > 0 {
		time.Sleep(netDelay)
	}
	msgsSent.Add(1)
	bytesSent.Add(int64(len(b)))
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func handleMetrics(w http.ResponseWriter, _ *http.Request) {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	ledgerMu.Lock()
	nb := len(ledger)
	ledgerMu.Unlock()
	json.NewEncoder(w).Encode(proto.Metrics{
		Node: self, MsgsSent: msgsSent.Load(), MsgsRecv: msgsRecv.Load(), BytesSent: bytesSent.Load(),
		SigsMade: sigsMade.Load(), SigsChecked: sigsChecked.Load(),
		CPUUserMs: float64(ru.Utime.Sec)*1000 + float64(ru.Utime.Usec)/1000,
		CPUSysMs:  float64(ru.Stime.Sec)*1000 + float64(ru.Stime.Usec)/1000,
		MaxRSSKB:  ru.Maxrss, Blocks: nb, Fault: fault,
	})
}

func handleReset(w http.ResponseWriter, _ *http.Request) {
	pbftMu.Lock()
	rounds = map[string]*pbft{}
	pbftMu.Unlock()
	w.WriteHeader(204)
}

func main() {
	var err error
	if self, err = strconv.Atoi(env("NODE_ID", "0")); err != nil {
		log.Fatal(err)
	}
	if nodeCount, err = strconv.Atoi(env("NODE_COUNT", "20")); err != nil {
		log.Fatal(err)
	}
	peerFmt = env("PEER_FMT", "http://node-%d:8080")
	fault = env("FAULT", "none")
	d, _ := strconv.Atoi(env("NET_DELAY_MS", "0"))
	netDelay = time.Duration(d) * time.Millisecond
	if t, _ := strconv.Atoi(env("TIMEOUT_MS", "1500")); t > 0 {
		timeout = time.Duration(t) * time.Millisecond
	}
	priv = proto.KeyFor(self)
	pubs = make([]ed25519.PublicKey, nodeCount)
	for i := range pubs {
		pubs[i] = proto.PubFor(i)
	}
	http.HandleFunc("/submit", handleSubmit)
	http.HandleFunc("/msg", handleMsg)
	http.HandleFunc("/vote", handleVote)
	http.HandleFunc("/metrics", handleMetrics)
	http.HandleFunc("/reset", handleReset)
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	addr := env("LISTEN", ":8080")
	log.Printf("node %d/%d fault=%s delay=%v listening %s", self, nodeCount, fault, netDelay, addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
