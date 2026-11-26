// bench: drives one consensus mode on a running cluster with a fixed, seeded workload and
// writes one JSON result. Both modes get the identical workload (same payloads, same
// tampered transactions, same order) when run with the same -seed.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"lsbdbh/internal/proto"
)

type txRecord struct {
	Seq       int     `json:"seq"`
	Valid     bool    `json:"valid"`
	Committed bool    `json:"committed"`
	Rejected  bool    `json:"rejected"`
	TimedOut  bool    `json:"timed_out"`
	Attempts  int     `json:"attempts"`
	LatencyMs float64 `json:"latency_ms"`
}

func main() {
	mode := flag.String("mode", "lsb", "lsb, lsb-pbft or dbh")
	nodes := flag.Int("nodes", 20, "number of validator nodes in the cluster")
	urlFmt := flag.String("url-fmt", "http://127.0.0.1:%d", "node URL format; %d = base-port + node id")
	basePort := flag.Int("base-port", 18000, "host port of node 0")
	txs := flag.Int("txs", 1000, "measured transactions")
	warm := flag.Int("warmup", 50, "warm-up transactions (not measured)")
	conc := flag.Int("concurrency", 1, "concurrent clients (1 = latency test)")
	tamper := flag.Float64("tamper", 0.10, "fraction of transactions with a wrong digest")
	payload := flag.Int("payload", 1024, "payload bytes")
	seed := flag.Int64("seed", 42, "workload seed")
	scenario := flag.String("scenario", "baseline", "label written into the result")
	out := flag.String("out", "", "output JSON file")
	flag.Parse()

	url := func(i int) string { return fmt.Sprintf(*urlFmt, *basePort+i) }
	cli := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: 256}}

	// identical node attributes for both modes
	rng := rand.New(rand.NewSource(*seed))
	battery := make([]float64, *nodes)
	trust := make([]float64, *nodes)
	for i := range battery {
		battery[i] = 0.2 + 0.8*rng.Float64()
		trust[i] = 0.9
	}

	// identical workload for both modes
	total := *warm + *txs
	work := make([]proto.Tx, total)
	valid := make([]bool, total)
	for i := 0; i < total; i++ {
		p := make([]byte, *payload)
		rng.Read(p)
		d := proto.Digest(p)
		valid[i] = true
		if rng.Float64() < *tamper {
			p[0] ^= 0xFF // payload altered after the digest was taken
			valid[i] = false
		}
		work[i] = proto.Tx{ID: fmt.Sprintf("%s-%s-%d", *scenario, *mode, i), Seq: uint64(i), Payload: p, Digest: d, Mode: *mode}
	}

	// committee per reselection round (timed separately: off the per-transaction path)
	committees := map[uint64][]int{}
	var selTotal time.Duration
	committeeFor := func(seq uint64) []int {
		r := seq / proto.ReselectionInterval
		if c, ok := committees[r]; ok {
			return c
		}
		t0 := time.Now()
		var c []int
		if *mode == "lsb" || *mode == "lsb-pbft" {
			c = proto.LSBCommittee(*nodes, r, battery, trust)
		} else {
			c = proto.DBHCommittee(*nodes, r)
		}
		selTotal += time.Since(t0)
		committees[r] = c
		return c
	}
	for i := range work {
		committeeFor(work[i].Seq)
	}

	submit := func(tx proto.Tx) txRecord {
		rec := txRecord{Seq: int(tx.Seq)}
		t0 := time.Now()
		order := committeeFor(tx.Seq)
		for a := 0; a < len(order) && a < 3; a++ { // simple view change: next committee member leads
			tx.Committee = order
			tx.Leader = order[a]
			tx.ID = fmt.Sprintf("%s-%s-%d-a%d", *scenario, *mode, tx.Seq, a)
			rec.Attempts = a + 1
			b, _ := json.Marshal(tx)
			resp, err := cli.Post(url(order[a])+"/submit", "application/json", bytes.NewReader(b))
			if err != nil {
				continue
			}
			var res proto.SubmitResult
			ok := resp.StatusCode == 200 && json.NewDecoder(resp.Body).Decode(&res) == nil
			resp.Body.Close()
			if !ok || res.TimedOut {
				continue
			}
			rec.Committed, rec.Rejected = res.Committed, res.Rejected
			break
		}
		rec.TimedOut = !rec.Committed && !rec.Rejected
		rec.LatencyMs = float64(time.Since(t0).Microseconds()) / 1000
		return rec
	}

	snapshot := func() (msgs, bytesOut, sigs int64, cpu float64, rss int64) {
		for i := 0; i < *nodes; i++ {
			resp, err := cli.Get(url(i) + "/metrics")
			if err != nil {
				log.Fatalf("node %d unreachable: %v", i, err)
			}
			var m proto.Metrics
			json.NewDecoder(resp.Body).Decode(&m)
			resp.Body.Close()
			msgs += m.MsgsSent
			bytesOut += m.BytesSent
			sigs += m.SigsMade + m.SigsChecked
			cpu += m.CPUUserMs + m.CPUSysMs
			if m.MaxRSSKB > rss {
				rss = m.MaxRSSKB
			}
		}
		return
	}

	run := func(batch []proto.Tx) ([]txRecord, time.Duration) {
		recs := make([]txRecord, len(batch))
		jobs := make(chan int)
		var wg sync.WaitGroup
		t0 := time.Now()
		for w := 0; w < *conc; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range jobs {
					recs[i] = submit(batch[i])
				}
			}()
		}
		for i := range batch {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		return recs, time.Since(t0)
	}

	run(work[:*warm])
	m0, b0, s0, c0, _ := snapshot()
	recs, wall := run(work[*warm:])
	m1, b1, s1, c1, rss := snapshot()
	for i := range recs {
		recs[i].Valid = valid[*warm+i]
	}

	var lat []float64
	committed, unsafe, falseRej, timeouts, retries := 0, 0, 0, 0, 0
	for _, r := range recs {
		lat = append(lat, r.LatencyMs)
		if r.Committed {
			committed++
			if !r.Valid {
				unsafe++
			}
		}
		if r.Rejected && r.Valid {
			falseRej++
		}
		if r.TimedOut {
			timeouts++
		}
		if r.Attempts > 1 {
			retries++
		}
	}
	sort.Float64s(lat)
	pct := func(p float64) float64 { return lat[int(math.Min(float64(len(lat)-1), p*float64(len(lat))))] }
	mean := 0.0
	for _, v := range lat {
		mean += v
	}
	mean /= float64(len(lat))
	n := float64(len(recs))
	c := committeeFor(uint64(*warm))
	res := map[string]any{
		"mode": *mode, "scenario": *scenario, "nodes": *nodes, "committee_size": len(c),
		"concurrency": *conc, "txs": len(recs), "seed": *seed, "tamper_fraction": *tamper, "payload_bytes": *payload,
		"latency_ms_mean": mean, "latency_ms_median": pct(0.5), "latency_ms_p95": pct(0.95), "latency_ms_p99": pct(0.99),
		"throughput_tx_s":          float64(len(recs)) / wall.Seconds(),
		"committed_tx_s":           float64(committed) / wall.Seconds(),
		"messages_per_tx":          float64(m1-m0) / n,
		"kbytes_per_tx":            float64(b1-b0) / n / 1024,
		"signature_ops_per_tx":     float64(s1-s0) / n,
		"cpu_ms_per_tx":            (c1 - c0) / n,
		"max_node_rss_mb":          float64(rss) / 1024,
		"committed":                committed,
		"unsafe_commits":           unsafe,
		"false_rejects":            falseRej,
		"timeouts":                 timeouts,
		"view_changes":             retries,
		"selection_ms_per_round":   float64(selTotal.Microseconds()) / 1000 / float64(len(committees)),
		"wall_seconds":             wall.Seconds(),
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(b))
	if *out != "" {
		os.WriteFile(*out, b, 0o644)
	}
}
