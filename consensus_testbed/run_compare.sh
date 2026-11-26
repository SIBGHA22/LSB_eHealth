#!/usr/bin/env bash
# LSB eHealth vs DBH-Chain on a real Docker validator network (same machine, same workload).
# Usage:  ./run_compare.sh            full run (about 2-4 h)
#         QUICK=1 ./run_compare.sh    5-minute check
set -u
cd "$(dirname "$0")"
OUT=results; mkdir -p "$OUT/raw" "$OUT/logs"
IMG=lsbdbh-node:1
NET=lsbdbh-net

if [ "${QUICK:-0}" = "1" ]; then
  NODES_LIST="20"; DELAYS="0"; SEEDS="42"; TXS=200; TXS_TP=400; SCENARIOS="normal byzantine30"
else
  NODES_LIST="${NODES_LIST:-20 50}"; DELAYS="${DELAYS:-0 5}"; SEEDS="${SEEDS:-42 43 44 45 46}"
  TXS=${TXS:-1000}; TXS_TP=${TXS_TP:-2000}; SCENARIOS="${SCENARIOS:-normal byzantine30 crash10}"
fi
MODES="${MODES:-lsb dbh lsb-pbft}"
CPUS="${CPUS:-1.0}"           # CPU limit per validator container (same for both systems)

[ -x bin/node ] && [ -x bin/bench ] || ./build.sh || { echo "build failed"; exit 1; }
docker build -q -t $IMG . >/dev/null || { echo "docker build failed"; exit 1; }
docker network inspect $NET >/dev/null 2>&1 || docker network create $NET >/dev/null

stop_cluster() { docker ps -aq --filter "label=lsbdbh=1" | xargs -r docker rm -f >/dev/null 2>&1; }
trap stop_cluster EXIT

faulty_ids() { # $1=N $2=fraction $3=seed -> comma list, same nodes for every mode
  python3 -c "import random,sys;N,f,s=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]);print(','.join(map(str,sorted(random.Random(s).sample(range(N),round(N*f))))))" "$1" "$2" "$3"
}

start_cluster() { # $1=N $2=delay $3=fault_kind $4=faulty_list
  stop_cluster
  for i in $(seq 0 $(( $1 - 1 ))); do
    F=none; [[ ",$4," == *",$i,"* ]] && F=$3
    docker run -d --label lsbdbh=1 --name node-$i --hostname node-$i --network $NET --cpus "$CPUS" \
      -p 127.0.0.1:$((18000+i)):8080 \
      -e NODE_ID=$i -e NODE_COUNT=$1 -e PEER_FMT='http://node-%d:8080' -e FAULT=$F -e NET_DELAY_MS=$2 \
      $IMG >/dev/null || { echo "failed to start node-$i"; exit 1; }
  done
  for i in $(seq 0 $(( $1 - 1 ))); do   # wait until every node answers
    for t in $(seq 1 50); do curl -sf http://127.0.0.1:$((18000+i))/health >/dev/null && break; sleep 0.2; done
  done
}

for N in $NODES_LIST; do for D in $DELAYS; do for S in $SEEDS; do for SC in $SCENARIOS; do
  case $SC in
    normal)      KIND=none;      FL="";;
    byzantine30) KIND=byzantine; FL=$(faulty_ids $N 0.30 $S);;
    crash10)     KIND=crash;     FL=$(faulty_ids $N 0.10 $S);;
  esac
  echo "$(date +%T)  N=$N delay=${D}ms seed=$S $SC faulty=[$FL]"
  start_cluster $N $D $KIND "$FL"
  for M in $MODES; do
    TAG="N${N}_d${D}_${SC}_s${S}_${M}"
    ./bin/bench -mode $M -nodes $N -txs $TXS -warmup 50 -concurrency 1 -seed $S -scenario $SC \
       -out "$OUT/raw/${TAG}_c1.json" > "$OUT/logs/${TAG}_c1.log" 2>&1 || echo "   bench failed: $TAG c1"
    if [ "$SC" = "normal" ]; then
      ./bin/bench -mode $M -nodes $N -txs $TXS_TP -warmup 50 -concurrency 16 -seed $S -scenario $SC \
         -out "$OUT/raw/${TAG}_c16.json" > "$OUT/logs/${TAG}_c16.log" 2>&1 || echo "   bench failed: $TAG c16"
    fi
    docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' $(docker ps -q --filter label=lsbdbh=1) \
       > "$OUT/logs/${TAG}_docker_stats.csv" 2>/dev/null
  done
done; done; done; done
stop_cluster
python3 summarize.py "$OUT"
echo "$(date +%T)  ALL DONE -> $OUT/summary.md"
