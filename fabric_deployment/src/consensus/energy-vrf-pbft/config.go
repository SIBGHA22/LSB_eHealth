package consensus

const (
	CommitteeSize        = 7
	ByzantineTolerance   = 2
	CertificateThreshold = 5
	ShardCount           = 4

	// Committee/proposer state is refreshed every 100 committed blocks.
	ReselectionInterval uint64 = 100

	// Energy preference is intentionally soft, not an eligibility cutoff.
	BatteryPreferenceThreshold = 0.60

	// Selection score weights.
	VRFWeight     = 0.72
	BatteryWeight = 0.18
	TrustWeight   = 0.10

	// Small preference for validators with >= 60% battery.
	HighBatteryBonus = 0.02
)
