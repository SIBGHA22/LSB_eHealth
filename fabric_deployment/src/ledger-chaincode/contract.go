package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

var (
	hashPattern = regexp.MustCompile(`^[a-fA-F0-9]{64}$`)
	txIDPattern = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)
)

type SmartContract struct {
	contractapi.Contract
}

type MedicalTransaction struct {
	TxID                     string `json:"tx_id"`
	Timestamp                string `json:"timestamp"`
	DeviceIDHash             string `json:"device_id_hash"`
	ShardID                  string `json:"shard_id"`
	PayloadCID               string `json:"payload_cid"`
	PayloadSHA256            string `json:"payload_sha256"`
	KeyHash                  string `json:"key_hash"`
	ConsensusCertificateHash string `json:"consensus_certificate_hash"`
	CommitteeEpoch           uint64 `json:"committee_epoch"`
	ModelVersion             string `json:"model_version"`
	EventType                string `json:"event_type"`
}

type LedgerStats struct {
	ChannelID        string `json:"channel_id"`
	TransactionCount int    `json:"transaction_count"`
}

func validateHash(name, value string) error {
	if !hashPattern.MatchString(value) {
		return fmt.Errorf("%s must be exactly 64 hexadecimal characters", name)
	}
	return nil
}

func validateShard(channelID, shardID string) error {
	if channelID == "lsb-registry" {
		if shardID != "registry" {
			return fmt.Errorf(
				"registry channel requires shard_id=registry, got %s",
				shardID,
			)
		}
		return nil
	}

	if !strings.HasPrefix(channelID, "lsb-shard-") {
		return fmt.Errorf("unsupported LSB ledger channel: %s", channelID)
	}

	expected := strings.TrimPrefix(channelID, "lsb-shard-")

	switch expected {
	case "0", "1", "2", "3":
	default:
		return fmt.Errorf("unsupported shard channel: %s", channelID)
	}

	if shardID != expected {
		return fmt.Errorf(
			"transaction shard_id=%s does not match channel %s",
			shardID,
			channelID,
		)
	}

	return nil
}

func (s *SmartContract) TransactionExists(
	ctx contractapi.TransactionContextInterface,
	txID string,
) (bool, error) {
	data, err := ctx.GetStub().GetState("tx:" + txID)
	if err != nil {
		return false, fmt.Errorf("failed reading transaction state: %w", err)
	}

	return data != nil, nil
}

func (s *SmartContract) CreateTransaction(
	ctx contractapi.TransactionContextInterface,
	txID string,
	timestamp string,
	deviceIDHash string,
	shardID string,
	payloadCID string,
	payloadSHA256 string,
	keyHash string,
	consensusCertificateHash string,
	committeeEpoch string,
	modelVersion string,
	eventType string,
) error {

	if !txIDPattern.MatchString(txID) {
		return fmt.Errorf("invalid tx_id")
	}

	if _, err := time.Parse(time.RFC3339Nano, timestamp); err != nil {
		return fmt.Errorf("timestamp must be RFC3339/RFC3339Nano: %w", err)
	}

	if err := validateHash("device_id_hash", deviceIDHash); err != nil {
		return err
	}

	if err := validateHash("payload_sha256", payloadSHA256); err != nil {
		return err
	}

	if err := validateHash("key_hash", keyHash); err != nil {
		return err
	}

	if err := validateHash(
		"consensus_certificate_hash",
		consensusCertificateHash,
	); err != nil {
		return err
	}

	if len(payloadCID) < 10 || len(payloadCID) > 256 {
		return fmt.Errorf("payload_cid length is invalid")
	}

	if strings.TrimSpace(modelVersion) == "" {
		return fmt.Errorf("model_version is required")
	}

	if strings.TrimSpace(eventType) == "" {
		return fmt.Errorf("event_type is required")
	}

	epoch, err := strconv.ParseUint(committeeEpoch, 10, 64)
	if err != nil {
		return fmt.Errorf("committee_epoch must be an unsigned integer")
	}

	channelID := ctx.GetStub().GetChannelID()

	if err := validateShard(channelID, shardID); err != nil {
		return err
	}

	exists, err := s.TransactionExists(ctx, txID)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("transaction %s already exists", txID)
	}

	tx := MedicalTransaction{
		TxID:                     txID,
		Timestamp:                timestamp,
		DeviceIDHash:             strings.ToLower(deviceIDHash),
		ShardID:                  shardID,
		PayloadCID:               payloadCID,
		PayloadSHA256:            strings.ToLower(payloadSHA256),
		KeyHash:                  strings.ToLower(keyHash),
		ConsensusCertificateHash: strings.ToLower(consensusCertificateHash),
		CommitteeEpoch:           epoch,
		ModelVersion:             modelVersion,
		EventType:                eventType,
	}

	payload, err := json.Marshal(tx)
	if err != nil {
		return fmt.Errorf("failed encoding transaction: %w", err)
	}

	if err := ctx.GetStub().PutState("tx:"+txID, payload); err != nil {
		return fmt.Errorf("failed storing transaction: %w", err)
	}

	return nil
}

func (s *SmartContract) ReadTransaction(
	ctx contractapi.TransactionContextInterface,
	txID string,
) (*MedicalTransaction, error) {

	payload, err := ctx.GetStub().GetState("tx:" + txID)
	if err != nil {
		return nil, fmt.Errorf("failed reading transaction: %w", err)
	}

	if payload == nil {
		return nil, fmt.Errorf("transaction %s does not exist", txID)
	}

	var tx MedicalTransaction

	if err := json.Unmarshal(payload, &tx); err != nil {
		return nil, fmt.Errorf("failed decoding transaction: %w", err)
	}

	return &tx, nil
}

func (s *SmartContract) GetAllTransactions(
	ctx contractapi.TransactionContextInterface,
) ([]*MedicalTransaction, error) {

	resultsIterator, err := ctx.GetStub().GetStateByRange("tx:", "tx;")
	if err != nil {
		return nil, fmt.Errorf("failed opening transaction iterator: %w", err)
	}
	defer resultsIterator.Close()

	var transactions []*MedicalTransaction

	for resultsIterator.HasNext() {
		item, err := resultsIterator.Next()
		if err != nil {
			return nil, fmt.Errorf("failed reading iterator: %w", err)
		}

		var tx MedicalTransaction
		if err := json.Unmarshal(item.Value, &tx); err != nil {
			return nil, fmt.Errorf("failed decoding ledger transaction: %w", err)
		}

		transactions = append(transactions, &tx)
	}

	return transactions, nil
}

func (s *SmartContract) GetLedgerStats(
	ctx contractapi.TransactionContextInterface,
) (*LedgerStats, error) {

	resultsIterator, err := ctx.GetStub().GetStateByRange("tx:", "tx;")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()

	count := 0

	for resultsIterator.HasNext() {
		if _, err := resultsIterator.Next(); err != nil {
			return nil, err
		}
		count++
	}

	return &LedgerStats{
		ChannelID:        ctx.GetStub().GetChannelID(),
		TransactionCount: count,
	}, nil
}
