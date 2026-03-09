package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

func loadCSV(path string, hasOutcome bool) (*mat.Dense, []float64, []string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	r:= csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}

	if len(records) == 0 {
		return nil, nil, nil, fmt.Errorf("empty CSV")
	}

	header := records[0]
	records = records[1:]

	nCols := len(records[0])
	outcomeIdx := nCols - 1
	featureCols := nCols
	if hasOutcome {
		featureCols = nCols - 1
	}

	rows := len(records)
	data := make([]float64, 0, rows*featureCols)
	outcomes := make([]float64, 0, rows)

	for _, record := range records {
		// parsing features
		for i := 0; i < featureCols; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("Parsing feature: %v", err)
			}
			data = append(data, val)
		}
		// parse outcome if present
		if hasOutcome {
			out, err := strconv.ParseFloat(record[outcomeIdx], 64)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("Parsing outcome: %v", err)
			}
			outcomes = append(outcomes, out)
		}
	}
	features := mat.NewDense(rows, featureCols, data)
	return features, outcomes, header, nil
}

func splitData(features *mat.Dense, outcomes []float64, testSize float64) (*mat.Dense, []float64, *mat.Dense, []float64) {
	r, c := features.Dims()
	nTest := int(float64(r) * testSize)
	nTrain := r - nTest

	// shuffle indices
	indices := rand.Perm(r)
	trainRows := indices[:nTrain]
	valRows := indices[nTrain:]

	// Build train matrix
	trainData := make([]float64, nTrain*c)
	trainOut := make([]float64, nTrain)
	for i, idx := range trainRows {
		for j := 0; j < c; j++ {
			trainData[i*c+j] = features.At(idx, j)
		}
		trainOut[i] = outcomes[idx]
	}

	// Build validation matrix
	valData := make([]float64, nTest*c)
	valOut := make([]float64, nTest)
	for i, idx := range valRows {
		for j := 0; j < c; j++ {
			valData[i*c+j] = features.At(idx, j)
		}
		valOut[i] = outcomes[idx]
	}

	trainMat := mat.NewDense(nTrain, c, trainData)
	valMat := mat.NewDense(nTest, c, valData)
	return trainMat, trainOut, valMat, valOut
}