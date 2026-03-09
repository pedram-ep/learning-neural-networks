package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

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


func main() {
	trainFeatures, _, header, err := loadCSV("data/diabetes_train.csv", true)
	if err != nil {
		log.Fatal(err)
	}
	testFeatures, _, header, err := loadCSV("data/diabetes_test.csv", false)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Train column names:", header)
	fmt.Printf("Train features: %v x %v\n", trainFeatures.RawMatrix().Rows, trainFeatures.RawMatrix().Cols)

	fmt.Println("Test column names:", header)
	fmt.Printf("Test features: %v x %v\n", testFeatures.RawMatrix().Rows, testFeatures.RawMatrix().Cols)
}