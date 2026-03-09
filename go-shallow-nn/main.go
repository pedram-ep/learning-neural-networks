package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
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

func computeMeanStd(m *mat.Dense) ([]float64, []float64) {
	r, c := m.Dims()
	means := make([]float64, c)
	stds := make([]float64, c)

	for j := 0; j < c; j++ {
		col := mat.Col(nil, j, m)
		sum := 0.0
		for _, val := range col {
			sum += val
		}
		means[j] = sum / float64(r)
	}

	for j := 0; j < c; j++ {
		col := mat.Col(nil, j, m)
		var sumSq float64
		for _, v := range col {
			diff := v - means[j]
			sumSq += diff * diff
		}
		stds[j] = sqrt(sumSq / float64(r))
	}
	return means, stds
}

func sqrt(x float64) float64 { return math.Sqrt(x) }

func standardize(m *mat.Dense, means []float64, stds []float64) *mat.Dense {
	r, c := m.Dims()
	data := make([]float64, r*c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := m.At(i, j)
			data[i*c + j] = (val - means[j]) / stds[j]

		}
	}
	return mat.NewDense(r, c, data)
}

func addBias(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	newData := make([]float64, r*(c+1))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			newData[i*(c+1) + j] = m.At(i, j)
		}
		newData[i*(c+1) + c] = 1.0
	}

	return mat.NewDense(r, c+1, newData)
}

func main() {
	// Loading datasets from CSV files
	trainFeatures, _, header, err := loadCSV("data/diabetes_train.csv", true)
	if err != nil {
		log.Fatal(err)
	}
	testFeatures, _, header, err := loadCSV("data/diabetes_test.csv", false)
	if err != nil {
		log.Fatal(err)
	}

	// Computing mean and std for all columns, and then standardizing both train and test
	means, stds := computeMeanStd(trainFeatures)
	trainNorm := standardize(trainFeatures, means, stds)
	testNorm := standardize(testFeatures, means, stds)

	// Adding bias term to both train and test
	trainNorm = addBias(trainNorm)
	testNorm = addBias(testNorm)

	fmt.Println("Train column names:", header)
	fmt.Printf("Train features: %v x %v\n", trainFeatures.RawMatrix().Rows, trainFeatures.RawMatrix().Cols)
	fmt.Println("Train standardized features (first row):", mat.Row(nil, 0, trainNorm))

	fmt.Printf("\n")

	fmt.Println("Test column names:", header)
	fmt.Printf("Test features: %v x %v\n", testFeatures.RawMatrix().Rows, testFeatures.RawMatrix().Cols)
	fmt.Println("Test standardized features (first row):", mat.Row(nil, 0,testNorm))
}