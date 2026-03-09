package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	W1 *mat.Dense
	W2 *mat.Dense
}

func newModel(inputSize, hiddenSize int) *Model {
	w1Data := make([]float64, hiddenSize*inputSize)
	w2Data := make([]float64, 1*hiddenSize)
	for i := range w1Data {
		w1Data[i] = 0.01 * rand.NormalFloat64()
	}
	for i := range w2Data {
		w2Data[i] = 0.01 * rand.NormFloat64()
	}
	return &Model{
		W1: mat.NewDense(hiddenSize, inputSize, w1Data),
		W2: mat.NewDense(1, hiddenSize, w2Data),
	}
}

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

func main() {
	// Loading datasets from CSV files
	trainFeatures, trainOutcomes, trainHeader, err := loadCSV("data/diabetes_train.csv", true)
	if err != nil {
		log.Fatal(err)
	}
	testFeatures, _, testHeader, err := loadCSV("data/diabetes_test.csv", false)
	if err != nil {
		log.Fatal(err)
	}

	// Computing mean and std for all columns, and then standardizing both train and test
	means, stds := computeMeanStd(trainFeatures)
	trainNorm := standardize(trainFeatures, means, stds)
	testNorm := standardize(testFeatures, means, stds)

	// Adding bias term to both train and test
	trainBias := addBias(trainNorm)
	testBias := addBias(testNorm)

	// Splitting datasets
	rand.New(rand.NewSource(42)) 
	X_train, y_train, X_val, y_val := splitData(trainBias, trainOutcomes, 0.2)

	fmt.Println("Train column names:\n", trainHeader)
	fmt.Printf("Train features: %v x %v\n", trainFeatures.RawMatrix().Rows, trainFeatures.RawMatrix().Cols)
	fmt.Printf("X_Train: %v x %v | X_val: %v x %v\n", X_train.RawMatrix().Rows, X_train.RawMatrix().Cols, X_val.RawMatrix().Rows, X_val.RawMatrix().Cols)
	fmt.Printf("y_Train: %v | y_val: %v\n", len(y_train), len(y_val))

	fmt.Printf("\n")

	fmt.Println("Test column names:\n", testHeader)
	fmt.Printf("Test features: %v x %v\n", testFeatures.RawMatrix().Rows, testFeatures.RawMatrix().Cols)

	fmt.Printf("\n")

	fmt.Println("Train standardized features (first row):\n", mat.Row(nil, 0, trainBias))
	fmt.Println("Test standardized features (first row):\n", mat.Row(nil, 0,testBias))
}