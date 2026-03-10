package main

import (
	"fmt"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

func evaluate(m *Model, X *mat.Dense, y []float64) float64 {
	_, A2 := m.Predict(X)

	preds := make([]float64, len(y))
	for i := 0; i < len(y); i++ {
		if A2.At(0, 1) > 0.5 {
			preds[i] = 1
		}
	}
	correct := 0
	for i := 0; i < len(y); i++ {
		if preds[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y)) * 100
}

// splitRowMajor splits a matrix with rows as samples.
func splitRowMajor(features *mat.Dense, outcomes []float64, testSize float64) (*mat.Dense, []float64, *mat.Dense, []float64) {
    r, c := features.Dims()
    nTest := int(float64(r) * testSize)
    nTrain := r - nTest

    indices := rand.Perm(r)
    trainRows := indices[:nTrain]
    valRows := indices[nTrain:]

    trainData := make([]float64, nTrain*c)
    trainOut := make([]float64, nTrain)
    for i, idx := range trainRows {
        for j := 0; j < c; j++ {
            trainData[i*c+j] = features.At(idx, j)
        }
        trainOut[i] = outcomes[idx]
    }

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

func printSizesHeader(trainHeader, testHeader []string, trainFeatures, X_train, X_val, testFeatures, trainBias, testBias *mat.Dense, y_train, y_val []float64) {
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

	fmt.Println("--------------------------------------------------------------")
}