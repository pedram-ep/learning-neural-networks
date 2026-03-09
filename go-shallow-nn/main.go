package main

import (
	"fmt"
	"log"
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

func confusionMatrix(preds, actual []float64) [2][2]int {
	var cm[2][2]int
	for i := range preds {
		cm[int(actual[i])][int(preds[i])]++
	}
	return cm
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

	// Printing Dataset sizes and headers
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

	// trainMat := mat.DenseCopyOf(trainBias.T())
	// testMat := mat.DenseCopyOf(testBias.T())

	X_train_row, y_train, X_val_row, y_val := splitRowMajor(trainBias, trainOutcomes, 0.2)
	X_train = mat.DenseCopyOf(X_train_row.T())
	X_val = mat.DenseCopyOf(X_val_row.T())

	// X_test := mat.DenseCopyOf(testBias.T())

	inputSize := X_train.RawMatrix().Rows // after transpose, rows = features, cols = samples
	hiddenSize := 1000
	model := NewModel(inputSize, hiddenSize)

	model.Fit(X_train, y_train, 0.1, 200)

	trainAcc := evaluate(model, X_train, y_train)
	valAcc := evaluate(model, X_val, y_val)
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc)
	fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc)

	// _, A2_test := model.Predict(X_test)
	// preds := make([]float64, A2_test.RawMatrix().Cols)
	// for i := range preds {
	// 	if A2_test.At(0, i) > 0.5 {
	// 		preds[i] = 1
	// 	}
	// }
	// fmt.Printf("Positive predictions on test: %d / %d\n", int(sum(preds)), len(preds))
}