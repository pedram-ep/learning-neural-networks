package main

import (
	"fmt"
	"log"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

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

	trainBias := addBias(trainNorm)
	testBias := addBias(testNorm)

	// Splitting datasets
	rand.New(rand.NewSource(42)) 
	X_train, y_train, X_val, y_val := splitData(trainBias, trainOutcomes, 0.2)

	// Printing the sizes of matrices and Headers
	printSizesHeader(trainHeader, testHeader, trainFeatures, X_train, X_val, testFeatures, trainBias, testBias, y_train, y_val)

	X_train_row, y_train, X_val_row, y_val := splitRowMajor(trainBias, trainOutcomes, 0.2)
	X_train = mat.DenseCopyOf(X_train_row.T())
	X_val = mat.DenseCopyOf(X_val_row.T())

	// Training the mmodel
	inputSize := X_train.RawMatrix().Rows
	hiddenSize := 1000
	model := NewModel(inputSize, hiddenSize)

	model.Fit(X_train, y_train, 0.01, 200)

	// Compact metrics printing
	trainAcc := evaluate(model, X_train, y_train)
	valAcc := evaluate(model, X_val, y_val)
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc)
	fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc)

	// Full metrics printing
	printMetrics(model, X_train, X_val, y_train, y_val, 0.5)
}