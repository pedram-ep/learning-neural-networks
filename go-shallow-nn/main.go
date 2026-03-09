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

	// Adding bias term to both train and test
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

	inputSize := X_train.RawMatrix().Rows
	hiddenSize := 1000
	model := NewModel(inputSize, hiddenSize)

	model.Fit(X_train, y_train, 0.01, 200)

	trainAcc := evaluate(model, X_train, y_train)
	valAcc := evaluate(model, X_val, y_val)
	fmt.Printf("Training Accuracy: %.2f%%\n", trainAcc)
	fmt.Printf("Validation Accuracy: %.2f%%\n", valAcc)

	// X_test := mat.DenseCopyOf(testBias.T())

	// _, A2_test := model.Predict(X_test)
	// preds := make([]float64, A2_test.RawMatrix().Cols)
	// positivePreds := 0
	// for i := range preds {
	// 	if A2_test.At(0, i) > 0.5 {
	// 		preds[i] = 1
	// 		positivePreds += 1
	// 	}
	// }
	

	// fmt.Printf("Positive predictions on test: %d/%d\n", positivePreds, len(preds))
}