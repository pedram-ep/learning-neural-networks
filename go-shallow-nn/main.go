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