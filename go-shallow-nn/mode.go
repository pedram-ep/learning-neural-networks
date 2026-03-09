package main

import (
	"math"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

type Model struct {
	W1 *mat.Dense
	W2 *mat.Dense
}

func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x))}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func newModel(inputSize, hiddenSize int) *Model {
	w1Data := make([]float64, hiddenSize*inputSize)
	w2Data := make([]float64, 1*hiddenSize)
	for i := range w1Data {
		w1Data[i] = 0.01 * rand.NormFloat64()
	}
	for i := range w2Data {
		w2Data[i] = 0.01 * rand.NormFloat64()
	}
	return &Model{
		W1: mat.NewDense(hiddenSize, inputSize, w1Data),
		W2: mat.NewDense(1, hiddenSize, w2Data),
	}
}

func (m *Model) Predict(X *mat.Dense) (A1, A2 *mat.Dense) {
	var Z1, Z2 mat.Dense

	Z1.Mul(m.W1, X)

	A1 = &mat.Dense{}
	A1.Apply(func(i, j int, v float64) float64 { return relu(v) }, &Z1)

	Z2.Mul(m.W2, X)

	A2 = &mat.Dense{}
	A2.Apply(func(i, j int, v float64) float64 { return sigmoid(v) }, &Z2)

	return A1, A2
}