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

func (m *Model) UpdateWeights(X, A1, A2 *mat.Dense, y []float64, learningRate float64) {
	n := float64(X.RawMatrix().Cols)
	yMat := mat.NewDense(1, len(y), y)

	var diff, prod, oneMinusA2, shared mat.Dense
	diff.Sub(yMat, A2)
	prod.MulElem(&diff, A2)
	oneMinusA2.Apply(func(_, _ int, v float64) float64 {return 1 - v }, A2)
	prod.MulElem(&prod, &oneMinusA2)
	shared.Scale(2/n, &prod)

	var sharedTimesA1T mat.Dense
	sharedTimesA1T.Mul(&shared, A1.T())
	deltaW2 := new(mat.Dense)
	deltaW2.Scale(learningRate, &sharedTimesA1T)
	m.W2.Add(m.W2, deltaW2)

	reluGrad := new(mat.Dense)
	reluGrad.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, A1)

	var sharedT mat.Dense
	sharedT.CloneFrom(shared.T())
	var sharedT_W2 mat.Dense
	sharedT_W2.Mul(&sharedT, m.W2)
	
	var temp mat.Dense
	temp.CloneFrom(sharedT_W2.T())

	var grad mat.Dense
	grad.MulElem(&temp, reluGrad)

	var grad_XT mat.Dense
	grad_XT.Mul(&grad, X.T())

	deltaW1 := new(mat.Dense)
	deltaW1.Scale(learningRate, &grad_XT)
	m.W1.Add(m.W1, deltaW1)
}

func (m *Model) Fit(X *mat.Dense, y []float64, learningRate float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		A1, A2 := m.Predict(X)

		yMat := mat.NewDense(1, len(y), y)

		var diff mat.Dense
		diff.Sub(yMat, A2)

		var sq mat.Dense
		sq.MulElem(&diff, &diff)
		loss := mat.Sum(&sq) / float64(len(y))

		if epoch%10 == 0 {
			println("Epoch:", epoch, "Loss:", loss)
		}

		m.UpdateWeights(X, A1, A2, y, learningRate)
	}
}