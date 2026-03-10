package main

import (
	"math"
	"gonum.org/v1/gonum/mat"
)

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
		stds[j] = math.Sqrt(sumSq / float64(r))
	}
	return means, stds
}

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