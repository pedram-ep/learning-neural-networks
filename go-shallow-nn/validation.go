package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type ConfusionMatrix struct {
	TruePositives int
	TrueNegatives int
	FalsePositives int
	FalseNegatives int
}

func (m *Model) ComputeConfusionMatrix(predictions []float64, actual []float64, threshold float64) *ConfusionMatrix {
	cm := &ConfusionMatrix{}

	for i := range predictions {
		predicted := 0
		if predictions[i] >= threshold {
			predicted = 1
		}

		actualLabel := int(actual[i])

		if predicted == 1 && actualLabel == 1 {
			cm.TruePositives++
		} else if predicted == 0 && actualLabel == 0 {
			cm.TrueNegatives++
		} else if predicted == 1 && actualLabel == 0 {
			cm.FalsePositives++
		} else if predicted == 0 && actualLabel == 1 {
			cm.FalseNegatives++
		}
	}

	return cm
}

func (cm *ConfusionMatrix) Precision() float64 {
	if cm.TruePositives+cm.FalsePositives == 0 {
		return 0
	}
	return float64(cm.TruePositives) / float64(cm.TruePositives+cm.FalsePositives)
}

func (cm *ConfusionMatrix) Recall() float64 {
	if cm.TruePositives+cm.FalseNegatives == 0 {
		return 0
	}
	return float64(cm.TruePositives) / float64(cm.TruePositives+cm.FalseNegatives)
}

func (cm *ConfusionMatrix) Accuracy() float64 {
	total := cm.TrueNegatives + cm.TruePositives + cm.FalsePositives + cm.FalseNegatives
	if total == 0 {
		return 0
	}
	return float64(cm.TruePositives + cm.TrueNegatives) / float64(total)
}

func (cm *ConfusionMatrix) F1Score() float64 {
	precision := cm.Precision()
	recall := cm.Recall()
	if precision + recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

func printMetrics(model *Model, X_train, X_val *mat.Dense, y_train, y_val []float64, threshold float64) {
    // Training set confusion matrix
    fmt.Println("\n=== Training Set Confusion Matrix ===")
    _, A2_train := model.Predict(X_train)
    trainPredictions := make([]float64, A2_train.RawMatrix().Cols)
    for i := range trainPredictions {
        trainPredictions[i] = A2_train.At(0, i)
    }
    
    cmTrain := model.ComputeConfusionMatrix(trainPredictions, y_train, threshold)
    
    fmt.Printf("True Positives:  %d\n", cmTrain.TruePositives)
    fmt.Printf("True Negatives:  %d\n", cmTrain.TrueNegatives)
    fmt.Printf("False Positives: %d\n", cmTrain.FalsePositives)
    fmt.Printf("False Negatives: %d\n", cmTrain.FalseNegatives)
    fmt.Printf("\nMetrics:\n")
    fmt.Printf("Accuracy:  %.4f\n", cmTrain.Accuracy())
    fmt.Printf("Precision: %.4f\n", cmTrain.Precision())
    fmt.Printf("Recall:    %.4f\n", cmTrain.Recall())
    fmt.Printf("F1 Score:  %.4f\n", cmTrain.F1Score())

    // Validation set confusion matrix
    fmt.Println("\n=== Validation Set Confusion Matrix ===")
    _, A2_val := model.Predict(X_val)
    valPredictions := make([]float64, A2_val.RawMatrix().Cols)
    for i := range valPredictions {
        valPredictions[i] = A2_val.At(0, i)
    }
    
    cmVal := model.ComputeConfusionMatrix(valPredictions, y_val, threshold)
    
    fmt.Printf("True Positives:  %d\n", cmVal.TruePositives)
    fmt.Printf("True Negatives:  %d\n", cmVal.TrueNegatives)
    fmt.Printf("False Positives: %d\n", cmVal.FalsePositives)
    fmt.Printf("False Negatives: %d\n", cmVal.FalseNegatives)
    fmt.Printf("\nMetrics:\n")
    fmt.Printf("Accuracy:  %.4f\n", cmVal.Accuracy())
    fmt.Printf("Precision: %.4f\n", cmVal.Precision())
    fmt.Printf("Recall:    %.4f\n", cmVal.Recall())
    fmt.Printf("F1 Score:  %.4f\n", cmVal.F1Score())
}