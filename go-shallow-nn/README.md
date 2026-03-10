# Shallow Neural Network in Go

A Go implementation of a shallow neural network for diabetes prediction, porting the Python version to Go.

## Description

This project implements a shallow neural network with one hidden layer (1000 neurons, ReLU activation) and sigmoid output layer. The model is trained using gradient descent, with all calculations implemented manually using Go's standard libraries (no deep learning frameworks).

This is a port of the [Shallow Neural Network for Diabetes Prediction](../shallow-neural-network-diabetes) project from Python to Go, demonstrating the same neural network concepts.

This mini-project was done as a practice of data science programming in Go programming language.

## Dataset

The project uses the "Pima Indians Diabetes" dataset containing information of 768 native American women from the Pima tribe, with factors related to type 2 diabetes.

| Column | Description |
| ------ | ----------- |
| `Pregnancies` | the number of pregnancies |
| `Glucose` | blood glucose level (`mg/dL`) |
| `BloodPressure` | systolic blood pressure (`mmHg`) |
| `SkinThickness` | the thickness of skin (`mm`) |
| `Insulin` | blood insulin level (`μU/mL`) |
| `BMI` | body-mass-index (`kg/m^2`) |
| `DiabetesPedigreeFunction` | a function showing family diabetes history |
| `Age` | age (years) |
| `Outcome` | having diabetes (`1`) or not having it (`0`) |

## Model Architecture

A shallow neural network with one hidden layer (1000 neurons) using ReLU activation, and a sigmoid output layer.

- **Hidden Layer:** 1000 neurons with ReLU activation
- **Output Layer:** 1 neuron with Sigmoid activation
- **Training Method:** Gradient descent with backpropagation
- **Loss Function:** Binary cross-entropy

## Results

- Training accuracy: ~65%
- Validation accuracy: ~63%

## Requirements

Go 1.16 or higher

## Usage

```bash
go run .
```

## Languages and Tools

**Languages:** Go
**Tags:** `from-scratch`, `classification`, `backpropagation`
