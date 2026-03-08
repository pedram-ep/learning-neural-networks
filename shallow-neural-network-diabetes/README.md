# Shallow Neural Network for Diabetes Prediction

A from-scratch implementation of a shallow neural network to predict diabetes using the Pima Indians Diabetes dataset.

## Description

This project implements a shallow neural network with one hidden layer (1000 neurons, ReLU activation) and sigmoid output layer. The model is trained using gradient descent, with all calculations implemented manually using NumPy (no deep learning frameworks).

## Dataset

I used the simple dataset "Pima Indians Diabetes". This dataset contains the information of 768 native American women from the Pima tribe, reviewing the factors related to type 2 diabetes.
This dataset contains age, weight, height, family history of diabetes, blood pressure, blood glucose level, and other factors.

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

## Model

A shallow neural network with one hidden layer (with `1000` neurons) and the rectified linear unit activation function (`ReLu`). The activation function of the output later is sigmoid. 

```py
sigmoid_Z = 1 / (1 + np.exp(-Z))
ReLu_Z = np.maximum(0, Z)
```

- Sigmoid function:

| Sigmoid Function | Derivative of Sigmoid Function |
| ---------------- | ------------------------------ |
| $f(z) = \frac{1}{1 + e^{-z}}$ | $f'(z) = f(z) (1 - f(z))$ |

- ReLu function:

| ReLu Function | Derivative of ReLu Function |
| ------------- | --------------------------- |
| $$f(z) = \begin{cases} 0 & \text{if } z < 0 \\ z & \text{if } z \geq 0\end{cases}$$ | $$f'(z) = \begin{cases} 0 & \text{if } z < 0 \\ 1 & \text{if } z \geq 0\end{cases}$$ |

### `__init__` method

In this method we initialize the weights of the hidden layer and output layer (`w1` and `w2`) randomly, with the mean $0$ and the standard deviation $0.01$.

### `predict` method

The method `predict(self, inputs)` gets the inputs, and returns the output of both layers (`A_1` and `A_2`).

The formulas for these actions:

$$Z^{[1]}=W^{[1]}.X$$
$$A^{[1]}= \operatorname{ReLU}(Z^{[1]})$$
$$Z^{[2]}=W^{[2]}A^{[1]}$$
$$A^{[2]}=\sigma(Z^{[2]})=\frac{1}{1+e^{-Z^{[2]}}}=Y_{pred}$$

### `update_weights_for_one_epoch` method

The method `update_weights_for_one_epoch(self, inputs, outputs, learning_rate)` updates the weights of the network for one `epoch`. The variable `learning_rate` is $\alpha$.
This is how `w2` is updates:

$$W^{[2]} = W^{[2]} + \Delta W^{[2]}$$
$$\Delta W^{[2]} = - \alpha \frac{\partial cost}{\partial W^{[2]}}$$
$$\frac{\partial cost}{\partial W^{[2]}} = (\frac{-2}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))\bullet A^{[1]T}$$
$$W^{[2]}=W^{[2]}+(\frac{2 \alpha}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))\bullet A^{[1]T}$$
And this is how `w1` is updated:

$$W^{[1]} = W^{[1]} + \Delta W^{[1]}$$
$$\Delta W^{[1]} = - \alpha \frac{\partial cost}{\partial W^{[1]}}$$

$$\frac{\partial cost}{\partial W^{[1]}} = (((\frac{-2}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))^T\bullet W^{[2]})^T\odot \frac{\partial A^{[1]}}{\partial Z^{[1]}}) \bullet X^T$$

$$W^{[1]}=W^{[1]}+(((\frac{2 \alpha}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))^T\bullet W^{[2]})^T\odot \frac{\partial A^{[1]}}{\partial Z^{[1]}}) \bullet X^T$$
The value of $\frac{\partial A^{[1]}}{\partial Z^{[1]}}$ is calculated using the code snippet below:

```py
relu_gradient = np.where(A_1 > 0, 1, 0)
```

And a part of $\Delta W^{[1]}$ is calculated in $\Delta W^{[2]}$, so we kept it as variable to avoid additional calculations.

### `fit` method

The mthod `fit(self, inputs, outputs, learning_rate, epochs=64)` updates the weights of the network to the number of `epochs`.

## Results

- Training accuracy: ~76%
- Validation accuracy: ~81%

## Requirements

```bash
pip install -r requirements.txt
```