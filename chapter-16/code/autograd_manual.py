import ctypes
import math
import random
from ctypes import POINTER, c_float, c_int

import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Load the C wrapper for cuBLAS
try:
    cublas_lib = ctypes.CDLL("./libcublas_wrapper.so")
    print("Successfully loaded cuBLAS wrapper library")
except Exception as e:
    print(f"Error loading library: {e}")
    raise

# Set up function signatures
cublas_lib.init_cublas.restype = c_int
cublas_lib.cleanup_cublas.restype = c_int
cublas_lib.sgemm_wrapper.argtypes = [
    POINTER(c_float),  # A
    POINTER(c_float),  # B
    POINTER(c_float),  # C
    c_int,  # m
    c_int,  # n
    c_int,  # k
    c_int,  # transa
    c_int,  # transb
]
cublas_lib.sgemm_wrapper.restype = c_int
cublas_lib.gpu_alloc.argtypes = [POINTER(c_float), c_int]
cublas_lib.gpu_alloc.restype = POINTER(c_float)
cublas_lib.gpu_to_host.argtypes = [POINTER(c_float), POINTER(c_float), c_int]
cublas_lib.gpu_to_host.restype = c_int
cublas_lib.gpu_free.argtypes = [POINTER(c_float)]

# Initialize cuBLAS
result = cublas_lib.init_cublas()
if result != 0:
    raise Exception("Failed to initialize cuBLAS")


# Helper function to convert numpy array to C float pointer
def np_to_c_float_p(np_array):
    # Make sure the array is contiguous and float32
    np_array = np.ascontiguousarray(np_array, dtype=np.float32)
    return np_array.ctypes.data_as(POINTER(c_float))


# Direct matrix multiplication using our C cuBLAS wrapper
def cublas_matmul(a, b, transa=False, transb=False):
    # Ensure arrays are float32 and contiguous
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    # Get dimensions
    if not transa and not transb:
        # C(m,n) = A(m,k) @ B(k,n)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    elif transa and not transb:
        # C(m,n) = A(k,m)^T @ B(k,n)
        k, m = a.shape
        k2, n = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    elif not transa and transb:
        # C(m,n) = A(m,k) @ B(n,k)^T
        m, k = a.shape
        n, k2 = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"
    else:  # transa and transb
        # C(m,n) = A(k,m)^T @ B(n,k)^T
        k, m = a.shape
        n, k2 = b.shape
        assert k == k2, "Inner dimensions must match for matrix multiplication"

    # Allocate output array
    c = np.zeros((m, n), dtype=np.float32)

    # Get C pointers to arrays
    a_ptr = np_to_c_float_p(a)
    b_ptr = np_to_c_float_p(b)
    c_ptr = np_to_c_float_p(c)

    # Call C function
    result = cublas_lib.sgemm_wrapper(
        a_ptr,
        b_ptr,
        c_ptr,
        c_int(m),
        c_int(n),
        c_int(k),
        c_int(1 if transa else 0),
        c_int(1 if transb else 0),
    )

    if result != 0:
        raise Exception(f"cublas_matmul failed with code {result}")

    return c


# Base Layer class
class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)


# Linear Layer with direct cuBLAS
class Linear(Layer):
    def __init__(self, in_features, out_features, name=None):
        # Xavier/Glorot initialization
        stdv = 1.0 / math.sqrt(in_features)
        self.weight = np.random.uniform(
            -stdv, stdv, size=(out_features, in_features)
        ).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None

        # Store input for backward pass
        self.input = None

        # Layer name for parameter identification
        self.name = name or id(self)

    def forward(self, x):
        # Save input for backward pass
        self.input = x

        # Compute output: y = x @ W^T + b using cuBLAS
        # Note: we transpose weight here (transa=False, transb=True)
        output = cublas_matmul(x, self.weight, transa=False, transb=True)

        # Add bias
        if self.bias is not None:
            output += self.bias

        return output

    def backward(self, grad_output):
        # Compute gradients for weights: dL/dW = grad_output.T @ input
        self.grad_weight = cublas_matmul(
            grad_output, self.input, transa=True, transb=False
        )

        # Compute gradients for bias: dL/db = sum(dL/dY)
        self.grad_bias = np.sum(grad_output, axis=0)

        # Compute gradient for input: dL/dX = dL/dY @ W
        grad_input = cublas_matmul(grad_output, self.weight, transa=False, transb=False)

        return grad_input

    def parameters(self):
        return [(self.weight, f"{self.name}_weight"), (self.bias, f"{self.name}_bias")]

    def get_gradients(self):
        return [
            (self.grad_weight, f"{self.name}_weight"),
            (self.grad_bias, f"{self.name}_bias"),
        ]


# Activation Functions
class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, x):
        # Compute sigmoid with numerical stability
        self.output = np.zeros_like(x)
        mask = x >= 0
        self.output[mask] = 1 / (1 + np.exp(-x[mask]))
        exp_x = np.exp(x[~mask])
        self.output[~mask] = exp_x / (1 + exp_x)
        return self.output

    def backward(self, grad_output):
        # Compute gradient: dL/dx = dL/dy * dy/dx
        # For sigmoid: dy/dx = y * (1 - y)
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

    def parameters(self):
        return []

    def get_gradients(self):
        return []


class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input

    def parameters(self):
        return []

    def get_gradients(self):
        return []


# MSE Loss
class MSELoss:
    def __init__(self):
        self.prediction = None
        self.target = None

    def forward(self, pred, target):
        self.prediction = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        batch_size = self.prediction.shape[0]
        return 2.0 * (self.prediction - self.target) / batch_size

    def __call__(self, pred, target):
        return self.forward(pred, target)


# Sequential Model
class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_gradients(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "get_gradients"):
                grads.extend(layer.get_gradients())
        return grads


# SGD Optimizer
class SGD:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

        # Initialize velocities for each parameter
        for param, name in model.parameters():
            self.velocities[name] = np.zeros_like(param)

    def step(self):
        # Get current gradients and parameters
        gradients = {name: grad for grad, name in self.model.get_gradients()}

        # Update parameters with their gradients
        for param, param_name in self.model.parameters():
            if param_name in gradients and gradients[param_name] is not None:
                # Update velocity with momentum
                self.velocities[param_name] = (
                    self.momentum * self.velocities[param_name]
                    + self.learning_rate * gradients[param_name]
                )

                # Update parameter
                param -= self.velocities[param_name]

    def zero_grad(self):
        # No need to zero gradients as they're recomputed each time
        pass


# Training function
def train_model(model, criterion, optimizer, x_data, y_data, epochs=1000, debug=True):
    losses = []

    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        losses.append(loss)

        # Backward pass
        grad_output = criterion.backward()
        model.backward(grad_output)

        # Debug gradients
        if debug and epoch % 100 == 0:
            gradients = model.get_gradients()
            print(f"Epoch {epoch}, Loss: {loss}")
            if gradients:
                for grad, name in gradients:
                    if grad is not None:
                        grad_norm = np.linalg.norm(grad)
                        print(f"  Gradient norm ({name}): {grad_norm}")

        # Update weights
        optimizer.step()

    return model, losses


# Example usage with debugging
def main():
    try:
        # Generate some simple data
        input_size = 2
        hidden_size = 4
        output_size = 1
        num_samples = 100

        # Create XOR-like data
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

        y = np.array([[0], [1], [1], [0]], dtype=np.float32)

        # Repeat data to create more samples
        x = np.tile(x, (num_samples // 4 + 1, 1))[:num_samples]
        y = np.tile(y, (num_samples // 4 + 1, 1))[:num_samples]

        # Add some noise to make it more realistic
        x += np.random.normal(0, 0.05, x.shape).astype(np.float32)

        # Create a simple model with named layers
        model = Sequential(
            [
                Linear(input_size, hidden_size, name="layer1"),
                ReLU(),
                Linear(hidden_size, hidden_size, name="layer2"),
                ReLU(),
                Linear(hidden_size, output_size, name="layer3"),
                Sigmoid(),
            ]
        )

        # Prepare loss function and optimizer
        criterion = MSELoss()
        optimizer = SGD(model, learning_rate=0.1, momentum=0.9)

        # Train the model
        print("Starting training...")
        trained_model, losses = train_model(
            model, criterion, optimizer, x, y, epochs=1000
        )

        print("\nFinal predictions:")
        test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

        predictions = model(test_inputs)
        print("Input => Output (Expected)")
        for i in range(4):
            print(f"{test_inputs[i].tolist()} => {predictions[i][0]:.4f} ({y[i%4][0]})")

        print("\nTraining complete!")
        print(f"Final loss: {losses[-1]}")

    finally:
        # Clean up cuBLAS resources
        result = cublas_lib.cleanup_cublas()
        if result != 0:
            print("Warning: Failed to properly clean up cuBLAS")
        else:
            print("cuBLAS cleaned up successfully")


if __name__ == "__main__":
    main()
