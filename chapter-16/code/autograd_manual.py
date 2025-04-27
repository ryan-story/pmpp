import torch
import math
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

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

# Linear Layer
class Linear(Layer):
    def __init__(self, in_features, out_features, name=None):
        # Xavier/Glorot initialization
        stdv = 1. / math.sqrt(in_features)
        self.weight = torch.empty(out_features, in_features).uniform_(-stdv, stdv)
        self.bias = torch.zeros(out_features)
        
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
        
        # Compute output: y = x * W^T + b
        output = x.matmul(self.weight.t())
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)
        
        return output
    
    def backward(self, grad_output):
        # Compute gradients for weights: dL/dW = (dL/dY)^T * X
        self.grad_weight = grad_output.t().matmul(self.input)
        
        # Compute gradients for bias: dL/db = sum(dL/dY)
        self.grad_bias = grad_output.sum(0)
        
        # Compute gradient for input: dL/dX = dL/dY * W
        grad_input = grad_output.matmul(self.weight)
        
        return grad_input
    
    def parameters(self):
        return [(self.weight, f"{self.name}_weight"), (self.bias, f"{self.name}_bias")]
    
    def get_gradients(self):
        return [(self.grad_weight, f"{self.name}_weight"), (self.grad_bias, f"{self.name}_bias")]

# Activation Functions
class Sigmoid(Layer):
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        # Compute sigmoid with numerical stability
        self.output = torch.zeros_like(x)
        mask = x >= 0
        self.output[mask] = 1 / (1 + torch.exp(-x[mask]))
        exp_x = torch.exp(x[~mask])
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
        return torch.clamp(x, min=0)
    
    def backward(self, grad_output):
        grad_input = grad_output.clone()
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
        return ((pred - target) ** 2).mean()
    
    def backward(self):
        batch_size = self.prediction.size(0)
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
            if hasattr(layer, 'get_gradients'):
                grads.extend(layer.get_gradients())
        return grads

# SGD Optimizer - FIXED VERSION
class SGD:
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        
        # Initialize velocities for each parameter
        for param, name in model.parameters():
            self.velocities[name] = torch.zeros_like(param)
    
    def step(self):
        # Get current gradients and parameters
        gradients = {name: grad for grad, name in self.model.get_gradients()}
        
        # Update parameters with their gradients
        for param, param_name in self.model.parameters():
            if param_name in gradients and gradients[param_name] is not None:
                # Update velocity with momentum
                self.velocities[param_name] = (self.momentum * self.velocities[param_name] + 
                                              self.learning_rate * gradients[param_name])
                
                # Update parameter
                param.sub_(self.velocities[param_name])
    
    def zero_grad(self):
        # No need to zero gradients as they're recomputed each time
        pass

# Training function with debugging
def train_model(model, criterion, optimizer, x_data, y_data, epochs=1000, debug=True):
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        losses.append(loss.item())
        
        # Backward pass
        grad_output = criterion.backward()
        model.backward(grad_output)
        
        # # Debug gradients
        # if debug and epoch % 100 == 0:
        #     gradients = model.get_gradients()
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")
        #     if gradients:
        #         for grad, name in gradients:
        #             if grad is not None:
        #                 grad_norm = torch.norm(grad).item()
        #                 print(f"  Gradient norm ({name}): {grad_norm}")
        
        # Update weights
        optimizer.step()
    
    return model, losses

# Example usage with debugging
def main():
    # Generate some simple data
    input_size = 2
    hidden_size = 4
    output_size = 1
    num_samples = 100
    
    # Create XOR-like data
    x = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    
    y = torch.tensor([
        [0],
        [1],
        [1],
        [0]
    ], dtype=torch.float32)
    
    # Repeat data to create more samples
    x = x.repeat(num_samples//4 + 1, 1)[:num_samples]
    y = y.repeat(num_samples//4 + 1, 1)[:num_samples]
    
    # Add some noise to make it more realistic
    x = x + torch.randn_like(x) * 0.05
    
    # Create a simple model with named layers
    model = Sequential([
        Linear(input_size, hidden_size, name="layer1"),
        ReLU(),
        Linear(hidden_size, hidden_size, name="layer2"),
        ReLU(),
        Linear(hidden_size, output_size, name="layer3"),
        Sigmoid()
    ])
    
    # Prepare loss function and optimizer
    criterion = MSELoss()
    optimizer = SGD(model, learning_rate=0.1, momentum=0.9)
    
    # Train the model with debugging
    trained_model, losses = train_model(model, criterion, optimizer, x, y, epochs=1000)
    
    print("\nFinal predictions:")
    test_inputs = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    
    predictions = model(test_inputs)
    print("Input => Output (Expected)")
    for i in range(4):
        print(f"{test_inputs[i].tolist()} => {predictions[i].item():.4f} ({y[i].item()})")
    
    print("\nTraining complete!")
    print(f"Final loss: {losses[-1]}")

if __name__ == "__main__":
    main()