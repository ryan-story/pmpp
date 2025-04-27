import torch

# Custom Linear Layer without autograd
class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights and biases
        self.weight = torch.randn(out_features, in_features) * 0.1  # Scale to avoid exploding gradients
        self.bias = torch.zeros(out_features)
        
        # Initialize gradients
        self.grad_weight = torch.zeros_like(self.weight)
        self.grad_bias = torch.zeros_like(self.bias)
        
        # Cache for backward pass
        self.input = None
    
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
        self.grad_weight.add_(grad_output.t().matmul(self.input))
        
        # Compute gradients for bias: dL/db = sum(dL/dY)
        self.grad_bias.add_(grad_output.sum(0))
        
        # Compute gradient for input: dL/dX = dL/dY * W
        grad_input = grad_output.matmul(self.weight)
        
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

# Custom Sigmoid activation without autograd
class Sigmoid:
    def __init__(self):
        # Cache for backward pass
        self.output = None
    
    def forward(self, x):
        # Compute sigmoid and cache for backward pass
        self.output = 1.0 / (1.0 + torch.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        # Compute gradient: dL/dx = dL/dy * dy/dx
        # For sigmoid: dy/dx = y * (1 - y)
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        return []  # Sigmoid has no parameters

# Simple MSE Loss
class MSELoss:
    def __init__(self):
        self.prediction = None
        self.target = None
    
    def forward(self, pred, target):
        self.prediction = pred
        self.target = target
        return ((pred - target) ** 2).mean()
    
    def backward(self):
        # For MSE, dL/dy = 2 * (y - t) / n
        # where n is the number of elements
        batch_size = self.prediction.size(0)
        return 2.0 * (self.prediction - self.target) / batch_size
    
    def __call__(self, pred, target):
        return self.forward(pred, target)

# Simple neural network using our custom layers
class SimpleNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = Linear(input_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        # First layer
        x1 = self.linear1(x)
        
        # Sigmoid activation
        x2 = self.sigmoid(x1)
        
        # Second layer
        output = self.linear2(x2)
        
        return output
    
    def backward(self, grad_output):
        # Backpropagation through the second linear layer
        grad = self.linear2.backward(grad_output)
        
        # Backpropagation through the sigmoid layer
        grad = self.sigmoid.backward(grad)
        
        # Backpropagation through the first linear layer
        grad = self.linear1.backward(grad)
        
        return grad
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []
        params.extend(self.linear1.parameters())
        params.extend(self.linear2.parameters())
        return params

# Simple SGD optimizer
class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def zero_grad(self):
        for param, grad in self.parameters:
            grad.zero_()
    
    def step(self):
        for param, grad in self.parameters:
            param.sub_(self.learning_rate * grad)

# Training function
def train_model(model, criterion, optimizer, x_data, y_data, epochs=100):
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_data)
        loss = criterion(outputs, y_data)
        
        # Backward pass
        grad_output = criterion.backward()
        model.backward(grad_output)
        
        # Update weights
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model

# Create and train the model
def main():
    # Generate some fake data
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 100
    
    x = torch.randn(num_samples, input_size)
    # Create a simple target function: y = sigmoid(sum(x))
    true_w = torch.randn(input_size)
    y = 1.0 / (1.0 + torch.exp(-x.matmul(true_w))).unsqueeze(1)
    
    # Initialize the model, loss, and optimizer
    model = SimpleNetwork(input_size, hidden_size, output_size)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=0.01)
    
    # Train the model
    trained_model = train_model(model, criterion, optimizer, x, y, epochs=100)
    
    print("Training complete!")

if __name__ == "__main__":
    main()