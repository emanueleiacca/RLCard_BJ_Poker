import torch

# Load the PyTorch model from the uploaded .pth file
model_path = 'rlcard/experiments/dmc_tuning_result_3/blackjack/0_2003400.pth'
model = torch.load(model_path)
print(model)
model_attributes = dir(model)

# Filter the attributes to show only non-dunder (special) methods and attributes
filtered_attributes = [attr for attr in model_attributes if not attr.startswith('__')]

print(filtered_attributes)
model.eval()  # Switch to evaluation mode if needed

# Check the device the model is using
print("Model is using device:", model.device)

# Example of using predict method if you have a sample input available
sample_input = torch.randn(model.action_shape) # Create a random tensor that matches the expected input shape
predicted_output = model.predict(sample_input.to(model.device))
print("Predicted Output:", predicted_output)
