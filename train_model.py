import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Define a Compact Neural Network for zkML
class VerifiableClassifier(nn.Module):
    def __init__(self):
        super(VerifiableClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # 4 input features, 8 hidden neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)  # 2 output classes

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 2. Prepare Tabular Dataset (Using 4 features for speed)
    data = load_breast_cancer()
    X = data.data[:, :4]
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # 3. Train Model
    model = VerifiableClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Training PyTorch classification model...")
    model.train()
    for epoch in range(60):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"Training completed. Final Loss: {loss.item():.4f}")

    # 4. Save PyTorch Weights & Export to ONNX
    torch.save(model.state_dict(), "models/model.pth")
    
    model.eval()
    sample_input = torch.tensor(X_test[0:1], dtype=torch.float32)
    onnx_path = "models/model.onnx"

    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model successfully exported to: {onnx_path}")

    # 5. Save Sample Input as JSON (Required Witness input for EZKL)
    input_data = {"input_data": [sample_input.flatten().tolist()]}
    with open("data/input.json", "w") as f:
        json.dump(input_data, f, indent=2)

    print("Sample inference data saved to: data/input.json")

if __name__ == "__main__":
    main()