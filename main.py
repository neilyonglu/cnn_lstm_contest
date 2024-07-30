import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import logging

# 設置日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 填充與截斷函數
def pad_or_truncate_sequence(sequence, desired_length):
    if len(sequence) < desired_length:
        padding = np.zeros((desired_length - len(sequence), sequence.shape[1]))
        return np.vstack((sequence, padding))
    else:
        return sequence[:desired_length]

# 讀取和預處理資料的函數
def load_and_preprocess_data(train_data_path, train_target_path, sequence_length=256):
    train_data = pd.read_csv(train_data_path)
    train_target = pd.read_csv(train_target_path)

    merged_data = train_data.merge(train_target, on=['PART_ID', 'STAGE'])
    merged_data = merged_data.sort_values(by=['PART_ID', 'TIMESTAMP'])

    sensor_data = merged_data[['SENSOR_1', 'SENSOR_2', 'SENSOR_3', 'SENSOR_4', 'SENSOR_5', 
                               'SENSOR_6', 'SENSOR_7', 'SENSOR_8', 'SENSOR_9', 'SENSOR_10', 
                               'SENSOR_11', 'SENSOR_12', 'SENSOR_13', 'SENSOR_14', 'SENSOR_15']]

    scaler = StandardScaler()
    sensor_data = scaler.fit_transform(sensor_data)

    part_ids = merged_data['PART_ID'].unique()
    X, y = [], []
    for part_id in part_ids:
        part_data = sensor_data[merged_data['PART_ID'] == part_id]
        part_target = merged_data[merged_data['PART_ID'] == part_id]['TARGET'].values

        for i in range(len(part_data) - sequence_length + 1):
            sequence_data = pad_or_truncate_sequence(part_data[i:i + sequence_length], sequence_length)
            X.append(sequence_data)
            y.append(part_target[i + sequence_length - 1])

    X = np.array(X)
    y = np.array(y)
    
    return X, y

# 建立資料集和數據加載器的函數
def create_dataloaders(X, y, test_size=0.2, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def create_test_dataloaders(X, y, batch_size=128):
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# 定義 CNN-LSTM 模型
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=15, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# 計算準確率
def accuracy(y_true, y_pred, threshold=0.01):
    return np.mean(np.abs((y_true - y_pred) / y_true) <= threshold)

# 訓練模型的函數
def train_model(model, train_loader, test_loader, num_epochs=500, patience=10):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    best_loss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve == patience:
            logging.info("Early stopping")
            break

    # 繪製loss曲線
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

    logging.info("\nEpoch-wise losses:")
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

# 測試模型的函數
def test_model(model, test_loader):
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().numpy())
            true_values.extend(targets.numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    test_mape = mean_absolute_percentage_error(true_values, predictions) * 100
    test_accuracy = accuracy(true_values, predictions)

    logging.info(f'Test MAPE: {test_mape:.6f}%')
    logging.info(f'Test Accuracy (within 1% error): {test_accuracy:.2f}%')

    # 繪製預測值vs實際值的散點圖
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.savefig('predictions_vs_true.png')
    plt.close()

    # 繪製預測誤差的直方圖
    errors = (predictions - true_values) / true_values
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.savefig('error_distribution.png')
    plt.close()

    # 輸出一些示例預測
    logging.info("\nSample predictions:")
    for i in range(min(10, len(true_values))):
        logging.info(f"True: {true_values[i]:.6f}, Predicted: {predictions[i]:.6f}, Relative Error: {(predictions[i] - true_values[i]) / true_values[i]:.2%}")


# 主程序
def main():
    X, y = load_and_preprocess_data('./600_train_data(1).csv', './600_train_target.csv')
    
    model = CNN_LSTM()

    if os.path.exists('./model/3/best_model.pth'):
        print("Loading existing model...")
        test_loader = create_test_dataloaders(X, y)
        model.load_state_dict(torch.load('./model/3/best_model.pth'))
        test_model(model, test_loader)
    else:
        print("Training new model...")
        train_loader, test_loader = create_dataloaders(X, y)
        train_model(model, train_loader, test_loader)
        print("Training completed. Now testing the model...")
        model.load_state_dict(torch.load('./model/3/best_model.pth'))
        test_model(model, test_loader)

if __name__ == "__main__":
    main()
