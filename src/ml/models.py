"""
Machine learning models for Street Level Change Detection.

This module provides classes and functions for training and evaluating
machine learning models for detecting changes in street-level imagery.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ChangeDetectionDataset(Dataset):
    """
    Dataset for change detection.
    
    This class provides a PyTorch Dataset for training and evaluating
    change detection models.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Initialize a ChangeDetectionDataset.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        labels : np.ndarray
            Labels
        transform : Optional[Callable], default=None
            Transform to apply to features
        """
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """
        Get the number of samples.
        
        Returns
        -------
        int
            Number of samples
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Parameters
        ----------
        idx : int
            Index
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Features and label
        """
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class ChangeDetectionModel:
    """
    Base class for change detection models.
    
    This class provides a common interface for training and evaluating
    change detection models.
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize a ChangeDetectionModel.
        
        Parameters
        ----------
        model_type : str, default='random_forest'
            Type of model to use
        **kwargs
            Additional parameters for the model
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.kwargs = kwargs
    
    def _create_model(self):
        """
        Create the underlying model.
        """
        if self.model_type == 'random_forest':
            n_estimators = self.kwargs.get('n_estimators', 100)
            max_depth = self.kwargs.get('max_depth', None)
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            n_estimators = self.kwargs.get('n_estimators', 100)
            learning_rate = self.kwargs.get('learning_rate', 0.1)
            max_depth = self.kwargs.get('max_depth', 3)
            
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        elif self.model_type == 'logistic_regression':
            C = self.kwargs.get('C', 1.0)
            
            self.model = LogisticRegression(
                C=C,
                random_state=42,
                max_iter=1000
            )
        
        elif self.model_type == 'svm':
            C = self.kwargs.get('C', 1.0)
            kernel = self.kwargs.get('kernel', 'rbf')
            
            self.model = SVC(
                C=C,
                kernel=kernel,
                probability=True,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_pca: bool = False,
        n_components: Optional[int] = None
    ):
        """
        Train the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        use_pca : bool, default=False
            Whether to use PCA for dimensionality reduction
        n_components : Optional[int], default=None
            Number of PCA components
            
        Returns
        -------
        ChangeDetectionModel
            Trained model
        """
        # Create model if not already created
        if self.model is None:
            self._create_model()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if requested
        if use_pca:
            if n_components is None:
                n_components = min(X.shape[0], X.shape[1])
            
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        # Predict probabilities
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns
        -------
        Dict[str, Any]
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Calculate ROC AUC if possible
        try:
            y_proba = self.predict_proba(X)
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
        except:
            pass
        
        return metrics
    
    def save(self, path: str):
        """
        Save the model.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'model_type': self.model_type,
            'kwargs': self.kwargs
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ChangeDetectionModel':
        """
        Load a model.
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        Returns
        -------
        ChangeDetectionModel
            Loaded model
        """
        # Load model and metadata
        data = joblib.load(path)
        
        # Create model instance
        model_instance = cls(model_type=data['model_type'], **data['kwargs'])
        
        # Set model and metadata
        model_instance.model = data['model']
        model_instance.scaler = data['scaler']
        model_instance.pca = data['pca']
        
        return model_instance


class NeuralNetworkModel(nn.Module):
    """
    Neural network model for change detection.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64],
        output_size: int = 2,
        dropout_rate: float = 0.5
    ):
        """
        Initialize a NeuralNetworkModel.
        
        Parameters
        ----------
        input_size : int
            Size of input features
        hidden_sizes : List[int], default=[128, 64]
            Sizes of hidden layers
        output_size : int, default=2
            Size of output layer
        dropout_rate : float, default=0.5
            Dropout rate
        """
        super().__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.model(x)


class PyTorchChangeDetectionModel:
    """
    PyTorch-based change detection model.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 64],
        output_size: int = 2,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001
    ):
        """
        Initialize a PyTorchChangeDetectionModel.
        
        Parameters
        ----------
        input_size : int
            Size of input features
        hidden_sizes : List[int], default=[128, 64]
            Sizes of hidden layers
        output_size : int, default=2
            Size of output layer
        dropout_rate : float, default=0.5
            Dropout rate
        learning_rate : float, default=0.001
            Learning rate
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_model(self):
        """
        Create the underlying model.
        """
        self.model = NeuralNetworkModel(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        batch_size : int, default=32
            Batch size
        epochs : int, default=10
            Number of epochs
        validation_split : float, default=0.2
            Fraction of data to use for validation
        verbose : bool, default=True
            Whether to print progress
            
        Returns
        -------
        PyTorchChangeDetectionModel
            Trained model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        
        # Create model if not already created
        if self.model is None:
            self._create_model()
        
        # Create datasets
        train_dataset = ChangeDetectionDataset(X_train, y_train)
        val_dataset = ChangeDetectionDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train model
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                train_loss += loss.item() * features.size(0)
                _, predictions = torch.max(outputs, 1)
                train_correct += (predictions == labels).sum().item()
            
            # Calculate epoch metrics
            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            
            # Evaluate on validation set
            val_loss, val_acc = self._evaluate_epoch(val_loader)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return self
    
    def _evaluate_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model for one epoch.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader
            
        Returns
        -------
        Tuple[float, float]
            Loss and accuracy
        """
        self.model.eval()
        
        loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                batch_loss = self.criterion(outputs, labels)
                
                # Calculate metrics
                loss += batch_loss.item() * features.size(0)
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()
        
        # Calculate epoch metrics
        loss /= len(data_loader.dataset)
        accuracy = correct / len(data_loader.dataset)
        
        self.model.train()
        
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns
        -------
        Dict[str, Any]
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Calculate ROC AUC if possible
        try:
            y_proba = self.predict_proba(X)
            metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
        except:
            pass
        
        return metrics
    
    def save(self, path: str):
        """
        Save the model.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'scaler': self.scaler
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PyTorchChangeDetectionModel':
        """
        Load a model.
        
        Parameters
        ----------
        path : str
            Path to the saved model
            
        Returns
        -------
        PyTorchChangeDetectionModel
            Loaded model
        """
        # Load model and metadata
        checkpoint = torch.load(path)
        
        # Create model instance
        model_instance = cls(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            output_size=checkpoint['output_size'],
            dropout_rate=checkpoint['dropout_rate'],
            learning_rate=checkpoint['learning_rate']
        )
        
        # Create model
        model_instance._create_model()
        
        # Load model state
        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_instance.scaler = checkpoint['scaler']
        
        return model_instance
