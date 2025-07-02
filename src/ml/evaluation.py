"""
Evaluation utilities for Street Level Change Detection models.

This module provides functions for evaluating machine learning models
for detecting changes in street-level imagery.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, learning_curve,
    validation_curve
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Parameters
    ----------
    model : object
        Model with predict and predict_proba methods
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    class_names : Optional[List[str]], default=None
        Names of classes
        
    Returns
    -------
    Dict[str, Any]
        Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None
    
    # Set default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names)
    }
    
    # Calculate ROC AUC if probabilities are available
    if y_proba is not None and y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics['average_precision'] = average_precision_score(y_test, y_proba[:, 1])
        metrics['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    normalize: bool = False
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : List[str]
        Names of classes
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    cmap : str, default='Blues'
        Colormap
    normalize : bool, default=False
        Whether to normalize the confusion matrix
        
    Returns
    -------
    plt.Figure
        Figure with confusion matrix plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a ROC curve.
    
    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    roc_auc : float
        Area under the ROC curve
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure with ROC curve plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        fpr,
        tpr,
        lw=2,
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: float,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a precision-recall curve.
    
    Parameters
    ----------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    average_precision : float
        Average precision score
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure with precision-recall curve plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        recall,
        precision,
        lw=2,
        label=f'Precision-Recall curve (AP = {average_precision:.2f})'
    )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    
    return fig


def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    scoring: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a learning curve.
    
    Parameters
    ----------
    estimator : object
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int, default=5
        Number of cross-validation folds
    train_sizes : np.ndarray, default=np.linspace(0.1, 1.0, 5)
        Training set sizes
    scoring : str, default='accuracy'
        Scoring metric
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure with learning curve plot
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color='blue'
    )
    
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color='orange'
    )
    
    ax.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color='blue',
        label='Training score'
    )
    
    ax.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color='orange',
        label='Cross-validation score'
    )
    
    ax.set_xlabel('Training examples')
    ax.set_ylabel(f'Score ({scoring})')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig


def plot_validation_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    param_name: str,
    param_range: List[Any],
    cv: int = 5,
    scoring: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a validation curve.
    
    Parameters
    ----------
    estimator : object
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    param_name : str
        Name of the parameter to vary
    param_range : List[Any]
        Range of parameter values
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='accuracy'
        Scoring metric
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
        
    Returns
    -------
    plt.Figure
        Figure with validation curve plot
    """
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color='blue'
    )
    
    ax.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color='orange'
    )
    
    ax.plot(
        param_range,
        train_scores_mean,
        'o-',
        color='blue',
        label='Training score'
    )
    
    ax.plot(
        param_range,
        test_scores_mean,
        'o-',
        color='orange',
        label='Cross-validation score'
    )
    
    ax.set_xlabel(param_name)
    ax.set_ylabel(f'Score ({scoring})')
    ax.set_title('Validation Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Union[str, List[str]] = 'accuracy',
    return_estimator: bool = False
) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : object
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int, default=5
        Number of cross-validation folds
    scoring : Union[str, List[str]], default='accuracy'
        Scoring metric(s)
    return_estimator : bool, default=False
        Whether to return the trained estimators
        
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_estimator=return_estimator,
        n_jobs=-1
    )
    
    # Convert numpy arrays to lists for JSON serialization
    results = {}
    
    for key, value in cv_results.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.float64:
                results[key] = value.tolist()
            else:
                results[key] = [str(v) for v in value]
        else:
            results[key] = value
    
    return results


def generate_evaluation_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report for a model.
    
    Parameters
    ----------
    model : object
        Trained model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    class_names : Optional[List[str]], default=None
        Names of classes
    output_dir : Optional[str], default=None
        Directory to save plots
        
    Returns
    -------
    Dict[str, Any]
        Evaluation report
    """
    # Set default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Create output directory if specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model on test data
    test_metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # Evaluate model on training data
    train_metrics = evaluate_model(model, X_train, y_train, class_names)
    
    # Create report
    report = {
        'test_metrics': test_metrics,
        'train_metrics': train_metrics
    }
    
    # Generate plots if output directory is specified
    if output_dir is not None:
        # Confusion matrix
        cm = np.array(test_metrics['confusion_matrix'])
        cm_fig = plot_confusion_matrix(cm, class_names)
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        cm_fig.savefig(cm_path)
        plt.close(cm_fig)
        
        # ROC curve if available
        if 'roc_curve' in test_metrics:
            fpr = np.array(test_metrics['roc_curve']['fpr'])
            tpr = np.array(test_metrics['roc_curve']['tpr'])
            roc_auc = test_metrics['roc_auc']
            
            roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            roc_fig.savefig(roc_path)
            plt.close(roc_fig)
        
        # Precision-recall curve if available
        if 'pr_curve' in test_metrics:
            precision = np.array(test_metrics['pr_curve']['precision'])
            recall = np.array(test_metrics['pr_curve']['recall'])
            avg_precision = test_metrics['average_precision']
            
            pr_fig = plot_precision_recall_curve(precision, recall, avg_precision)
            pr_path = os.path.join(output_dir, 'pr_curve.png')
            pr_fig.savefig(pr_path)
            plt.close(pr_fig)
        
        # Add plot paths to report
        report['plot_paths'] = {
            'confusion_matrix': cm_path
        }
        
        if 'roc_curve' in test_metrics:
            report['plot_paths']['roc_curve'] = roc_path
        
        if 'pr_curve' in test_metrics:
            report['plot_paths']['pr_curve'] = pr_path
    
    return report


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']
) -> pd.DataFrame:
    """
    Compare multiple models on test data.
    
    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary mapping model names to model objects
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    class_names : Optional[List[str]], default=None
        Names of classes
    metrics : List[str], default=['accuracy', 'precision', 'recall', 'f1']
        Metrics to compare
        
    Returns
    -------
    pd.DataFrame
        DataFrame with model comparison
    """
    # Set default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # Evaluate each model
    results = {}
    
    for name, model in models.items():
        # Evaluate model
        metrics_dict = evaluate_model(model, X_test, y_test, class_names)
        
        # Extract requested metrics
        results[name] = {metric: metrics_dict[metric] for metric in metrics if metric in metrics_dict}
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    return df
