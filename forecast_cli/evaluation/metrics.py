import torch

def weighted_absolute_percentage_error(
    y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Calculates the Weighted Absolute Percentage Error (WAPE).

    WAPE is defined as sum(abs(y_true - y_pred)) / sum(abs(y_true)).
    It's also known as MAD/Mean ratio (Mean Absolute Deviation / Mean).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: A small value to add to the denominator to avoid division by zero.

    Returns:
        The WAPE score as a tensor.
    """
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("Inputs y_true and y_pred must be torch.Tensor")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match.")

    numerator = torch.sum(torch.abs(y_true - y_pred))
    denominator = torch.sum(torch.abs(y_true)) + epsilon
    
    wape_score = numerator / denominator
    return wape_score

def smape(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE).
    Formula: mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred) + epsilon)) * 100

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small constant to avoid division by zero.

    Returns:
        The sMAPE score as a tensor (percentage).
    """
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("Inputs y_true and y_pred must be torch.Tensor")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match.")

    numerator = 2 * torch.abs(y_pred - y_true)
    denominator = torch.abs(y_true) + torch.abs(y_pred) + epsilon
    return torch.mean(numerator / denominator) * 100

def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
    """
    Calculates the Pinball Loss (Quantile Loss).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values for a specific quantile.
        quantile: The quantile level (e.g., 0.5 for median).

    Returns:
        The Pinball loss as a tensor.
    """
    if not (0 < quantile < 1):
        raise ValueError("Quantile must be between 0 and 1.")
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("Inputs y_true and y_pred must be torch.Tensor")
    if y_true.shape != y_pred.shape:
        # Allow y_pred to have an extra dimension if it's direct from model with target dim
        if y_true.unsqueeze(-1).shape == y_pred.shape: # y_true (B,S), y_pred (B,S,1)
             y_true = y_true.unsqueeze(-1)
        elif y_true.shape != y_pred.shape : # Still not matching
            raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match or be compatible.")

    error = y_true - y_pred
    loss = torch.mean(torch.max((quantile - 1) * error, quantile * error))
    return loss

def hierarchical_consistency_metric(
    daily_true_or_pred: torch.Tensor, 
    hourly_aggregated_pred: torch.Tensor,
    aggregation_factor: int = 24 # Typically 24 hours in a day
) -> torch.Tensor:
    """
    Calculates a hierarchical consistency metric (MAE) between daily values 
    and aggregated hourly predictions.
    Assumes hourly_aggregated_pred is ALREADY aggregated (e.g., summed or averaged from hourly).
    The roadmap HCLoss was MAE(sum_hourly - daily).

    Args:
        daily_true_or_pred: Daily ground truth or predictions. Shape (batch, num_days, ...).
        hourly_aggregated_pred: Hourly predictions aggregated to daily frequency. 
                                Shape should match daily_true_or_pred.
        aggregation_factor: The factor used for aggregation (e.g. 24 for hours in day).
                            Not used in current MAE calc if pre-aggregated, but good for context.

    Returns:
        The MAE between daily and aggregated hourly predictions.
    """
    if not isinstance(daily_true_or_pred, torch.Tensor) or not isinstance(hourly_aggregated_pred, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if daily_true_or_pred.shape != hourly_aggregated_pred.shape:
        raise ValueError(
            f"Shapes must match. Got daily: {daily_true_or_pred.shape}, hourly_aggregated: {hourly_aggregated_pred.shape}"
        )
    
    # Using L1 loss (MAE) as per roadmap for HCLoss consistency
    return torch.nn.functional.l1_loss(daily_true_or_pred, hourly_aggregated_pred)

def coverage_p10_p90(
    y_true: torch.Tensor, 
    y_pred_p10: torch.Tensor, 
    y_pred_p90: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the coverage of the P10-P90 prediction interval.
    This is the proportion of true values that fall between the 10th and 90th percentile predictions.

    Args:
        y_true: Ground truth values.
        y_pred_p10: Predicted values for the 10th percentile.
        y_pred_p90: Predicted values for the 90th percentile.

    Returns:
        The coverage rate (proportion) as a tensor.
    """
    if not all(isinstance(t, torch.Tensor) for t in [y_true, y_pred_p10, y_pred_p90]):
        raise TypeError("All inputs must be torch.Tensor")
    if not (y_true.shape == y_pred_p10.shape == y_pred_p90.shape):
        # Allowunsqueeze for target dim as in pinball loss
        if y_true.unsqueeze(-1).shape == y_pred_p10.shape == y_pred_p90.shape:
            y_true = y_true.unsqueeze(-1)
        else:
            raise ValueError(
                f"Shapes of y_true ({y_true.shape}), y_pred_p10 ({y_pred_p10.shape}), "
                f"and y_pred_p90 ({y_pred_p90.shape}) must match or be compatible."
            )
    if torch.any(y_pred_p10 > y_pred_p90):
        print("Warning: In some instances, p10 prediction is greater than p90 prediction.")

    within_interval = (y_true >= y_pred_p10) & (y_true <= y_pred_p90)
    coverage_rate = torch.mean(within_interval.float())
    return coverage_rate

# Placeholder for other metrics mentioned in the roadmap (sMAPE, pinball, etc.)
# def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
#     pass

# def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantile: float) -> torch.Tensor:
#     pass

# def hierarchical_consistency(daily_pred: torch.Tensor, hourly_aggregated_pred: torch.Tensor) -> torch.Tensor:
#     pass

# def coverage_p90_p10(y_true: torch.Tensor, p10_pred: torch.Tensor, p90_pred: torch.Tensor) -> torch.Tensor:
# pass
