import torchmetrics
from segmentation_models_pytorch.utils import base
from torchmetrics import JaccardIndex


class PerClassIou(base.Metric):
    def __init__(
        self,
        base_metric: torchmetrics.Metric,
        class_index: int,
        class_name: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Clone the metric to ensure independent state if the base_metric is used elsewhere
        self.metric = base_metric.clone()
        self.class_index = class_index
        self.class_name = class_name if class_name is not None else f"Class_{class_index}"
        self._name = f"{self.class_name}"

    def forward(self, y_pr, y_gt):
        # Convert model outputs (logits) to predicted class indices [B, H, W]
        # For softmax: preds = y_pr.argmax(dim=1)  # [B, 1, H, W] -> Shape: [B, H, W]
        # For sigmoid: preds = (y_pr > 0.5).float()  # [B, 1, H, W] -> Shape: [B, H, W]
        preds = (y_pr[:, 0, :, :] >= 0.5).float()

        # Convert one-hot masks to integer class indices [B, H, W]
        # For softmax: preds = y_pr.argmax(dim=1)  # [B, 1, H, W] -> Shape: [B, H, W]
        # For sigmoid: preds = (y_pr > 0.5).float()  # [B, 1, H, W] -> Shape: [B, H, W]
        target = (y_gt[:, 0, :, :] >= 0.5).float()

        # Call the base metric with the specified class index
        return self.metric(preds, target)[self.class_index]


class IoU(base.Metric):
    def __init__(
        self,
        device,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Clone the metric to ensure independent state if the base_metric is used elsewhere
        self.metric = JaccardIndex(task="multiclass", num_classes=2, average="weighted").to(
            device,
        )
        self._name = "weighted_iou"

    def forward(self, y_pr, y_gt):
        # Convert model outputs (logits) to predicted class indices [B, H, W]
        # For softmax: preds = y_pr.argmax(dim=1)  # [B, 1, H, W] -> Shape: [B, H, W]
        # For sigmoid: preds = (y_pr > 0.5).float()  # [B, 1, H, W] -> Shape: [B, H, W]
        preds = (y_pr[:, 0, :, :] >= 0.5).float()

        # Convert one-hot masks to integer class indices [B, H, W]
        # For softmax: preds = y_pr.argmax(dim=1)  # [B, 1, H, W] -> Shape: [B, H, W]
        # For sigmoid: preds = (y_pr > 0.5).float()  # [B, 1, H, W] -> Shape: [B, H, W]
        target = (y_gt[:, 0, :, :] >= 0.5).float()

        # Call the base metric with the specified class index
        return self.metric(preds, target)
