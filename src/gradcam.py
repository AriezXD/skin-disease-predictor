import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model

        # Last convolutional layer of EfficientNet-B0
        self.target_layer = model.features[-1]

        self.activations = None
        self.gradients = None

        # Forward hook
        self.target_layer.register_forward_hook(
            self.save_activation
        )

        # Backward hook
        self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_idx):

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor)

        # Select score for the predicted class
        score = output[:, class_idx]

        # Backward pass
        score.backward()

        # Feature maps
        activations = self.activations

        # Gradients
        gradients = self.gradients

        # Average gradients across spatial dimensions
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (weights * activations).sum(dim=1)

        # Keep only positive influence
        cam = F.relu(cam)

        # Remove batch dimension
        cam = cam[0].detach().cpu().numpy()

        # Normalize between 0 and 1
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam

    def overlay(self, image, cam, alpha=0.4):

        # image should be RGB numpy array
        height, width = image.shape[:2]

        # Resize CAM to image size
        cam = cv2.resize(
            cam,
            (width, height)
        )

        # Convert to 0-255
        heatmap = np.uint8(255 * cam)

        # Apply color map
        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # OpenCV uses BGR, convert to RGB
        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Blend original image + heatmap
        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return overlay