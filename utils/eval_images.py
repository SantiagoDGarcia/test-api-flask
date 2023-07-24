import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from typing import List


class Predict:
    def __init__(self, list_masks: np.ndarray, type_analisis: str) -> None:
        self.list_masks = list_masks  # List of images to make predictions for
        self.classes = [
            "B",
            "M",
        ]  # "B" and "M" represent two different classes for prediction
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Use GPU if available
        self.type_analysis = type_analisis  # Type of analysis to perform

    def transform_numpy_to_tensor(self, list_masks: np.ndarray) -> List[torch.Tensor]:
        # Define image transformations
        transform = transforms.Compose(
            [
                transforms.Resize([128, 128]),
                transforms.Grayscale(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # Apply transformations to each image in the list
        new_x = list(map(lambda x: x.astype("float"), list_masks))
        new_x = list(map(lambda x: np.expand_dims(x, axis=-1), new_x))
        new_x = list(map(lambda x: np.transpose(x, (2, 0, 1)), new_x))
        new_x = list(map(lambda x: torch.from_numpy(x), new_x))
        new_x = list(map(lambda x: transform(x), new_x))
        new_x = list(map(lambda x: (x - x.min()) / (x.max() - x.min()), new_x))

        return new_x

    def make_predictions(
        self,
        model_net: torch.nn.modules.container.Sequential,
        image: torch.Tensor,
        model_dir: str,
    ) -> str:
        try:
            # Load the model and set it to evaluation mode
            net_test = model_net.double()
            net_test.eval()
            checkpoint = torch.load(model_dir)
            net_test.load_state_dict(checkpoint["classifier_state"])

            # Make a prediction for the given image
            with torch.no_grad():
                outputs = model_net(image.unsqueeze(0)).double()
                _, predictions = torch.max(outputs, 1)
                return self.classes[int(predictions)]

        except Exception as error:
            print("Error: ", error)

    def get_predictions(self) -> List[str]:
        # Define the path to the model checkpoint based on the type of analysis
        if self.type_analysis == "ultrasound":
            model_dir = "models/chckp1.pth"
        else:
            model_dir = "models/chckp1.pth"

        # Define a function to create a convolutional layer with the given number of input and output channels
        layer_conv = lambda x, y: nn.Conv2d(
            in_channels=x, out_channels=y, kernel_size=3, padding=1
        )

        # Load a pre-trained ResNet18 model and remove the first and last layers
        model_resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        model_aux = nn.Sequential(*list(model_resnet.children())[1:-1])

        # Define a new model using the pre-trained ResNet18 layers and additional layers
        model_tl = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            model_aux,
            layer_conv(512, 1024),
            nn.BatchNorm2d(1024),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=10, bias=True),
        )

        # Make predictions for each image in the list
        list_predictions: List[str] = []
        for image in self.transform_numpy_to_tensor(self.list_masks):
            list_predictions.append(self.make_predictions(model_tl, image, model_dir))

        return list_predictions
