import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        hidden1=16
        hidden2=32
        hidden3=64
        hidden4=128
        hidden5=256
        linear2=1024
        linear3=512
        self.model = nn.Sequential(
            # assume 3x224x224 image tensor
            # produces hidden1 feature maps 16x224x224
            nn.Conv2d(in_channels=3, out_channels=hidden1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   #16x112x112
            
            nn.Conv2d(hidden1, hidden2, 3, padding=1),  # -> 32x112x112
            nn.ReLU(),
            nn.BatchNorm2d(hidden2),
            nn.MaxPool2d(2, 2),  #32x56x56
            
            nn.Conv2d(hidden2, hidden3, 3, padding=1),  # -> 64x56x56
            nn.BatchNorm2d(hidden3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x28x28
            nn.Dropout2d(p=dropout),
            
            nn.Conv2d(hidden3, hidden4, 3, padding=1),  # -> 128x28x28
            nn.ReLU(),
            nn.BatchNorm2d(hidden4),
            nn.MaxPool2d(2, 2),  # -> 128x14x14
            nn.Dropout2d(p=dropout),
            
            nn.Conv2d(hidden4, hidden5, 3, padding=1),  # -> 128x28x28
            nn.ReLU(),
            nn.BatchNorm2d(hidden5),
            nn.MaxPool2d(2, 2),  # -> 256x7x7
            nn.Dropout2d(p=dropout),

            nn.Flatten(),  # -> 1x256x7x7
            
            nn.Linear(hidden5 * 7 * 7, linear2),                   
            nn.BatchNorm1d(linear2),
            nn.ReLU(),
            
            nn.Linear(linear2, linear3),            
            nn.BatchNorm1d(linear3),
            nn.ReLU(),
            nn.Dropout(p=dropout),           

            nn.Linear(linear3, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
