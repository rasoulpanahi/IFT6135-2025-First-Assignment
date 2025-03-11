import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # Weight matrix
        self.bias   = nn.Parameter(torch.randn(out_features))  # Bias vector

    def forward(self, input):
        return torch.matmul(input, self.weight.T) + self.bias  # Linear transformation


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)

        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)

    def _build_layers(self, input_size: int,
                        hidden_sizes: List[int],
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(Linear(in_size, h_size))
            in_size = h_size
        hidden_layers = nn.ModuleList(layers)
        output_layer  = Linear(in_size, num_classes)
        return hidden_layers, output_layer

    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        if activation == 'tanh':
            return torch.tanh(inputs)
        elif activation == 'relu':
            return torch.relu(inputs)
        elif activation == 'sigmoid':
            return torch.sigmoid(inputs)
        else:
            raise ValueError(f"Invalid activation function: {activation}")


    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors.
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.

        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        x = images.view(images.size(0), -1)  # Flatten the images
        for layer in self.hidden_layers:
            x = self.activation_fn(self.activation, layer(x))
        logits = self.output_layer(x)
        return logits

