import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, 
                 features_dim: int = 512,
                 hidden_channel_size: int = [16,32,64], num_layers: int = 3,
                 linear_num_layers = 2,  linear_layer_size = [64, 64]) -> None:
        super().__init__(observation_space, features_dim)

        self._observation_space = observation_space
        self.num_layers = num_layers
        self.num_linear_layers = linear_num_layers
        self.linear_layer_size = linear_layer_size
        self.hidden_channel_size = hidden_channel_size
        self.sample_img = observation_space.sample()[None]

        self.build_image_component()
        self.build_linear_component()

        assert len(hidden_channel_size) == num_layers, "hidden_channel_size should be a list of length num_layers"

    
    def build_image_component(self):
        """
        Builds the image processing component of the neural network using Convolutional Neural Networks (CNN).
        This method constructs a sequential CNN model with the specified number of layers and hidden channel sizes.
        It initializes the CNN layers, including input, hidden, and output layers, and computes the output shape
        by performing a forward pass with a sample observation.
        Attributes:
            n_input_channels (int): Number of input channels from the observation space.
            layers (list): List to hold the CNN layers.
            cnn (nn.Sequential): The constructed CNN model.
            cnn_out_shape (int): The flattened output shape of the CNN.
        Raises:
            AttributeError: If `self._observation_space`, `self.hidden_channel_size`, or `self.num_layers` are not defined.
        """


        n_input_channels = self._observation_space.shape[0]
        
        # build CNN layers
        layers = []
        # input layer
        layers.append(nn.Conv2d(n_input_channels, self.hidden_channel_size[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(1, self.num_layers):
            layers.append(nn.Conv2d(self.hidden_channel_size[i-1], self.hidden_channel_size[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*layers)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(self.sample_img).float()).shape[1]

        self.cnn_out_shape = n_flatten

    def build_linear_component(self):
        """
        Builds a linear component consisting of a sequence of linear layers followed by ReLU activations.
        The linear component is constructed as follows:
        - An initial linear layer that takes the input dimension (cnn_out_shape + 1 for direction feature) 
          and maps it to the specified linear layer size.
        - A series of hidden linear layers, each followed by a ReLU activation, that map the linear layer size 
          to itself.
        - A final linear layer that maps the linear layer size to the specified features dimension.
        The constructed linear component is stored in the `self.linear` attribute as an `nn.Sequential` module.
        """

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        linear_layers = []

        # input dim
        _input_dim = self.cnn_out_shape
        
        # input linear layer
        linear_layers.append(nn.Linear(_input_dim, self.linear_layer_size[0]))
        linear_layers.append(nn.ReLU())
        # hidden linear layers
        for i in range(1, self.num_linear_layers):
            linear_layers.append(nn.Linear(self.linear_layer_size[i-1], self.linear_layer_size[i]))
            linear_layers.append(nn.ReLU())
        # output linear layer
        linear_layers.append(nn.Linear(self.linear_layer_size[-1], self.features_dim))
        linear_layers.append(nn.ReLU())
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        return self.linear(self.cnn(observations))

class MultiInputFE(BaseFeaturesExtractor):
    """
    Use dict observation to handle mission and 
    image inputs separately.
    """

    def __init__(self, observation_space: gym.Space, 
                 features_dim: int = 128,
                 hidden_channel_size: int = [16,32,64], num_layers: int = 3):
        super().__init__(observation_space, features_dim)
        self.observation_space = observation_space
        
        # image arch hyperparams
        self.hidden_channel_size = hidden_channel_size
        self.num_layers = num_layers
        
        self.IMG_SIZE = 7 * 7 * 3
        self.DIRECTION_SPACE_LEN = 4

        self.build_image_component(n_input_channels=7)

    def build_image_component(self, n_input_channels):
        """
        Builds the image processing component of the neural network using Convolutional Neural Networks (CNN).
        This method constructs a sequential CNN model with the specified number of layers and hidden channel sizes.
        It initializes the CNN layers, including input, hidden, and output layers, and computes the output shape
        by performing a forward pass with a sample observation.
        Attributes:
            n_input_channels (int): Number of input channels from the observation space.
            layers (list): List to hold the CNN layers.
            cnn (nn.Sequential): The constructed CNN model.
            cnn_out_shape (int): The flattened output shape of the CNN.
        Raises:
            AttributeError: If `self._observation_space`, `self.hidden_channel_size`, or `self.num_layers` are not defined.
        """
        
        # build CNN layers
        layers = []
        # input layer
        layers.append(nn.Conv2d(n_input_channels, self.hidden_channel_size[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(1, self.num_layers):
            layers.append(nn.Conv2d(self.hidden_channel_size[i-1], self.hidden_channel_size[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Flatten())
        # convert to features dim through a linear layer
        flatten_output_shape = 2688
        layers.append(nn.Linear(flatten_output_shape, flatten_output_shape // 4 )) # we leave one placeholder for the OHE direction
        layers.append(nn.ReLU())
        layers.append(nn.Linear(flatten_output_shape // 4, self.features_dim - self.DIRECTION_SPACE_LEN)) # we leave one placeholder for the OHE direction
        layers.append(nn.ReLU())

        self.image_comp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        """

        img = observations['image']
        img_features = self.image_comp(img)

        # get direction
        direction = observations['direction']
        direction_tensor = torch.zeros((direction.shape[0], 1))
        direction_tensor[direction == 0] = 1

        return torch.cat([img_features, direction_tensor], dim=1)
    
class FlatObsFeatureExtractor(BaseFeaturesExtractor):
    """
    Buid a feature extractor for the flat observation space.
    This takes a (2385, ) numpy array as an input and returns a tensor
    of the dimension of the action space.

    The numpy array is split in the image and mission encoding.
    """

    def __init__(self, observation_space: gym.Space, 
                 features_dim: int = 128,
                 linear_layer_size: list = [256,256,128], linear_num_layers: int = 3):
        super().__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.INPUT_SIZE = 2835

        # build MLP
        layers = []
        layers.append(nn.Linear(self.INPUT_SIZE, linear_layer_size[0]))
        layers.append(nn.ReLU())

        for i in range(1, linear_num_layers):
            layers.append(nn.Linear(linear_layer_size[i-1], linear_layer_size[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(linear_layer_size[-1], features_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        """
        return self.mlp(observations)
    
class ImageFeatureExtractor(BaseFeaturesExtractor):
    """
    Create a simple CNN to extract features from the image.
    Receives an image of shape (7,7,3) and returns a tensor of the features dimension.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, 
                 **kwargs) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim*2),
                                    nn.ReLU(),
                                    nn.Linear(features_dim*2, features_dim),
                                    nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def get(name):
    if name == "MinigridFeaturesExtractor":
        return MinigridFeaturesExtractor
    elif name == "MultiInputFE":
        return MultiInputFE
    elif name == "FlatObsFeatureExtractor":
        return FlatObsFeatureExtractor
    elif name == "ImageFeatureExtractor":
        return ImageFeatureExtractor
    else:
        raise ValueError(f"Unknown feature extractor: {name}")
    
def validate_feature_extractor_args(config):
    """
    Validate that all the keys in the configuration dict correspond
    to valid feature extractor arguments.
    """
    valid_keys = ["observation_space", "features_dim", "hidden_channel_size", "num_layers", "linear_num_layers", "linear_layer_size"]
    for key in config.keys():
        if key not in valid_keys:
            raise ValueError(f"Invalid feature extractor argument: {key}")
    return True