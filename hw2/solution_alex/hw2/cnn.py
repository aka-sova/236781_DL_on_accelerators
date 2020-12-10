import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: list,
            pool_every: int,
            hidden_dims: list,
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======

        P = self.pool_every
        N = len(self.channels)
        M = self.hidden_dims


        def ceil(n):
            res = int(n)
            return res if res == n else res + 1

        # update the dimensions along while adding the layers
        curr_h = in_h
        curr_w = in_w

        # print(in_channels, in_h, in_w)

        conv_act_pool_num = ceil(N / P) - 1 if N % P != 0 else ceil(N / P)

        def add_activation_function(layers_in, activation_type, **activation_params):
            if activation_type == 'relu':
                layers_in.append(nn.ReLU(**activation_params))
            else:
                layers_in.append(nn.LeakyReLU(**activation_params))
            return layers_in

        def add_pool_function(layers_in, pooling_type, **pooling_params):
            if pooling_type == 'max':
                layers_in.append(nn.MaxPool2d(**pooling_params))
            else:
                layers_in.append(nn.AvgPool2d(**pooling_params))
            return layers_in

        def update_size_filter(input_size, dim, last_filter):
            # after filter, size changes:
            #   size_out = ((size_in +2*padding - (dilation * (kernel_size - 1)) -1 ) / stride) +1

            # dilation = 1 for all our cases, so:
            #   size_out = (size_in +2*padding - ( kernel_size-1 ) -1 ) / stride) + 1

            if type(last_filter) == torch.nn.modules.pooling.MaxPool2d:
                padding = last_filter.padding
                dilation = last_filter.dilation
                kernel_size = last_filter.kernel_size
                stride = last_filter.stride
            elif type(last_filter) == torch.nn.modules.pooling.AvgPool2d:
                padding = last_filter.padding
                dilation = 1
                kernel_size = last_filter.kernel_size
                stride = last_filter.stride
            else:
                padding = last_filter.padding[dim]
                dilation = last_filter.dilation[dim]
                kernel_size = last_filter.kernel_size[dim]
                stride = last_filter.stride[dim]

            return int(((input_size + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1)


        in_ch_list = [in_channels]
        conv_ch_list = in_ch_list + self.channels

        for i in range(conv_act_pool_num):

            # CONV -> ACT
            for j in range(P):
                layers.append(nn.Conv2d(conv_ch_list[i*P + j], conv_ch_list[i*P + (j+1)], **self.conv_params))
                curr_h = update_size_filter(curr_h, 0, layers[-1])
                curr_w = update_size_filter(curr_w, 1, layers[-1])

                layers = add_activation_function(layers, self.activation_type, **self.activation_params)


            # POOL
            layers = add_pool_function(layers, self.pooling_type, **self.pooling_params)
            curr_h = update_size_filter(curr_h, 0, layers[-1])
            curr_w = update_size_filter(curr_w, 1, layers[-1])


        # check if need conv layer without pool
        if N % P > 0:

            init_i = conv_act_pool_num * P

            for i in range(N % P):
                layers.append(nn.Conv2d(conv_ch_list[init_i + i], conv_ch_list[init_i + i + 1], **self.conv_params))
                curr_h = update_size_filter(curr_h, 0, layers[-1])
                curr_w = update_size_filter(curr_w, 1, layers[-1])
                # print(f"layer size {curr_h}")
                layers = add_activation_function(layers, self.activation_type, **self.activation_params)

        self.classified_input_size = int(curr_h), int(curr_w)

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        in_channels, in_h, in_w, = tuple(self.in_size)

        input_h, input_w = tuple(self.classified_input_size)

        M = self.hidden_dims

        def add_activation_function(layers_in, activation_type, **activation_params):
            if activation_type == 'relu':
                layers_in.append(nn.ReLU(**activation_params))
            else:
                layers_in.append(nn.LeakyReLU(**activation_params))
            return layers_in

        # add the FCNs
        last_cnn_out_c = self.channels[-1]

        # print(f"input params: w: {input_w}, h : {input_h}, c: {last_cnn_out_c}")
        last_cnn_params_n = input_w * input_h * last_cnn_out_c

        layers.append(nn.Linear(last_cnn_params_n, self.hidden_dims[0]))
        layers = add_activation_function(layers, self.activation_type, **self.activation_params)

        for i in range(len(M) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers = add_activation_function(layers, self.activation_type, **self.activation_params)

        # last layer - connect to the output features amount
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.shape[0], -1)
        classification = self.classifier(features)
        out = classification
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: list,
            kernel_sizes: list,
            batchnorm=False,
            dropout=0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======

        # main path
        layers = []

        # each convolution layer is followed by
        #   dropout (optional)
        #   batch normalization (optional)
        #   relu

        in_ch_list = [in_channels]
        conv_ch_list = in_ch_list + channels

        for i in range(len(conv_ch_list) - 1):

            # this should preserve the size

            padding = int((kernel_sizes[i] - 1) / 2)  # calculated
            stride = 1
            dilation = 1
            layers.append(nn.Conv2d(in_channels=conv_ch_list[i],
                                    out_channels=conv_ch_list[i + 1],
                                    kernel_size=kernel_sizes[i],
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True))

            if i < len(conv_ch_list) - 2:
                # for all layers except the last layer
                if dropout > 0:
                    layers.append(nn.Dropout2d(p=dropout))

                if batchnorm:
                    layers.append(nn.BatchNorm2d(num_features=conv_ch_list[i + 1]))


                if activation_type == 'relu':
                    layers.append(nn.ReLU(**activation_params))
                else:
                    layers.append(nn.LeakyReLU(**activation_params))

        self.main_path = nn.Sequential(*layers)

        # skip path layer

        skip_layers = []

        if in_channels == channels[-1]:

            # have to apply identity
            pass
            # skip_layers.append(nn.Identity())

        else:
            conv_identity_layer = nn.Conv2d(in_channels=in_channels,
                                            out_channels=channels[-1],
                                            kernel_size=1,
                                            padding=0,
                                            stride=1,
                                            dilation=1,
                                            bias=False)
            # set weights to 1 to make it identity
            # conv_identity_layer.weight = torch.nn.Parameter(torch.ones_like(conv_identity_layer.weight))



            skip_layers.append(conv_identity_layer)



        self.shortcut_path = nn.Sequential(*skip_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======

        # extract the kwargs


        def ceil(n):
            res = int(n)
            return res if res == n else res + 1

        P = self.pool_every
        N = len(self.channels)
        M = self.hidden_dims

        # update the dimensions along while adding the layers
        curr_h = in_h
        curr_w = in_w


        conv_act_pool_num = ceil(N / P) - 1 if N % P != 0 else ceil(N / P)
        in_ch_list = [in_channels]
        conv_ch_list = in_ch_list + self.channels

        # print(f"conv_ch_list = {conv_ch_list}")

        def add_pool_function(layers_in, pooling_type, **pooling_params):
            if pooling_type == 'max':
                layers_in.append(nn.MaxPool2d(**pooling_params))
            else:
                layers_in.append(nn.AvgPool2d(**pooling_params))
            return layers_in

        def update_size_filter(input_size, dim, last_filter):
            # after filter, size changes:
            #   size_out = ((size_in +2*padding - (dilation * (kernel_size - 1)) -1 ) / stride) +1

            # dilation = 1 for all our cases, so:
            #   size_out = (size_in +2*padding - ( kernel_size-1 ) -1 ) / stride) + 1

            if type(last_filter) == torch.nn.modules.pooling.MaxPool2d:
                padding = last_filter.padding
                dilation = last_filter.dilation
                kernel_size = last_filter.kernel_size
                stride = last_filter.stride
            elif type(last_filter) == torch.nn.modules.pooling.AvgPool2d:
                padding = last_filter.padding
                dilation = 1
                kernel_size = last_filter.kernel_size
                stride = last_filter.stride
            else:
                padding = last_filter.padding[dim]
                dilation = last_filter.dilation[dim]
                kernel_size = last_filter.kernel_size[dim]
                stride = last_filter.stride[dim]

            return int(((input_size + 2 * padding - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

        print("\nmain block")
        if conv_act_pool_num > 0:
            for i in range(conv_act_pool_num):
                init_channel = i * (P - 1)
                #
                # print(f"init_channel = {init_channel}")
                # print(f"in_channels={conv_ch_list[init_channel]}")
                # print(f"conv_ch_list[init_channel + 1: init_channel + P + 1] = {conv_ch_list[init_channel + 1: init_channel + P + 1]}")

                layers.append(ResidualBlock(in_channels=conv_ch_list[init_channel],
                                            channels=conv_ch_list[init_channel + 1: init_channel + P + 1],
                                            kernel_sizes=[3] * P,
                                            batchnorm=self.batchnorm,
                                            dropout=self.dropout,
                                            activation_type=self.activation_type,
                                            activation_params=self.activation_params))

                layers = add_pool_function(layers, self.pooling_type, **self.pooling_params)

                # only pool layer affects the size
                curr_h = update_size_filter(curr_h, 0, layers[-1])
                curr_w = update_size_filter(curr_w, 1, layers[-1])

        # check if need Res block without pool
        print("\nleft block")
        if N % P > 0:
            init_channel = (ceil(N / P)-1) * P
            # print(f"init_channel = {init_channel}")
            # print(f"in_channels={conv_ch_list[init_channel]}")
            # print(f"conv_ch_list[init_channel + 1:] = {conv_ch_list[init_channel + 1:]}")
            # without batchnorm and dropout
            layers.append(ResidualBlock(in_channels=conv_ch_list[init_channel],
                                        channels=conv_ch_list[init_channel + 1:],
                                        kernel_sizes=[3] * (N % P),
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

        self.classified_input_size = int(curr_h), int(curr_w)

        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()

    # ========================
