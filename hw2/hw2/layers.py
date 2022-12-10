import abc
import torch


class Layer(abc.ABC):
    """
    A Layer is some computation element in a network architecture which
    supports automatic differentiation using forward and backward functions.
    """

    def __init__(self):
        # Store intermediate values needed to compute gradients in this hash
        self.grad_cache = {}
        self.training_mode = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Computes the forward pass of the layer.
        :param args: The computation arguments (implementation specific).
        :return: The result of the computation.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Computes the backward pass of the layer, i.e. the gradient
        calculation of the final network output with respect to each of the
        parameters of the forward function.
        :param dout: The gradient of the network with respect to the
        output of this layer.
        :return: A tuple with the same number of elements as the parameters of
        the forward function. Each element will be the gradient of the
        network output with respect to that parameter.
        """
        pass

    @abc.abstractmethod
    def params(self):
        """
        :return: Layer's trainable parameters and their gradients as a list
        of tuples, each tuple containing a tensor and it's corresponding
        gradient tensor.
        """
        pass

    def train(self, training_mode=True):
        """
        Changes the mode of this layer between training and evaluation (test)
        mode. Some layers have different behaviour depending on mode.
        :param training_mode: True: set the model in training mode. False: set
        evaluation mode.
        """
        self.training_mode = training_mode

    def __repr__(self):
        return self.__class__.__name__


class LeakyReLU(Layer):
    """
    Leaky version of Rectified linear unit.
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        if not (0 <= alpha < 1):
            raise ValueError("Invalid value of alpha")
        self.alpha = alpha

    def forward(self, x, **kw):
        """
        Computes max(alpha*x, x) for some 0<= alpha < 1.
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: ReLU of each sample in x.
        """

        # TODO: Implement the LeakyReLU operation.
        # ====== YOUR CODE: ======
        out = torch.max(self.alpha*x, x)
        # ========================

        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """
        x = self.grad_cache["x"]

        # relu = max(alpha *x, x)
        # drely/dx = {x>0 --> 1 , x<=0 --> alpha}
        # TODO: Implement gradient w.r.t. the input x

        # ====== YOUR CODE: ======
        drelu_dx_pos = (x > 0) * 1
        drelu_dx_neg = (x < 0) * self.alpha
        dx = (drelu_dx_pos + drelu_dx_neg)*dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"LeakyReLU({self.alpha=})"


class ReLU(LeakyReLU):
    """
    Rectified linear unit.
    """

    def __init__(self):
        # ====== YOUR CODE: ======
        LeakyReLU.__init__(self, 0)
        # ========================

    def __repr__(self):
        return "ReLU"


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes s(x) = 1/(1+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # TODO: Implement the Sigmoid function.
        #  Save whatever you need into grad_cache.
        # ====== YOUR CODE: ======
        out = 1 / (1 + torch.exp(-x))
        self.grad_cache["x"] = x
        # ========================

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        x = self.grad_cache["x"]  # restoring x
        #according to lecture2
        dx = (self.forward(x)*(1-self.forward(x)))*dout
        # ========================

        return dx

    def params(self):
        return []


class TanH(Layer):
    """
    Hyperbolic tangent activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, **kw):
        """
        Computes tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        :param x: Input tensor of shape (N,*) where N is the batch
        dimension, and * is any number of other dimensions.
        :return: Sigmoid of each sample in x.
        """

        # TODO: Implement the tanh function.
        #  Save whatever you need into grad_cache.
        # ====== YOUR CODE: ======
        exp_x = torch.exp(x)
        exp_neg_x =  torch.exp(-x)
        out = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        self.grad_cache["x"] = x
        # ========================

        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, *).
        :return: Gradient with respect to layer input, shape (N, *)
        """

        # TODO: Implement gradient w.r.t. the input x
        # ====== YOUR CODE: ======
        x = self.grad_cache["x"]
        x_forward = self.forward(x)
        dx = (1 - x_forward*x_forward)
        dx = dx * dout
        # ========================

        return dx

    def params(self):
        return []


class Linear(Layer):
    """
    Fully-connected linear layer.
    """

    def __init__(self, in_features, out_features, wstd=0.1):
        """
        :param in_features: Number of input features (Din)
        :param out_features: Number of output features (Dout)
        :param wstd: standard deviation of the initial weights matrix
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # TODO: Create the weight matrix (self.w) and bias vector (self.b).
        # Initialize the weights to zero-mean gaussian noise with a standard
        # deviation of `wstd`. Init bias to zero.
        # ====== YOUR CODE: ======

        # the solution in here based on https://vistalab-technion.github.io/cs236781/lecture_notes/lecture_03/
        
        self.b = torch.zeros(self.out_features) # Initialising bias to zero
        """
         Initilising the wights with zero-mean gaussian noise with a standard
         deviation of `wstd`
          a. the dimensions of W0 = out_features x in_features 

        """
        self.w = torch.normal(mean = 0, std = wstd, size=(out_features, in_features))

        # ========================

        # These will store the gradients
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def params(self):
        return [(self.w, self.dw), (self.b, self.db)]

    def forward(self, x, **kw):
        """
        Computes an affine transform, y = x W^T + b.
        :param x: Input tensor of shape (N,Din) where N is the batch
        dimension, and Din is the number of input features.
        :return: Affine transform of each sample in x.
        """

        # TODO: Compute the affine transform
        # ====== YOUR CODE: ======
        out = torch.matmul(x, self.w.t()) + self.b
        # ========================

        self.grad_cache["x"] = x
        return out

    def backward(self, dout):
        """
        :param dout: Gradient with respect to layer output, shape (N, Dout).
        :return: Gradient with respect to layer input, shape (N, Din)
        """
        x = self.grad_cache["x"]

        # TODO: Compute
        #   - dx, the gradient of the loss with respect to x
        #   - dw, the gradient of the loss with respect to w
        #   - db, the gradient of the loss with respect to b
        #  Note: You should ACCUMULATE gradients in dw and db.
        # ====== YOUR CODE: ======
        # dl_db = torch.ones(dout.shape[0])1
        # self.db += 
        dx = torch.matmul(dout, self.w)
        self.dw += torch.matmul(dout.T, x)
        self.db += torch.matmul(torch.ones(dout.shape[0]), dout)
        # ========================

        return dx

    def __repr__(self):
        return f"Linear({self.in_features=}, {self.out_features=})"


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Computes cross-entropy loss directly from class scores.
        Given class scores x, and a 1-hot encoding of the correct class yh,
        the cross entropy loss is defined as: -yh^T * log(softmax(x)).

        This implementation works directly with class scores (x) and labels
        (y), not softmax outputs or 1-hot encodings.

        :param x: Tensor of shape (N,D) where N is the batch
            dimension, and D is the number of features. Should contain class
            scores, NOT PROBABILITIES.
        :param y: Tensor of shape (N,) containing the ground truth label of
            each sample.
        :return: Cross entropy loss, as if we computed the softmax of the
            scores, encoded y as 1-hot and calculated cross-entropy by
            definition above. A scalar.
        """

        N = x.shape[0]

        # Shift input for numerical stability
        xmax, _ = torch.max(x, dim=1, keepdim=True)
        x = x - xmax

        # TODO: Compute the cross entropy loss using the last formula from the
        #  notebook (i.e. directly using the class scores).
        # ====== YOUR CODE: ======
        sum_ex = torch.sum(torch.exp(x), 1)
        log_sum_ex = torch.log(sum_ex) # segma(e^xk)
        x_y = -x[range(N), y].view(-1,1) # Calculating Xy
        loss = log_sum_ex.view(-1,1) + x_y
        loss = torch.mean(loss)

        # ========================

        self.grad_cache["x"] = x
        self.grad_cache["y"] = y
        return loss

    def backward(self, dout=1.0):
        """
        :param dout: Gradient with respect to layer output, a scalar which
            defaults to 1 since the output of forward is scalar.
        :return: Gradient with respect to layer input (only x), shape (N,D)
        """
        x = self.grad_cache["x"]
        y = self.grad_cache["y"]
        N = x.shape[0]

        # TODO: Calculate the gradient w.r.t. the input x.
        # ====== YOUR CODE: ======
        # the solution is based on : https://deepnotes.io/softmax-crossentropy ( derivation of Cross Entropy Loss with Softmax)
        sum_ex = torch.sum(torch.exp(x), 1).view(-1,1)
        softmax = torch.exp(x) / sum_ex.repeat(1, x.shape[1])

        grad = softmax
        grad[range(N),y] -= 1
        grad /= N # A.I Sarah( to understand why we need to divide by N I'm )
        dx = grad * dout
        # ========================

        return dx

    def params(self):
        return []


class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        Initializes a Dropout layer.
        :param p: Probability to drop an activation.
        """
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x, **kw):
        # TODO: Implement the dropout forward pass.
        #  Notice that contrary to previous layers, this layer behaves
        #  differently a according to the current training_mode (train/test).
        # ====== YOUR CODE: ======
        out = x
        self.grad_cache["x"] = x

        if self.training_mode:
            neuron_sel = (torch.Tensor(x.shape).uniform_(0, 1) > self.p).float()
            self.grad_cache["not_dropped"] = neuron_sel
            out = torch.mul(neuron_sel, x) * (1.0 / (1 - self.p))
        # ========================

        return out

    def backward(self, dout):
        # TODO: Implement the dropout backward pass.
        # ====== YOUR CODE: ======
        
        if self.training_mode:
            neuron_sel = self.grad_cache["not_dropped"]
            dout = torch.mul(neuron_sel, dout) * (1.0 / (1 - self.p))
        
        dx = dout
        # ========================

        return dx

    def params(self):
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Layer):
    """
    A Layer that passes input through a sequence of other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, **kw):
        out = None

        # TODO: Implement the forward pass by passing each layer's output
        #  as the input of the next.
        # ====== YOUR CODE: ======
        out = x
        for layer in self.layers:
            out = layer.forward(out, **kw)
        # ========================

        return out

    def backward(self, dout):
        din = None

        # TODO: Implement the backward pass.
        #  Each layer's input gradient should be the previous layer's output
        #  gradient. Behold the backpropagation algorithm in action!
        # ====== YOUR CODE: ======
        din = dout
        for layer in reversed(self.layers):
            din = layer.backward(din)
        # ========================

        return din

    def params(self):
        params = []

        # TODO: Return the parameter tuples from all layers.
        # ====== YOUR CODE: ======
        for layer in self.layers:
            params = params + layer.params()
        # ========================

        return params

    def train(self, training_mode=True):
        for layer in self.layers:
            layer.train(training_mode)

    def __repr__(self):
        res = "Sequential\n"
        for i, layer in enumerate(self.layers):
            res += f"\t[{i}] {layer}\n"
        return res

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]


class MLP(Layer):
    """
    A simple multilayer perceptron based on our custom Layers.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_features=(),
        activation="relu",
        dropout=0,
        **kw,
    ):
        super().__init__()
        """
        Create an MLP model Layer.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        layers = []

        x = in_features
        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        all_layers_features = hidden_features.copy()
        all_layers_features.append(num_classes)
        all_layers_features.insert(0, in_features)

        layers_len = len(all_layers_features)

        for layer in range(layers_len-1):
            layers.append(Linear(all_layers_features[layer], all_layers_features[layer+1]))
            if layer != (layers_len-2):
                if activation == "relu":
                    layers.append(ReLU())
                else:
                    layers.append(Sigmoid())
                if dropout > 0:
                    layers.append(Dropout(dropout))
                
        # ========================

        self.sequence = Sequential(*layers)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f"MLP, {self.sequence}"
