r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

used references: 
https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant - explanation of Jaccobian Calclations.


1.A. Given in_features = 1024, out_features = 512, and the layer is fully connected laye
 --> W shape is = 512 X 1024; also the shape of X is 64x1024
the shape of the Jacobian is: 64 x 1024 x 64 x 512

1.B. yes, element-ij in the jaccobian J_Yij depends only on row xi as we are deferentiating Y w.r.t on X;
therfore when calculating the derivations , when i != j -> J_Yji = 0, so for sure teh Jacobian is sparse§

1.C. No, we can calculate the downstream gratdient w.r.t. to the input X using 
the given  gradient of the output w.r.t. some downstream scalar loss  𝐿 ,  𝛿𝒀=∂𝐿∂𝒀 multiplyin by matrix W.


2.A. Shpae of W is 512x1024 and Y is 64x512, thurfore the shape of Jacobian is 64 x 1024 x 64 x 512
((((A.I Majd )))) please  review this
2.B. yes, it is sparsed with the same explanantion but in here w.r.t to wi instead of xi.

2.C. No, with the same explanation but n calculation using X instead of W.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

A.I Majd - please review the answer
Reference to be removed: https://www.quora.com/Is-Deep-Learning-possible-without-back-propagation

No, it is not required.
The back-propagation computes the gradient of the loss function 
with respect to the weights of the network for a single input–output,as part of the optimization procedure,
it was proved by researchers that it is so efficient. but it is not required, a different algorithm can be
used to calculate the gradients for the loss function with respect to the model parameters,
such as alternate algorithms that implement the chain rule

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1, #Following the concept to start with high learning rate and keep dividing by 2
        0.0125,
        0.002,
        0.00025,
        0.000001,
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = (
        0.001,
        0.00025,
    )
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. The graphs are as expected.
In with-droput graphs we are seeing lower accuracy in train-acc which its expected as the dropout must vanish
the over-fitting behaviour that appears in no-dropout graph which suffers from  generalization error 
- this is what the regularization dropout offers.

2. the model behaviour with dropout=0.4 is better than with dropout=0.8.
As the test loss is the minimum and train acc dropped to be lower than 100% while still better than when having dropout=0.8 .
 
The regularization of dropout with dropout=0.4 prevent overfitting as it tries to filter some of the information
in train time and it keeps most of it which keeps the balance of taking the benfit of training,
not as but with dropout=0.8 as in train time it causes to lose most of the information which lead to underfit
as we can see in the graphs so it leads to the lower test-acc and train-acc.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

Yes.
The cross entropy loss calculation uses the scores of the outputs of the model which resembels: distance between ground truths and the predicted score, and
the accuracy of the models measured by if the result is true / false. With those definitions what might happen is getting wrong predictions while having "big distance"
which will result in getting an increased test loss, or having a correct prediction but with a low score.

For example, taking binary classification task (results are 0,1):
    a. if a sample should be classified with 1,and it gets a right prediction but with class score a little bit over 50, the prediction will be correct so the accuracy 
        will be increased, but from the other side, this is a low score which its distance is big w.r.t to 1 which will contribute for increasing the loss.
    b. if a sample should be classified with 0,and it gets a right prediction but with class score a little smaller than 50, the prediction will be correct so the accuracy 
        will be increased, but from the other side, this is a low score which its distance is big w.r.t to 0 which will contribute for increasing the loss.
    having a lot of samples that behves like a + b will result in increasing test loss while the test accuracy getting increased.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**



1. Backpropagation algorithm is used for calculating the gradient of a loss function with respect to variables( wights/inputs) to update the inputs, and its calculated
after each step (hidden layer) while Gradient Descent is the algorithm that tries to minimize the loss function of the model and reach its minimum and its calculated
at the output of the model.


2. 
    a. In training error: gradient descent sums the error for each point in a training set, updating the model only after all training examples have been evaluated,
        while SGD picks up a single ( or some algo picks ranomly subgroup of samples < training) sample (randomly) and runs a training epoch for each example within
        the dataset and it updates each training example's parameters one at a time.

    b. from memory utilization, GD consumes more memory than SGD as it needs to hold all the dataset in memory.
    
    c. SGD may result in losses in computational efficiency when compared to batch gradient descent - SGD computationally's a whole lot faster
        (as there's single/subgroup sample/s each time )
    
    d. SGD works well (Not well, I suppose, but better than batch gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat
        noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. 
        Single samples are really noisy, while minibatches tend to average a little of the noise out. Thus, the amount of jerk is reduced when using minibatches.
        A good balance is struck when the minibatch size is small enough to avoid some of the poor local minima, but large enough that it doesn't avoid the global
        minima or better-performing local minima. (Incidently, this assumes that the best minima have a larger and deeper basin of attraction, and are therefore easier to fall into.)


3. As mentioned in two: Memory utilization, performance of the model (SGD is faster), avoiding local minimas while reaching the global minima.
Especially SGD might be more compatible to use when having large datasets which we will feel its memory & performance benifits.

4. A. No, This does not produce a gradient equivalent to GD. the GD's steps and gradient updates differ according to the size of the batch
Thus it does matter whether the loss is calculated on the dataset in batches or whole as N in the following formula will be different.

B. After each batch, we are storing the loss in the memory, when recieving out of memory that means there's no enough space to store all the losses after number of batches,
this could resulted by that the splitted batches are not small enough.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""