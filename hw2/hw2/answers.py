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
    activation = 'relu'
    n_layers = 4
    hidden_dims = 30
    out_activation = 'softmax'
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
    lr = 0.0025
    weight_decay = 0.002
    loss_fn = torch.nn.CrossEntropyLoss() # Maybe another loss function.
    momentum = 0.99
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""

1. Since we used in our model the crossEntropy loss function, which is a convex function, gradient descent is guaranteed to converge to a global minimum of the loss function. Since the achieved accuracy was very high on the training set, and since it platued at the end, that means we managed to converge and we can say the optimization error is relatively low and close to 0.

2. Generlization error refers to how well we can predict unseen samples after training our model and whether there is overfitting or not. That is, we need to look at the accuracy of the validation set (test set in the plots) - as you can see the validation set accuracy is very high which means the Generlization error is low.

3. The approximation error refers to the fact that we used a restricted family of hypothesis to train our model which might not include the ground-truth function. We did achieve high training/test accuracies which means the Approximation error is not high, but we could've gotten better results if we had chosen a different family of hypothesis, or different hyperparameters for our model (like you can see in the Architecture Experiment section).


"""

part3_q2 = r"""

Based on the confusion matrix plots, we expect the FPR/FNR ratio of the validation set to be correlated to the FPR/FNR ratio of the training set. And indeed, we can see that in both the FPR is higher than the FNR. Although the threshold is 0.5, this bias towards lower FNR can be explained from the real-world data-generation process, where the samples were not generated i.i.d. As such, it looks like there were more duplicate/correlated samples for the positive class compared to the negative which explains why we were able to generalize better on the negative class. Finally, our claim is supported by the ROC evaluation where we found the optimal threshold to be 0.57 which is biased towards reducing the FPR error.

"""

part3_q3 = r"""

Assuming 'positive' refers to the patient having the disease, then:

1. In this scenario, since the disease is not lethal and can be treated at a low-cost if not diagnosed early, then we want to prioritize reducing the False-Positive diagnosis to reduce the overall cost. This means, we want to be more money-wise conversative and opt for a higher threshold which lowers the chances of FPs. 

2. On the opposite of scenario 1, since the disease is lethal if not diagnosed early,  we want to prioritize increasing the False-Positive diagnosis to reduce the risk of human-lives. That means, we want to choose a threshold lower than the optimal one, increase the rate of FP and reduce the risk of missing patients that could actually have the disease (better be safe than sorry).

"""

## Sarah AI
part3_q4 = r"""
1. For fixed depth and varying widths, we notice that for depth=4, both the valid_accuracy and test_accuracy increase as we increase the width, while for the rest of the depth values, increasing the width doesn't necessarily improve the accuracies. As we know, adding layers and more neurons to the MLP can actually reduce accuracy at times, as it depends on the data we have and whether we overfit it or not.

2. For fixed width and varying depths, we notice that for depth=2, both the valid_accuracy and test_accuracy decrease as we increase the depth (could be a sign of overfitting the training data), while for the rest of the depth values, increasing the depth doesn't necessarily improve/reduce the accuracies. As we know, adding layers and more neurons to the MLP can actually reduce accuracy at times, as it depends on the data we have and whether we overfit it or not.

3. Comparing both configurations, we can see that for depth=1, width=32 we get better test/validation accuracies compared to depth=4, width=8 although both of them have the same number of parameters. The difference could be due to the fact that in the second case, we are applying the 'tanh' activation function 4 times in a row while in the first case we apply it only once on the 32 different neurons.

4. The purpose of the validation set is to help us determine the optimal value for the hyperparameters, which is lambda the threshold in this case. Since the validation set acts as an unseen test set for the training set, then finding the optimal lambda that minimzes the loss on the validation set, should help us when we generalize on new unseen samples and give lower errors. And indeed, we noticed that the test_accuracy improved when we chose the optimal lambda.

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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.1
    momentum = 0.002
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. Using the formula of calculating the number of params from CNN tutorial:
K x (Cin * F^2 +1 )

in regular block with the params: 
kernel = 3, Cin=265, FxF = (3,3) ==>
we have 64 * (3*3*256+1) + 64*(3*3*64+1) = 184,448 

in ResnetBlock:
the 1st block we have, k=64, f=1, cin = 256
the 2nd block we have, k=64, f=3, cin = 64
the 3rd block we have, k=256,f=1, cin = 64

64*(1*1*256+1) + 64*(3*3*64+1) + 256*(1*1*64+1) = 70,016

In resnet block we have less parameters than in regular block.


2. Having H,W, the number of multiplication we must to do is H*W* NumOfParams
so in regular block we will do: 184,448 * H * W
& for the bottleneck block we'll have 70,016 * H * W
Again, resnet block will have less operations to apply.


3. 
a. The recieptive field in regular block is 5 , as there are two conv layers with 3x3 kernels; and the recieptive
field of the block in resnet is 3, as the maximum kernel size of a layer is 3, and the other layers have 1x1 kernels;
as a result, the input spatially is better in regular block as more input sample (pixels) are combined in the
computation of each output sample (pixel).

b. Both architectures go through all the channels, as a result, the ability to combine across feature maps is the
same in both architectures.

In spatially( within feature maps), because the receptive field in regular block is bigger than in bottlnek block, it is better to combine the
the inputs in there.




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

1.a For the first image (Dolphins), the ultralytics yolov3  algorithm managed to correctly identify bounding boxes which include objects in them, but failed to classify them correctly as dolphins. It instead classified them as a bird or a person.

1.b For the second image (Dogs/Cats), the ultralytics yolov3  algorithm managed to correctly localize and classify the two right most dogs (white/brown), but failed to localize and calssify the black dog and grey cat.

2.a.  For the first image (Dolphins), from the documentation behind yolov3, it's stated that the algorithm was pretrained on the COCO dataset, which doesn't include any images of dolphins, and this explains the misclassification (I downloaded a bunch of dolphins images from googles for and tried them out. It failed to detect any dolphins). Also, assuming dolphin images were included in the training sets, if there was misclassifcation then it's probably due to the fact that the image includes silhouettes of dolphins instead of colored real images of dolphins which might confuse the algorithm.

2.b For the second image (Dogs/Cats), we noticed that the image size was large 750x750 (the model was trained on 640x640 images), and also the dogs themselves are large and take up a noticable portion of the image. What we tried to do was downsample the image to 50% of its size or lower, and we noticed that by doing so the yolov3 algoirthm managed to correctly identify the black dog. Since the yolov3 algorithm heavily depends on the anchor boxes which are representatives of the most accurate bounding boxes from the training dataset (using K-means), the failure in detecing the black dog of this size indicates that the training set did not include objects this large of dogs, and as such the given anchor boxes were not ideal for detecting a black dog of this size. By downsampling the image, the black dog's bounding box became closer to the default anchor boxes and this helped yolov3 detect it correctly as the yolov3 performs better when the detection requires only a small alteration in the anchor boxes. While in the original sized image, the bounding box of the dog was so large and the closest thing that the algorithm found was that the box contains a cat. 

Now the question is why the cat was not detected in the original-sized image. For that, we ran a couple of tests as you can see in the code cell below:

1) We tried cropping out the bounding box which includes both a dog and a cat (classified as a cat) and passing it through the model again ==> The small cat bounding box was localized and classified correctly. But the dog was still classified as a cat. Notice here that the confidence level for the bounding box of the cat was 71%.

2) From the image in (1) above, we cropped out the bounding box of the cat, downsampled the image and passed it through the model again. This time it correctly identified the dog.

3) We tried changing the parameters of the model. The default confidence level is 0.25, so we changed it to 0.06 and downsampled the image to 0.48%. Here we managed to classify and localize all the objects correctly.

4) From (3), we lowered the confidence to 0.06 and ran the model on the original image. What we noticed was the cat was actually identified and localized correctly, but its confidence was at the limit (0.25), so from NPS, it was omitted from the detection.

From the experiments above, our conclusion regarding how to improve the algorithm are as follows:

1) The dog can be detected by downsampling the image.
2) The cat can be detected by lowering the confidence level.
3) Cropping out the false bounding box enclosing the black dog/cat and passing it through the model again increased the confidence level of the cat's bounding box from 0.25->0.71. This means the confidence heavily depends on the surroundings in the original image. This suggests a psuedo-postprocessing-reiterative-algorithm where for bounding boxes with low confidence, we can crop them out and run them through the model again to see if any new objects are detected or were missed. Or, we can from the beginning, remove the high confidence boxes from the original image, and pass the leftover image through the model again and see if it managed to detect any missing objects.
4) We can improve the algorithm by including more similar images in the training where objects are rather large and have overlaps between them.
5) We believe the algorithm could have performed better had we selected a more "appropriate" intital set of anchor boxes.

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
We demonstrated four types of object detection obstacles:
1) Textured Background: As you can see above, the dog's fur blended in with the carpet and as such no objects were detected.
2) Deformation: As you can see above, the bottled was deformed and squished, and as such the algorithm mislabled it as a kite.
3) Blurring: As you can see above, the red bus on the right was not detected due to motion blurring.

"""

part6_bonus = r"""
1) For the deformed image, we tried strecthing the image vertically so it resembles a bottle more. We can see the algorithm managed to detect it.
2) For the blurred image, we did not have time to write a code but we believe we could've cropped the right misdetected bus, used a deconvoluion Kernel to deblur the image or a tool which deblurs images uses GAN (there is an opensource code), and the algorithm would then be able to detect it.

"""