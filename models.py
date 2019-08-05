import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        val = self.run(x)
        if nn.as_scalar(val) >= 0: return 1
        return -1
        
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        badPass = True
        while badPass:
            badPass = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    badPass = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        
        # 2 hidden layer
        self.w1 = nn.Parameter(1, 200)
        self.b1 = nn.Parameter(1, 200)

        self.w2 = nn.Parameter(200, 200)
        self.b2 = nn.Parameter(1, 200)

        # 1 output neuron
        self.w3 = nn.Parameter(200, 1)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.
        
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        z1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        #print(z1)
        z2 = nn.ReLU(nn.AddBias(nn.Linear(z1, self.w2), self.b2))
        #print(z2)
        z3 = nn.AddBias(nn.Linear(z2, self.w3), self.b3)
        return z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss = 100
        while loss >= 0.005:
            for x, y in dataset.iterate_once(5):
                loss = nn.as_scalar(self.get_loss(x, y))
                # need to figure out how to compute average loss over dataset still
                grad_wrt_w1, grad_wrt_w2, grad_wrt_w3, grad_wrt_b1, grad_wrt_b2, grad_wrt_b3 = nn.gradients(self.get_loss(x, y), 
                    [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                self.w1.update(grad_wrt_w1, -0.01)
                self.w2.update(grad_wrt_w2, -0.01)
                self.w3.update(grad_wrt_w3, -0.01)
                self.b1.update(grad_wrt_b1, -0.01)
                self.b2.update(grad_wrt_b2, -0.01)
                self.b3.update(grad_wrt_b3, -0.01)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # 2 hidden layer
        self.w1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1, 100)

        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)

        # 1 output neuron
        self.w3 = nn.Parameter(100, 10)
        self.b3 = nn.Parameter(1, 10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        z1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        z2 = nn.ReLU(nn.AddBias(nn.Linear(z1, self.w2), self.b2))
        z3 = nn.AddBias(nn.Linear(z2, self.w3), self.b3)
        return z3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        import time
        then = time.time()
        while True:
            for x, y in dataset.iterate_once(10):
                grad_wrt_w1, grad_wrt_w2, grad_wrt_w3, grad_wrt_b1, grad_wrt_b2, grad_wrt_b3 = nn.gradients(self.get_loss(x, y), 
                    [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                self.w1.update(grad_wrt_w1, -0.008)
                self.w2.update(grad_wrt_w2, -0.008)
                self.w3.update(grad_wrt_w3, -0.008)
                self.b1.update(grad_wrt_b1, -0.008)
                self.b2.update(grad_wrt_b2, -0.008)
                self.b3.update(grad_wrt_b3, -0.008)
            if time.time() - then > 300:
                acc = dataset.get_validation_accuracy()
                print("Time Passed: {0}".format(time.time() - then))
                print("Validation Accuracy: {0}".format(acc))
                if acc >= .97: return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.w1 = nn.Parameter(self.num_chars, 300)
        self.w2 = nn.Parameter(300, 300)
        self.b2 = nn.Parameter(1, 300)
        self.w3 = nn.Parameter(300, 300)
        self.b3 = nn.Parameter(1, 300)
        self.w4 = nn.Parameter(300, 5)
        self.b4 = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """

        z = nn.Linear(xs[0], self.w1)
        for x in xs:
            z = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(x, self.w1), nn.Linear(z, self.w2)), self.b2))
            z = nn.ReLU(nn.AddBias(nn.Linear(z, self.w3), self.b3))
        return nn.AddBias(nn.Linear(z, self.w4), self.b4)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        iteration = 0
        while True:
            for x, y in dataset.iterate_once(3):
                l = self.get_loss(x, y)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_w3, grad_wrt_w4, grad_wrt_b2, grad_wrt_b3, grad_wrt_b4 = nn.gradients(l,
                    [self.w1, self.w2, self.w3, self.w4, self.b2, self.b3, self.b4])

                self.w1.update(grad_wrt_w1, -0.0075)
                self.w2.update(grad_wrt_w2, -0.0075)
                self.w3.update(grad_wrt_w3, -0.0075)
                self.w4.update(grad_wrt_w4, -0.0075)
                self.b2.update(grad_wrt_b2, -0.0075)
                self.b3.update(grad_wrt_b3, -0.0075)
                self.b4.update(grad_wrt_b4, -0.0075)

                iteration += 1
                if iteration > 35000 and dataset.get_validation_accuracy() > 0.83:
                    return