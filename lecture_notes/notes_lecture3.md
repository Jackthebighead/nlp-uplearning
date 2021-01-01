# Lecture 3: Word Window Classification, Nueral Network and Matrix Calculus

- ### **Classifications in NLP**
    - Classification
        - for training dataset consisting of samples {x_i, y_i}
        - x_i are inputs, like words (indices/vectors), sentences, documents
        - y_i are labels, like classes (sentiment, name entity, decisions), other words, multi-word sequences, etc.
        - the task is to train a model to detect a decision boundary over the training dataset.
            - in Logistic Regression we learn the weights, and predict$p(y|x)=\frac{e^{W_yX}}{\sum_{c=1}^ce^{W_cX}}$
                - the objective is to minimize the negative log probability, i.e. $-log(p(y|x))$
                - what we usually use is the cross entropy
                - actually, cross entropy equals to negative log probability of the true class
                    - $H(p,q)=\sum_{c=1}^cp(c)log(q(c))$, it equals to $-\sum_{c=1}^{c}log(q(c))$ in the classification tasks.
                - train with softmax and cross entropy loss
                    - but softmax **alone** is not powerful, it is good at mapping values into a probability distribution for classification (when **alone** and, not serves as the activation) but it gives only linear decision boundaries, limiting.
    - Neural Network for the win
        - NN can learn much more complex functions and nonlinear decision boundaries.
        - Neural Network
            - artificial neuron
                
                <img src="pics_lecture3/Untitled.png" width = "300" height = "150" alt="d" align=center />
            - a neuron can be a  binary logistic regression unit
                - where the activation is nonlinear like sigmoid, and there are linear weights and bias.
                    - activation function can be regarded as indicators of the combination of features (at a higher abstraction).
                - and a NN can be a structure that running several logistic regressions at the same time, and can have multiple layers.
                - some matrix notation of a layer

                    <img src="pics_lecture3/Untitled 1.png" width = "300" height = "150" alt="d" align=center />

    - classification difference with word vectors
        - in NLP deep learning
            - we learn both the conventional parameters W and the word vector representations x.
            - so we need word embeddings and more complex networks.
- ### **Name Entity Recognition (NER)**
    - the task: find and classify names in text.
    - possible purposes: question answering, etc.
    - NER on word sequences
        - we predict entities by classifying words in context and then extracting entities as word subsequences.
    - challenges: hard to know if it is an entity, hard to know the class of unknown/novel entity, hard to work out the boundaries of entity, and entity maybe ambiguous in meaning.
    - Methods: Word Window Classifications
        - window classification: classify a word in its context window.
        - problems of simply average word vectors in a window: no position information.
            - solution
                - train a simple softmax classifier to classify a center word by taking concatenation of word vectors of context it in a window.

                    <img src="pics_lecture3/Untitled 2.png" width = "300" height = "100" alt="d" align=center />

        - problems of unnormalized scores: corrupt windows (center word isn't labels as a NER location in the corpus) are too easy to find.
            - solution: go over all positions in a corpus, but it will be supervised and only some positions should get a high score.
                - Feed Forward Computation

                    <img src="pics_lecture3/Untitled 3.png" width = "400" height = "200" alt="d" align=center />

                    - the intuition of feed forward is that, take NER for example, we want our model to classify whether the center word of the sentence 'museums in Paris are amazing', it is likely that this task depends not only the word vectors in the window but the interactions between the words (a higher feature, like, the first word should be 'museum' and the second is 'is'), and such non-linear decisions can be captures by a more complex way such as the operations of intermediate layer, to be simple, we can use a U matrix and generate the score of the hidden layer as our final score.

                        <img src="pics_lecture3/Untitled 4.png" width = "200" height = "150" alt="d" align=center />

                - the idea for training objective: make the true window (opposite to the corrupt window) scores higher and corrupt window lower.

                    <img src="pics_lecture3/Untitled 5.png" width = "300" height = "100" alt="d" align=center />

                    - in the objective, we try to make s bigger and s_c smaller so as to min the loss, so there is a (-s+s_c). and there should also be a margin of safety in order to let the true labeled data to score higher than the false labeled data by a margin (which means error is calculated when $s-s_c<\Delta (\Delta >0)\ rather\ than\ s-s_c<0$, so we use 1 to be the margin and apply it to the objective.
                        - the objective is called max margin loss
                - sample several corrupt windows per true window. sum over all.
                    - similar to negative sampling in w2v
                - train with SGD
- ### **Calculus, Chain rules, etc.**
    - skipped because they are fundamental
- ### **Back Propagation**

    <img src="pics_lecture3/Untitled 6.png" width = "300" height = "150" alt="d" align=center />

    - Back propagation is technique that allows us to use the chain rule of differentiation to calculate loss gradients for any parameter used in the feed-forward computation on the model.
    - computational efficiency
        - vector wise updates
        - save the intermediate result to reduce redundant computation in bp.
- ### **NN: Tips and Tricks**
    - gradient check
        - numerically approximating gradients rather than analytically calculating it.
        - allow to precisely estimate the derivative to any parameter, serves as a useful sanity check on the correctness of the analytic derivatives.
        - the numerical gradient of parameter theta is given by centered difference formula
            - $f'(\theta)\approx \frac{J(\theta^{(i+)})-J(\theta^{(i-)})}{2\sigma}$, where sigma is called 扰动(often e^-5), J(\theta) for i+ is the loss for the para theta+sigma after finishing a forward propagation.
            - however, it is computationally expensive since there are two forward propagations.
    - overfitting
        - regularization
            - add a regularization term to the loss function
            - $J_R=J+\lambda\sum_{i=1}^{L}||W^{(i)}||_F$
            - the essence of regularization is penalizing weights whose value are too large when optimizing the loss function.
            - the regularization term can reduce the flexibility of the model so as to reduce the probability of overfitting.
            - the regularization term can be regraded as the prior distribution in Bayes.
            - no restriction on the bias since it is not sensitive to input and little contribution to the output.
        - dropout
            - actually dropout is another tech of regularization in NN
            - drop neurons with a certain probability when doing forward or back propagationin the training process.
            - the essence of dropout is to train a different subnet from a net once and average it to avoid overfitting.
            - in RNN we use variational dropout since neurons in RNN are time series. Thus we can drop the non time dimension connection (i.e. acyclic connection) is randomly lost. As shown in the figure, the dotted line indicates random discard, and different colors indicate different discard masks.
        - neuron unit
            - sigmoid

                <img src="pics_lecture3/Untitled 7.png" width = "200" height = "100" alt="d" align=center />

                - $\sigma(z)=\frac1{1+exp(-z)}$
                - derivative: $\sigma'(z)=\sigma(z)(1-\sigma(z))$
            - tanh: similar to sigmoid but faster convergence, and output (-1,1).

                <img src="pics_lecture3/Untitled 8.png" width = "200" height = "100" alt="d" align=center />

                - $tanh(z)=\frac{exp(z)-exp(-z)}{exp(z)+exp(-z)}=2\sigma(2z)-1$
                - derivative: $tanh'(z)=1-tanh^2(z)$
            - hard tanh: computation less

                <img src="pics_lecture3/Untitled 9.png" width = "200" height = "100" alt="d" align=center />
            - soft sign

                <img src="pics_lecture3/Untitled 10.png" width = "200" height = "100" alt="d" align=center />

                - $softsign(z)=\frac z{1+|z|}$
                - derivative: $softsign'(z)=\frac{sgn(z)}{(1+z)^2}$
            - ReLU: Rectified Linear Unit, commonly used in CV

                <img src="pics_lecture3/Untitled 11.png" width = "200" height = "100" alt="d" align=center />

                - $rect(z)=max(z,0)$
                - leaky ReLU: still a gradient when z<0

                    $leaky(z)=max(z,k*z)$

                    <img src="pics_lecture3/Untitled 12.png" width = "200" height = "100" alt="d" align=center />

    - data preprocessing
        - mean subtraction: zero center the dataset by subtracting mean vector from the data.
            - the mean is calculated only across the training set, the subtraction is done across the training, validation and testing sets.
        - normalization: scale every input feature dimension to have similar ranges of magnitudes.
            - done simply by dividing the features by their respective standard deviation calculated across the training set.
            - there are other normalization methods like min-max normalization etc.
        - standardization: transfer data to a standard normal distribution. minus mean and divided by standard deviation.
            - mean subtraction+normalization
        - whitening: not so popular: essentially converts the data to a state that features become uncorrelated and have a variance of 1, that is, the correlation matrix is a identity.
            - done by mean-subtracting and then SVD(get USV), and project data with U matrix, then divided by the single value in S matrix to scale our data.
    - parameter initialization
        - a proper way of initializing parameters is a key step in optimizing the NN performance.
        - usually we set the parameters with a random small value from a distribution around 0.
        - in the thesis *Understanding the difficulty of training deep feedforward neural networks*, for sigmoid and tanh activation unit, following the below distribution to initialize the parameters can achieve faster convergence and lower bias.

            <img src="pics_lecture3/Untitled 13.png" width = "250" height = "50" alt="d" align=center />

            - where n^l is the input unit number while n^l+1 is the number of output unit.
    - learning strategies
        - the speed of updating parameters can be controlled by the learning rate.
        - the problem of learning rate is that, if too large, the loss function may be hard to converge to a minimum point. On the other hand, the time and resources fro training will be larger if too small.
        - we need to adjust the learning rate during training
            - improve the naive approach to setting learning learning rates: scale the learning rate of a weight by the inverse square root of the number of input neurons.
            - annealing: 退火 learning rate is reduced after iterations.
                - a common way to perform annealing: reduce the learning rate by a factor x after every n iterations.
                - decrease over time: $\alpha(t)=\frac{\alpha_0\sigma}{max(t,\sigma)}$, where sigma is the time threshold.
                - exponential decay: $\alpha(t)=\alpha_0e^{-kt}$
    - optimizations on parameter update
        - Momentum updates: a variant of gradient descent inspired by momentum in physics.

            ```python
            # Computes a standard momentum update
            # on parameters theta
            v = /mu * v - /alpha * grad_theta
            theta += v
            ```

            - a method which doesn't need to set learning rate artificially. the parameter is decreasing automatically.
            - where v is the momentum, grad_theta is the gradient.
        - AdaGrad: adaptive gd
            - one key difference with SGD: different parameter has different learning rate and, the update of each parameter depends on the history of update. i.e. parameters with little update will have larger update in the future.

                ```python
                # Assume the gradient dx and parameter vector x
                cache += dx ** 2
                x += -learning_rate * dx / np.sqrt(cache + 1e-8)
                ```

            - RMSProp is another adaptive method, it is a variation of AdaGrad which add a decay-rate of the history. seems better.

                ```python
                # Update rule for RMS prop
                cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
                x += -learning_rate * dx / (np.sqrt(cache) + eps)
                ```

        - Adam: combines RMSProp with momentum, and a bias correlation.

            ```python
            # Update rule for Adam
            m = beta * m + (1 - beta1) * dx # momentum for gradient update.
            v = beta * v + (1 - beta2) * (dx ** 2) # RMSProp for history(cache) update.
            x += -learning_rate * m / (np.sqrt(v) + eps) # combine them.
            ```