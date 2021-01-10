# Lecture 6: Language Models and RNNs

### **Language Models**
- A new language task: language models
    - why LM?
        - it is a benchmark task of understanding languages
        - it is a sub-component of many NLP tasks
    - LM(language modeling): the task of predicting what word comes next.
        - i.e. predict $P(x^{t+1}|x^{t},x^{t-1},...,x^{1})$
        - alternatively and intuitively, LM can be a system that assigns probability distributions to a piece of text.
        - the probability distribution of the whole piece of text is:
            - $P(x^1,x^2,...,x^T)=P(x^1)*P(x^2|x^1)*...*P(x^{T+1}|x^{T},x^{T-1},...,x^{1})=\prod_{i=1}^TP(x^i|x^{i-1},...,x^1)$
        - LM application: search engine, smart phone texting, etc.
- N-gram Language Model
    - N-gram: a chunk of n consecutive words.
    - assumption: the next word depends only the (n-1) previous words
        - so the next word probability: $P(x^{t+1}|x^{t},x^{t-1},...,x^{1})=\frac{n-gram\ prob}{(n-1)-gram\ prob}=\frac{P(x^n,...,x^1)}{P(x^{n-1},...,x^1)}$
        - how to compute the probability?
            - by counting by counting the probability of occurrence of words in the large corpus: P(x) = count(x)
    - categories
        - unigram, bigram, trigram, 4-gram
    - problems of N-gram
        - context problem
            - small_value gram may capture smaller context which is not good for for semantic understanding.
        - sparsity problem
            - on the one hand, when we look at the probability formula, if the numerator is 0, that is, if the true next word is never seen in the corpus (can't be infinite), then the probability will be assigned to 0.
                - a solution is do smoothing: add small probability to each output prob.
            - on the other hand, if the denominator, if the context (n-1 gram prefix) is never seen  in the corpus, then we can't  compute the probability.
                - especially, increase the value of n may aggravate this problem since adding the n-1 prefix will be less repeated in the corpus as the value of n increases.
                - a solution is that we can back off to n-2 gram if n-1 gram prefix doesn't work and so forth.
        - storage problem
            - need to store the count
                - increase the value of n can also aggravate this problem, nor can increase the corpus.
    - N-gram model in practice
        - e.g. Reuters: business & finance LM
        - limitations
            - only (n-1) context, results are grammatically correct but incoherent.
            - however, increase the corpus or the value of n won't solve the problems. need better ideas.
    - how to improve
        - window-based neural LM
            - just like the application in NER
            - just a gram-based, or window-based feed forward network structure.
            - a fixed window neural LM

                 <img src="pics_lecture6/Untitled.png" width = "400" height = "150" alt="d" vertical-align=center />

                - details
                    - a window of pre-words and the current word
                    - one-hot (can be word embeddings) input, concatenated.
                    - one hidden layer
                    - softmax output
                - advantages
                    - no sparsity problem: it will all assigns an probability (softmax on every distinct word in the vocabulary)
                    - it doesn't need to store the n-grams (just word-vectors and the corpus).
                - disadvantages
                    - fixed window: context still not fully captured. if larger window: the parameters in the NN will increase
                    - there are also problems in the weight matrix in NN

                        <img src="pics_lecture6/Untitled 1.png" width = "400" height = "150" alt="d" align=center />

                        - image the weight matrix W multiplies the input word embedding column E. for the first element in E, during the matrix multiplication it will only multiplies with the first column in W. Thus we may learn the same function for every column(area) in the W since the window is sliding through every word.
                        - on the other hand, at each time, every embedding in E is multiplied with different columns in W,
                        - it turns out to be inefficient, everyone in the input should share the same learning matrix W. **we need a neural architecture that can handle any input with symmetric processing.**
        - RNN

<br>

### **RNN**
- Character

    <img src="pics_lecture6/Untitled 2.png" width = "500" height = "250" alt="d" vertical-align=center />

    - input a sequence of any arbitrary length
    - hidden states. one mutating weight matrix W.
    - output at every time stem is depended on the previous hidden state and the input now.
- RNN LM
    - details

        <img src="pics_lecture6/Untitled 3.png" width = "500" height = "300" alt="d" align=center />

        - input: word embeddings $e^{(t)}=E*x^{(t)}$
        - hidden layer: $h^t=\sigma(W_hh^{(t-1)}+W_ee^t+b_h)$
        - output layer: $y^t=softmax(U*h^t+b_o)$ (softmax classifies over the whole vocabulary).
    - advantages
        - can process any length of input
        - can use the information from many steps back
        - model size is fixed, we only need to store W_h, W_e, vectors, corpus, etc.
        - same weights, symmetry in how inputs are transformed.
    - disadvantage
        - slow of recurrent process
        - hard to trace information from many steps back
            - LSTM mechanism may solve this problem
    - how to train RNN

        <img src="pics_lecture6/Untitled 4.png" width = "500" height = "250" alt="d" align=center />

        - large corpus of text (a sequence of words)
        - feed in to RNN with cross entropy loss function

            <img src="pics_lecture6/Untitled 5.png" width = "400" height = "150" alt="d" vertical-align=center />

            - the loss of step_t is the cross entropy of the predicted and the true one-hot label
            - the total loss is averaged J over a dataset.
        - optimization: compute the loss over a whole corpus is expensive, based on the idea of SGD, in RNN we regard one or some sentences as a batch of data, compute the total loss of a batch and do back propagation.
        - back propagation
            - derivative of W_h: $\frac{\partial J^{t}}{\partial W_h}=\sum_{i=1}^t \frac{\partial J^{t}}{\partial W_h}|_{i}$
            - why? multivariable chain rule: $\frac{d}{dt}f(x(t),y(t))=\frac{df}{dx}\frac{dx}{dt}+\frac{df}{dy}\frac{dy}{dt}$, so the derivative of J over W_h is the sum of the derivative of J over W_h at every time step.
            - how to compute the probability? accumulate at each time step, summing the gradients before moving forward to next step.
                - this is called back propagation through time
    - evaluation
        - perplexity = $\prod_{t=1}^T=\frac{1}{P_{lm}(x^{(t+1)}|x^t,...,x^1)}^{\frac1T}=exp(\frac1T\sum_{i=1}^{T}-log(y_{x+1}^t))=exp(J(\theta))$
        - the inverse probability to the corpus, the lower the better.
    - text generated with RNN model
        - input word, RNN model, get a probability distribution, sample the output as the input in next time step, RNN model....
        - word LM, character LM
- RNN Applications
    - build a LM
    - POS Tagging, NER

        <img src="pics_lecture6/Untitled 6.png" width = "400" height = "150" alt="d" align=center />

    - sentiment/sentence classification

        <img src="pics_lecture6/Untitled 7.png" width = "400" height = "150" alt="d" align=center />

        - get sentence embedding and use the final hidden state
        - or element-wise max/mean of hidden states.

            <img src="pics_lecture6/Untitled 8.png" width = "400" height = "150" alt="d" align=center />

    - question answering, machine translation
        - RNN as a encoder
    - speech recognition

        <img src="pics_lecture6/Untitled 9.png" width = "400" height = "150" alt="d" align=center />

        - conditional LM
    - Vanilla RNN is the naive RNN we used in this lecture.