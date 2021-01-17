# Lecture 11: Convolutional Networks for NLP

### **Intro to CNNs (including materials)**

- from RNN to CNN
    - RNN cannot capture phrases without (prefix) context, and always capture too much of last words in the final vector.

        <img src="pics_lecture11/Untitled.png" width = "300" height = "150" alt="d" vertical-align=center />

        - in RNN's perspective, a phrase(or word) is understood by NN from it's **context.** no **independent** representations of a span of words.
    - CNN is good at what RNN is not. we may compute vectors for every possible word subsequence of a certain length, even regardless of grammatical or linguistically plausible.
        - intuitively, learn from the inside.
        - then to group them as the higher-level representation.
- Convolution (discrete convolution)
    - often used 2d convolution in vision tasks
    - 1d convolution is for text

        <img src="pics_lecture11/Untitled 1.png" width = "400" height = "250" alt="d" vertical-align=center />

        - the text shape: (word, vector)
        - the filter shape: (size, vector)
        - the output shape: (word_num-filter_size+padding+1, filter_num/features/channels)
            - zero-padding tech to maintain the size in the output: wide convolution.
            - could also use different filters to increase the output size: different filters→different latent features.
        - max-pooling over time: torch.max(x,dim=2)
            - max each feature/channel, get a (1,feature_num) vector as the output
            - the concept of max pooling is that: on the one hand, if the value if high, it is indicated that there are something relevant with the feature in the sentence.
            - alternatively we can do average-pooling as well, which has a different semantics, that is, the average amount of relevance on the feature.
            - in most cases, max-pooling is better because of sparsity.
            - the idea of pooling in NLP: to turn different length in to one by max the column, but the positional information of features is hard to maintain after pooling. to fix this, there are tricks like k-max pooling, chunk-max pooling(kind of like k-max but k-max is global max).
                - min-pooling is not preferred since we may use ReLU for activation
        - other useful notions
            - stride: the (the default is 1)
                - less useful
            - local pooling with stride 2: max through rows in the convolutional output.
                - less useful
            - k-max pooling over time: pick top k max value.
            - dilation: 扩张卷积, useful.
                - in CNN, skip some rows (dilation: 2), this enables to see a bigger spread of sentence without having many parameters.

                    <img src="pics_lecture11/Untitled 2.png" width = "400" height = "250" alt="d" vertical-align=center />

            - stacked convolutional layers
    - conclusion
        - a standard text 1d convolution process is this:
            - first, there is the matrix of (seq_len, vetor_dim)
            - in the first convolutional layer, we have filters with many features, each feature filter can varies with sizes(n-gram filtering). The output of conv is: (seq_len-filter_size+1 (without padding and striding), feature_num (filter_num)).
            - then there is a pooling layer, if we use max pooling, then the output is (1, feature_num), which is the input of the flatten layer.
            - if you wanna another convolutional layer, you should use local max-pooling instead.
        - how much context we look at one time step is important in CNN.
        - bigger filter, dilation or stacked CNN may be helpful
        - **CNN Formula**
          - $H_{out}=\frac{H_{in}+2*padding-(dilation*(kernel-1)+1)}{stride}+1$
          - 其中，$\frac{H_{in}-kernel}{stride}+1$，可由数列推算得
          - 对于分母，最终得kernel size为原kernel-1（扩张的个数）*扩张的范围+1（最后一行/列），最终得H_in得加上padding，所以formula如上。


<br>

### **Simple CNN for Sentence Classification**

- CNN for sentence classification. EMNLP 2014
    - single layer CNN
- sentence classification
    - mainly sentimental
    - or can be whether subjective or objective, etc.
- details

    <img src="pics_lecture11/Untitled 3.png" width = "500" height = "300" alt="d" vertical-align=center />

    - a word is represented in a k-dimensional vector
    - n*k matrix represents a sentence
    - filter is h*k where h is the window-size
    - use different filter (with variants in filter_size) to obtain multiple features and construct different feature maps.
    - a max-pooling is after convolutional layer for each feature map: to capture the most important feature.
    - a fully-connected layer with dropout is added and softmax the classification
- trick: multi-channel input idea: CNN-multichannel
    - initialize with pre-trained word vectors (both from W2V), start with two copies.
    - one is kept static while the other is fine-tuned through bp.
    - Both channel sets are added together before max-pooling.
    - why multi-channel? because if fine-tuned, the vector is changed by focusing the downstream task, but in the case that some words may only appear in the test set, we need more generalized representations to them. So multi-channel (static pre-trained representations) is also needed.
        - and the way to deal with the combination of the two channels varies. Averaging, or concatenating, etc.
- regularization
    - **dropout**: gives 2-4% accuracy improvement.
    - L2 norm in softmax

<br>

### **CNN Potpourri**

- **Model Comparison: A Growing Toolkit**
    - **bag of vectors:** W2V, good baseline for simple classification problems, especially when following by a few ReLU layers.
        - kind of ignore context, focusing on the statistics of words.
        - Deep Averaging Networks: DAN, try to combine BOW models and RNN models, capturing both statistical and syntactical features.
    - **window model**: good for single word classification for problems do not need wide context. e.g. POS, NER.
        - a little context from windows.
        - based on W2V
    - **CNNs**:
        - good for text classification
        - need zero padding for shorter phrases
        - hard to interpret.
        - compile well on GPU, parallel, efficient.
    - **RNNs**:
        - cognitively plausible
        - not best for classification but good for LM (predicting what's coming next) with mechanisms like attention, good for sequence tagging and classification.
        - slower than CNN
- gated units used vertically

    <img src="pics_lecture11/Untitled 4.png" width = "400" height = "200" alt="d" vertical-align=center />

    - residual block
        - need padding in the convolutional layers to fill the shrinking parts so as to be added by the identity.
    - highway block
        - gated units used in the addition of short-connections with the candidate output.
        - T gate and C gate (forget and input gates)
        - more complex, 'feels' more powerful. but not really so powerful.
        - anything computed by the highway is actually 'can be computed' by the residual.
- batch normalization: (BatchNorm)
    - often used in CNNs
    - makes models much less sensitive to parameter initialization, since outputs are automatically rescaled.
- size one convolution
    - aka Network-in-network connections, are convolutional kernels with kernel_size=1
    - Intuitively, every convolutional layer contains a feature_num (input channel & output channel) changes, which gives a fully connected linear layer across channels!
    - it is useful to map many channels to fewer channels
    - it also add additional neural network layers with very few additional parameters
- CNN application:
    - translation

        <img src="pics_lecture11/Untitled 5.png" width = "300" height = "400" alt="d" vertical-align=center />

        - Use CNN for encoding and RNN for decoding
    - learning character-level representations for POS tagging

        <img src="pics_lecture11/Untitled 6.png" width = "300" height = "250" alt="d" vertical-align=center />

        - convolution over characters to generate word embeddings
        - fixed window of word embeddings is used for POS tagging
    - character-aware neural language models

        <img src="pics_lecture11/Untitled 7.png" width = "400" height = "500" alt="d" vertical-align=center />

        - character-based
        - using convolutional, highway network and LSTM

<br>

### **Deep CNN for Sentence Classification**

- Conneau et. alEACL 2017: VD-CNN.
- motivation: what happens if we build a vision-like very-deep system for NLP?
- details

    <img src="pics_lecture11/Untitled 8.png" width = "400" height = "500" alt="d" vertical-align=center />

    - works from the character level, same input_size (the length of the document in the character-level): 1024 (if longer then truncate it).
    - local pooling (stride=2 let's say): halves sequence_length (length/2) and doubles the feature_num in the next layer (every filter has two features output).
    - convolutional layer: (Temp Conv): 3 kernel size, 64 output channels.
    - convolutional  block

        <img src="pics_lecture11/Untitled 9.png" width = "300" height = "250" alt="d" vertical-align=center />

        - pad to maintain the dimension
    - optional shortcut to keep the same size of output in each subpart.
- the system much looks like a vision system, similar to VGG or ResNet
- max pooling is fine

<br>

### **Quasi-Recurrent Neural Networks (QRNN)**

- the problem of RNN
    - don't parallel well, thus slow
- details: a more efficient way to train RNN

    <img src="pics_lecture11/Untitled 10.png" width = "500" height = "150" alt="d" vertical-align=center />

    - first use a convolutional layer to extract features

        <img src="pics_lecture11/Untitled 11.png" width = "400" height = "150" alt="d" vertical-align=center />

        - use convolutional filter to calculate the gate function (rather than the input and the previous hidden state).
        - masked convolution is used so that only the previous sequence can effect the current state
        - the size of the filter depends the n-gram feature the filter will extract. (if the kernel width is 2, the information of the previous 1 time-step and the current input will be used to calculated the current gate value).
    - next, element-wise gated pseudo-recurrence for parallelism across channels is done in the pooling layer
        - and the hidden output: $h_t = f_t * h_{t-1} + (1-f_t)*z_t$, faking the recurrence by computing gates locally for better parallelism.
            - this is called f-pooling
        - there are also fo-pooling($h_t =h_t * o_t$)
        - and ifo-pooling ($h_t = f_t * h_{t-1} + (i_t)*z_t$)
        - much less time-step based computation, more parallelism.
    - for regularization, QRNN uses 'zone out', a new way of dropout. (pick a subset of the channels to drop)

        <img src="pics_lecture11/Untitled 12.png" width = "400" height = "50" alt="d" vertical-align=center />

- outcomes: better than LSTMs in sentiment analysis; slightly better and faster in LM
    - QRNN解决了两种标准架构的缺点。 它允许并行处理并捕获长期依赖性，例如CNN，还允许输出依赖序列中令牌的顺序，例如RNN。
- similar architecture: SRU(Simple Recurrent Units)
- limitations:
    - doesn't work for character-level LMs as well as LSTMs
        - trouble learning much longer dependencies?
    - often need deeper network to outperforms than LSTMs
        - use depth as a substitute for true recurrence
- problems with RNNs and Motivation for Transformers
    - parallelization
    - RNNs still gain from attention mechanism to deal with long range dependencies – path length between states grows with sequence otherwise.
    - if attention gives us access to any state ... maybe we don’t need the RNN?