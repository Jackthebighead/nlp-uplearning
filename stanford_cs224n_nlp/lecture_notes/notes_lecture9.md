# Lecture 9: Practical Tips for FP

### **Ways To Do A Project**

- at a prosaic level, we'll touch on
    - baselines
    - benchmarks
    - evaluation
    - error analysis
    - paper writing
- project types
    - default project: question answering system for SQuAD (Stanford Question Answering Dataset).
    - custom project
        - how to solve an existing task effectively (like using NN).
        - implement a complex neural architecture and demonstrate the performance on some data.
        - come up with a new NN
        - analysis project, analyze the behavior of a model, how it represents linguistic knowledge or what kinds of phenomena it can handle or errors that it makes.
        - rare theoretical project
- somewhere to start

    <img src="pics_lecture9/Untitled.png" width = "400" height = "200" alt="d" vertical-align=center />l%20Tips%20for%20FP%200f28c851eaf1410f9d408b9f9ba846a6/Untitled.png)

    - some useful links
        - [http://www.arxiv-sanity.com](http://www.arxiv-sanity.com/)
        - [https://paperswithcode.com/sota](https://paperswithcode.com/sota)
    - data
        - [https://universaldependencies.org](https://universaldependencies.org/): tree banks
        - [http://statmt.org](http://statmt.org/) MT
        - [https://catalog.ldc.upenn.edu/](https://catalog.ldc.upenn.edu/) linguistic data consortium

<br>

### **A Review of RNNs**

- the idea of solving vanishing gradient problem, is: one way to do that is attention, sort of create shortcut connections of every time step and calculate attention distribution. Another way is another way is gated RNNs, that is, instead of keep long-distance direct connections we can create adaptive shortcut connections. An analogy is these are like registers in computer, so we can achieve a long-term memory.
- GRU: we using a computation from one step back, partly inheriting the hidden state from the previous time step, and partly taking the current new candidate content (the process of calculating the candidate is exactly what Vanilla RNN does for the hidden state). And we control the adapting choice by the gate function, called update gate.
    - And also, in the way of getting the new candidate hidden state, we can sort of partly inheriting from the previous state, so we control that by setting another gate function called reset gate.
- the real secret of both LSTM and GRU is rather than multiplying things together, we add things together by create direct addition of the current state and the previous state.
- the difference between LSTM and GRU is that, GRU uses a linear function of update gate while LSTM uses two separate gates to control ,which is theoretically powerful
    - three gates to perform forget or keep parts of the information.

        <img src="pics_lecture9/Untitled 1.png" width = "500" height = "200" alt="d" vertical-align=center />


<br>

### **A Little More in MT**

- the large output vocabulary problem
    - expensive computations of softmax
    - the word generation problem: OOV
        - vocabulary: often set: 50K
        - hierarchical softmax
        - NCE: noise-contrastive estimation, negative sampling
        - train on a subset of the vocabulary at a time, test on a smart subset of possible translations.
        - use attention to work out what you are translating
            - simply look up the new OOV word like dictionary lookup
        - more ideas: word pieces, character models
- evaluation
    - how BLUE evaluation metric works

        <img src="pics_lecture9/Untitled 2.png" width = "400" height = "250" alt="d" vertical-align=center />

        - the precision: weighted average of n-gram (often 1-4)
            - for n-gram (lets say 3), the precision is the prob of the occurrence of translated gram in the reference text.
        - not allowed for a match of two same translation with one reference (that will count 2 in the precision metric), to avoid the 'the the the the the' trick.
        - penalty on the translated length
    - BLUE is not the only answer in automatic evaluation of MT
        - TER, METEOR, MaxSim, SEPIA, RTE_MT, etc
        - TERpA is representative
        - still a developing domain

<br>

### **On Doing Research**

- set a benchmark
- train, dev, and test data
- evaluation metric
- model
    - learning rate
    - initialization
    - for RNNs
        - LSTM or GRU is fine for a start
        - initialize the recurrent matrices to be orthogonal 初始化递归矩阵为正交矩阵(AA^T=E)
        - initial other matrices with a mall scale
        - **initialize forget gate bias to 1**
        - use adaptive learning rate algorithms, e.g. Adam, AdaDelta
        - clip the norm for the gradient: 1-5 seems reasonable threshold when using Adam or AdaDelta.
        - either dropout only vertically or loop into using Bayes dropout. (Gal and Gahramani – not natively in PyTorch)