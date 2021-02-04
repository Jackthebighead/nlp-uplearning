# Lecture 18: Tree RNN, Constituency Parsing and Sentiment

### **Movitation: Compositionality and Recursion**

- semantic in NLP
    - word vectors (char vectors)
    - more than word vectors: phrases, etc.
        - people interpret the meaning of larger text units like entities, descriptive terms, facts, arguments, stories, by semantic **composition** to smaller elements.
- view: languages is recursive.

    <img src="pics_lecture18/Untitled.png" width = "600" height = "350" alt="d" vertical-align=center />

    - e.g. noun phrase containing a noun phrase containing a noun phrase...
    - it's arguable but recurrency is good at describing languages.
- constituency parsing
    - Penn-Treebank: dataset
    - semantics: building on phrases in a compositional manner.
        - how to build?
            - i.e. how to map phrases into a vector space?

                <img src="pics_lecture18/Untitled 1.png" width = "600" height = "350" alt="d" vertical-align=center />

    - the difference between vanilla RNN?

        <img src="pics_lecture18/Untitled 2.png" width = "400" height = "350" alt="d" vertical-align=center />
        - RNN: doesn't give the meaning of phrases inside the sentence.

<br>

### **Structure Prediction with simple Tree RNN: Parsing**

- RNN for structure prediction
    - inpus: 2 candidate children's representation

        <img src="pics_lecture18/Untitled 3.png" width = "400" height = "300" alt="d" vertical-align=center />

    - the prediction NN outputs
        - the composition score (合理程度)
        - the representation of composition
    - the model: concatenate the two candidates, pass through sth like FFNN, and get the representation, and output the score scala by multiplying with another weight matrix.

        <img src="pics_lecture18/Untitled 4.png" width = "600" height = "300" alt="d" vertical-align=center />

    - details
        - first parsing each word-pair

            <img src="pics_lecture18/Untitled 5.png" width = "600" height = "350" alt="d" vertical-align=center />

        - second, greedy parsing

            <img src="pics_lecture18/Untitled 6.png" width = "600" height = "350" alt="d" vertical-align=center />

        - third, form a tree

            <img src="pics_lecture18/Untitled 7.png" width = "600" height = "350" alt="d" vertical-align=center />

    - loss: max-margin loss
        - score: $s(x,y) = \sum_{n\in node(y)}s_n$ where x is the sentence and y is the deterministic tree.
        - objective: $J=\sum_{i}s(x_i,y_i)-max_{y\in A(x_i)(s(x_i,y)+\Delta(y,y_i))}$, max-margin loss (the score we get from our structure search should be closs to the label).
        - greedy way, others? Beam search.
- scene parsing: images pieces composition.

    <img src="pics_lecture18/Untitled 8.png" width = "600" height = "350" alt="d" vertical-align=center />

    - multi-class segmentation

        <img src="pics_lecture18/Untitled 9.png" width = "600" height = "350" alt="d" vertical-align=center />

        - using RNN to recognize segments in one image and recognize them.

<br>

### **BP through Structure**

- same structure as bp
  
<br>

### **More complex TreeRNN units**

- discussion: simple **TreeRNN**
    - there are no interaction between inpus. (just concatenate, no actual modeling on interactions)
    - the combination function is the same for all kinds of combinations (suntactc **categories**, punctuations).
    - complexity. simple TreeRNN can capture some feaures but we definately need more complex structure to capture more complex features.
- version 2: syntactically united RNN: **SU-RNN**

    <img src="pics_lecture18/Untitled 10.png" width = "600" height = "250" alt="d" vertical-align=center />

    - CFG(context-free grammer)
    - every node and sequence has a category of a symbolic context free grammer. there is a matrix for symbolic combination as well.
    - have different weight matrix rather than a universal one
    - better semantic representation
    - problem: speed
        - solution: compute score only for a subset of trees coming from a simpler and faster model (PCFG) i.e. use PCFG to predict the candidate.
        - compositional vector grammar = **PCFG + TreeRNN**
- version3: compositionality through reversive matrix-vector spaces
    - *Socher, Huval, Bhat, Manning, & Ng, 2012*
    - before the probability: $p = tanh(W(c_1,c_2)+b)$, c_1 and c_2 have not relation.
    - idea: maybe we can make composition function more powerful by untying the representation of candidates.
        - i.e. every word, every phrases is represented by a weight matrix and a vector
    - detail

        <img src="pics_lecture18/Untitled 11.png" width = "600" height = "350" alt="d" vertical-align=center />

        - matrix-vector RNNs: **MV-RNN**

            <img src="pics_lecture18/Untitled 12.png" width = "600" height = "350" alt="d" vertical-align=center />

            - the combination: the vector A multiplies matrix B, the vector B multiplies matrix A, concatenate to get the logit.
            - good: non-linearity, better semantic composition
            - bad: huge parameters, and the way of doing matrixes is not necesarrily good.
    - version 4: Recursive Neural Tensor Network: **RNTN**
        - *Socher, Perelygin, Wu, Chuang, Manning, Ng, and Potts 2013*
        - less parameters than MV-RNN
        - sentiment detection
            - dataset: Standford Sentiment Treebank. Better dataset helps all models. Treebank is more powerful than just labels.
            - motivation: we also need a stronger model.
        - details

            <img src="pics_lecture18/Untitled 13.png" width = "600" height = "350" alt="d" vertical-align=center />

            - idea: allow both additive and mediated multiplicative interactions of vectors.
            - the tensor: a three-dimensional array, the first two dimension is for bi-linear attention-like multiplication between the two input vectors, and the extra dimension is the word-vectors.
            - use resulting vectors in tree as input to a classifier like LR for sentiment detection/classification (1/0).
        - results
            - good at capturing negating negatives meaning. e.g. not bad = pretty good.(不恰当)
    - version 5: **TreeLSTM**
        - *[Tai et al., ACL 2015; also Zhu et al. ICML 2015] Improving Deep Learning Semantic Representations using a **TreeLSTM**.*
        - details
            - an understanding of LSTMs

                <img src="pics_lecture18/Untitled 14.png" width = "600" height = "350" alt="d" vertical-align=center />

            - tree structure LSTM

                <img src="pics_lecture18/Untitled 15.png" width = "600" height = "350" alt="d" vertical-align=center />

            - a closer look

                <img src="pics_lecture18/Untitled 16.png" width = "600" height = "350" alt="d" vertical-align=center />

### Other uses of Tree-RNN

- **Tree-to-Tree NN for program translation**
    - *Chen, Liu, and Song NeurIPS 2018*
    - translate codes in a tree-structure to a different langugage

    <img src="pics_lecture18/Untitled 17.png" width = "600" height = "350" alt="d" vertical-align=center />

    - attention can be great here
