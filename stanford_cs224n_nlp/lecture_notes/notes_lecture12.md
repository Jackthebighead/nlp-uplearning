# Lecture 12: Subword Models

### **Linguistics**

- Phonology
    - 语音学
    - phoneme: 音素, the unique feature
    - of no actual meaning in linguistics
- Morphology: parts of words

    <img src="pics_lecture12/Untitled.png" width = "300" height = "200" alt="d" vertical-align=center />

    - 形态学，词法，形态论
    - morphemes: semantic unit
    - an easy alternative is to work with n-gram character:
        - Wickelphones
        - Microsoft's DSSM
        - related idea is to use convolutional layer
- words in writing system
    - writing systems vary in how to represent words
        - no word segmentation: Chinese
        - words segmented: words components sentences
    - the writing system is different for each language. The data is hard to obtain and process.
- models below the word level: challenges
    - need to handle large, open vocabulary
        - rich morphology, informal spelling, transliteration (many translation is based on spelling).

            <img src="pics_lecture12/Untitled 1.png" width = "400" height = "100" alt="d" vertical-align=center />


<br>

### **Purely character-level models**

- character-level models: 2 methods
    - word embeddings can be composed from character embeddings
        - generates embeddings for unknown words: OOV problem.
        - the similarity between characters: the spelling.
    - connected language can be processed as characters, that is, all languages can be build on top of character-sequences, no consideration of word-level.
    - remarks: both methods have been proven to work successfully.
        - the key is that a  phoneme/character can be grouped by deep learning models, which is not a semantic unit traditionally (the meaning of h, a and t can represent what a 'hat' means?).
        - Deep learning models can construct character-group meaning to represent the semantic.
- Purely character-level model: the second method.
    - 以字符作为输入和输出的机器翻译系统
    - we have seen one in the last lecture: use very deep convolutional network for text classification: Conneau, Schwenk, Lecun, Barrault.EACL 2017
        - powerful because of the stacked convolutional layers
    - purely character-level NMT models
        - not very good at first
        - decoders only works fine
        - promising results
            - Luong and Maning built a baseline character-level Seq2Seq LSTM NMT system, worked well. against word-level basedline.
                - but it was slow
            - fully character-level NMT without explicit segmentation

                <img src="pics_lecture12/Untitled 2.png" width = "500" height = "250" alt="d" vertical-align=center />

                - encoder contains ConvNets with segment embeddings and highway networks
                    - input: letter sequence of character embeddings
                    - text convolution layer, max pooling with stride 5, the output is a segment embedding matrix.
                      -  this is a char-sequence embedding not word
                    - multiple layers of highway network
                    - 理解：CNN的应用条件一般是要求卷积对象有局部相关性，而文本是满足的，之前我们用ConvNet作用于word级别的，可以有效识别关键phrases，而用在character-level就是可以识别word pieces信息，n-gram characters，有效构建character-level的LM&embeddings。
                - decoder is a **character-level** GRU
                    - the decoder is the key and the encoder didn't improve much
            - stronger character results with depth in LSTM Seq2Seq model
                - deeper stacked of LSTM
                - the improvement is bigger in complex languages like 捷克语, but less improvement in English and French.
                - **Word-level is good when model is small while Character-level is better when model is huge.**

<br>

### **Subword-models: Byte Pair Encoding and friends**

- two trends
    - same architecture as for word-level model: but use smaller units—word pieces.
    - hybrid architectures
        - main model has words, something else for characters
        - e.g. for some unknown words we use characters.
- byte pair encoding
    - aka. BPE, doesn't use any algorithms with Deep Learning.
    - originally a **compression** algorithm
        - idea: add the most frequent byte pair as a new byte
        - quite successful in representing pieces of words. 2016-18
        - details

            <img src="pics_lecture12/Untitled 3.png" width = "400" height = "200" alt="d" vertical-align=center />

            - a word segmentation algorithm: bottom up clustering. 分词算法
            - start with a unigram vocabulary of all (Unicode) characters in data
            - the most frequent n-gram pairs form a new n-gram
                - starts from bi-gram
                - clustering based on frequency of n-gram pairs
            - choose a vocabulary size and work until the vocabulary size meets.
            - split data with deterministic longest piece segmentation of word, based on the vocabulary, to form word pieces set.
            - run the MT system using word pieces (as if we are using words).
- Word piece/Sentence piece model by Google NMT
    - use a variant of BPE
        - word piece model
            - tokenizes inside words
        - sentence piece model
            - use the raw data
            - blank space is retained as a special tag.
            - *Subword Regularization: Improving Neural Network Translation Models
            with Multiple Subword Candidates, 2018, Google*
    - idea: rather than using character n-gram count, we can use a greedy approximation to maximizing language model log likelihood to choose the pieces by add n-gram that maximally reduces perplexity (a measure of LM).
        - that's how NN way does!
    - 2018 best WMT
    - BERT uses a variant of the word piece model
        - common words are in the vocabulary. e.g. 1910s, LeBron James.
        - other words are built from word pieces
            - non-initial word pieces are represented with two ##.
        - a large vocabulary, i mean, large. word pieces is a good way to increase the vocabulary. But BERT is not a word-level model.

<br>

### **Hybrid character and word level models**

- character-level to build word-level [LM]

    <img src="pics_lecture12/Untitled 4.png" width = "300" height = "300" alt="d" vertical-align=center />

    - *learning character level representations for POS tagging 2014*
    - convolution over characters to generate word embeddings
    - the difference between pure character model is that here we use character embedding to replace(or represent) word embedding. (still word embedding)
    - then we can use the model to do higher-level task like, fixed window of word embeddings used for POS tagging.
- character-based LSTM to build word representations. [LM]
    - on unknown words

        <img src="pics_lecture12/Untitled 5.png" width = "300" height = "300" alt="d" vertical-align=center />

        - Bi-LSTM
        - the training process: he input is each character, output the embeddings from LSTMs and the goal is to optimize the perplexity as a language model.
    - the whole architecture

        <img src="pics_lecture12/Untitled 6.png" width = "400" height = "200" alt="d" vertical-align=center />

- character-aware Neural Language Models
    - a more complex/sophisticated approach, *Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush. 2015.*
    - a combination of the previous 2
        - CNNs over characters in words are better at capturing features and generate character-level embeddings.
            - 理解：CNN的应用条件一般是要求卷积对象有局部相关性，而文本是满足的，之前我们用ConvNet作用于word级别的，可以有效识别关键phrases，而用在character-level就是可以识别word pieces信息，n-gram characters，有效构建character-level的LM&embeddings。
        - RNNs are better at building a LM with context informations
        - the difference between pure character-level model (also CNN_Encoder_RNN_Decoder based) is that here we use character embedding to replace(or represent) word embedding. (still word embedding)
    - motivation: derive a powerful robust language model across a variety of languages.
    - highlights: encode subword correlation, solve OOV (character-based), less parameters (CNN).
    - details
        - architecture

            <img src="pics_lecture12/Untitled 7.png" width = "400" height = "400" alt="d" vertical-align=center />

        - CNN layer

            <img src="pics_lecture12/Untitled 8.png" width = "400" height = "400" alt="d" vertical-align=center />

            - generate character-embedding (many features) for words
        - highway network

            <img src="pics_lecture12/Untitled 9.png" width = "400" height = "300" alt="d" vertical-align=center />

            - LSTM-liked structure in CNN
        - LSTM Decoder

            <img src="pics_lecture12/Untitled 10.png" width = "300" height = "150" alt="d" vertical-align=center />

            - word level
            - hierarchical softmax
            - truncated bp through time for training
    - results
        - comparing size, it is better than simple LSTM
        - comparing perplexity, it is better than word-level models
        - for vocabulary

            <img src="pics_lecture12/Untitled 11.png" width = "500" height = "400" alt="d" vertical-align=center />

            - using char, it is obvious that it learnt character-like things.
            - using highway structure, it learnt more 'meanings' than the previous just-CNN model.
                - because highway structure approximate RNN's function.
        - for OOV, char-model is way more better.

            <img src="pics_lecture12/Untitled 12.png" width = "500" height = "400" alt="d" vertical-align=center />

        - the paper also questioned the necessity of using word embeddings as inputs for neural language modeling since CNN+highway can extract rich semantic and structure information.
- Hybrid NMT: an application
    - a best of both worlds architecture: 2016
    - lots of translation need only word-level, we just need to enter character level when necessary (OOV, <UNK>).
    - architecture

        <img src="pics_lecture12/Untitled 13.png" width = "400" height = "300" alt="d" vertical-align=center />

        - loss at the word level and character level
        - 2-stage decoding: word-level beam search and char-level beam search
    - results: beating word-level and character-level model in NMTin 2016.

<br>

### **FastText**

- chars for word embeddings
    - using characters to build word-embeddings like W2V
    - a joint model for word embedding and word morphology (2016)
    - details

        <img src="pics_lecture12/Untitled 14.png" width = "300" height = "300" alt="d" vertical-align=center />

        - same objective as w2v but using characters
        - bi-LSTM to compute embedding
        - model attempts to capture morphology
        - model can infer roots of words
- FastText: subword model for word embeddings
    - *Bojanowski, Grave, Joulinand Mikolov. FAIR. 2016*
    - motivation: to generate word2vec library, more useful for oov and morphological words and languages.
    - details
        - the expansion of w2v using n-gram character based skip-gram model
        - represent words with n-gram characters with boundary notation.

            <img src="pics_lecture12/Untitled 15.png" width = "300" height = "50" alt="d" vertical-align=center />

            - $where=<wh,whe,her,ere,re>,<where>$
        - the objective (score) is the total score of components of words.
            - $S(w,c)=\sum_g\in G(w)Z_g^TV_c$
            - the sum of six results (six representations of center word)
            - rather than sharing representation for all n-grams (char-sequences), FastText uses 'hashing trick' to have fixed number of vectors to represent a word.
                - Fowler-Noll-Vo hashing function
    - measure: word similarity.