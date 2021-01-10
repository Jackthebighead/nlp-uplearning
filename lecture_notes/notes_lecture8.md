# Lecture 8: Machine Translation, Sequence-to-Sequence and Attention

### **New Task: Machine Translation**

- MT: a NLP task relies on sequential output. taking a sentence in one language as input and outputting the same sentence in another language.
- Pre-neural MT: SMT
    - early-time machine translation: rule based
    - statistical MT (SMT)
        - the task: translating one language to another, e.g. from French to English.
            - learn a probabilistic model from data, $argmax_y(P(y|x))$, where y is English and x is French.
        - details
            - use Bayes rule: $argmax_y(P(x|y)P(y))$ to break down into 2 components.
                - the first part is called the translation model, which describes how words should be translated, it is learnt from parallel data.
                - the second part is called the language model (like the ones we learned from the previously), it describes how the word are represented in a specific language, it is learnt from monolingual data.
            - learn a translation model based on the translation pairs
            - how to learn P(x|y)?
                - we can view this problems as how to learn P(x,a|y) where a is the alignment.
                    - alignment: the relationship of translation in a word-level, can be one-one, can be one-many or many-one as well.

                        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled.png)

                    - to learn alignments, we need learn many things like: a particular word corresponding to a particular word or a particular word corresponding to its fertilities.
                - how to compute or optimize probability?
                    - the computation of the probability is way too expensive. image that we need to find all the possible solutions for possible alignments to complete a translation.
                    - heuristic search algorithm to search for the best translation is one solution. it discards hypothesis with low probability.
                    - this process is called Decoding
                        - it can be viewed as search a tree of hypothesis and do pruning to find the best path.

                        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%201.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%201.png)

        - remarks on SMT: it is a complex system and a huge research field. the cost is its main problem, it may need huge amount of human effort to do things like feature engineering or adding extra knowledge.
- Neural MT: NMT
    - do MT using NN
    - Seq2Seq: (2 RNNs)
        - it's a network structure, a conditional language model. conditional means it needs information from Encoder (the first RNN) to the second RNN (Decoder). aka. Encoder-Decoder models.
        - it can be useful in many NLP tasks like: dialog, summarization, parsing and code generation.
        - details

            ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%202.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%202.png)

            - **Encoder**
                - produce an encoding of the sentence, a fixed-size "context (higher-level meaning) vector".
                - in practice the Encoder usually consists of stacked RNNs like LSTM where the input of the next layer is the output of the current layer.
                    - Seq2Seq may do sth strange: like process the input in reverse order, this is because by doing this, the first output of Decoder will pay more attention on the last Encoder hidden states which will make it easier for Decoder to get started to generate the correct translation at a higher prob.

                        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%203.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%203.png)

            - **Decoder**
                - the Decoder is more complex, the Decoder should learn the ability to generate sentences (a LM!) based on the Encoder output as the initial hidden state.
                - a conditional language model, generates target sentences conditional on the encoded context.
                - both Encoder and Decoder are trained in the same time.
                - how to run Decoder? an <EOS> tag token is appended at the end of the input (as the sign of output generation), then we run the stacked layers of RNNs (LSTM in the pic downside), softmax the output as the translation. (also an <EOS> as the end of the output). the output is feed back in to the network as the input in the next time step, and repeated...
                    - after translating the whole sentence, we define a loss (averaging total CE loss, or sth) and bp it, and learn.

                    ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%204.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%204.png)

            - in the test time (generate translation), noticed that in the pic, the output of Encoder is the (last) hidden state vectors, so the input hidden state in Decoder is the output of the encoder.
                - the final output is all the hidden state of the Decoder (let's view as every output is one word in the translation sentence).
                - in Decoder, **the output of the previous time step is the current input.**
            - NMT: $P(y|x)=P(y_1|x)P(y_2|y_1,x)...P(y_t|y_{t-1},...,y_2,y_1)$
                - in this way we can directly learn P(y|x), no Bayes, so NMT is simpler and easier.
        - how to train?

            ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%205.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%205.png)

            - a parallel corpus: English-French pair.
            - get target sentence as labels, input into Decoder, the loss function is the negative log probability of the correct word, the probability distribution is under the hidden state (output by the Encoder), then average the total loss as the Loss and bp it and learn.
                - end to end: bp throughout the whole system, directly optimize the system as a whole.
                - another notion is that pre-trained model is not end to end which is trained separated with encoder and decoder.
        - decoding in test? (generating)
            - exhaustive search: compute every possible translation, choose with the sequence generated with accumulated highest probability.
            - ancestral sampling: sample the predicted words based on the conditional prob of the past word. it works well but has a high variance and low performance.
            - greedy decoding: generate the argmax word every time step in Decoder. stops when model produces <END> tag.

                ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%206.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%206.png)

                - problems: no draw back: 一错再错. (if the argmax output as the input of the next time step).
                - solutions
                    - exhaustive search: try computing all possible sequences, far too expensive, although this method guarantees to find the optimal one.
                    - beam search: choose k candidates based on the scores (5 to 10 usually).

                        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%207.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%207.png)

                        - not guarantees to find the optimal one
                        - the criterion of when to stop searching
                            - when hypothesis produce an <END>, stop search the current path
                                - when to produce the <END> tag is trained in LM
                            - when to stop the whole process? when time step reaches a threshold, or when the hypothesis amount reaches a threshold.
                        - how to select the highest score?
                            - why this question? the shorter hypothesis may have a lower score, which may be biased.
                            - solution: normalization by the length. $\frac1t\sum_{i=1}^{t}$
            - improvement
                - one of them: ATTENTION
    - variants of NN structure in NMT
        - Bidirectional RNNs

            ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%208.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%208.png)

            - the final hidden state of Encoder is: $h=[h^{forward},h^{backward}]$
    - advantages in NMT
        - better performance: fluent LM, better context extraction, similarity measure based on word expression.
        - end 2 end: convenient
        - less human effort: e.g. feature engineering
    - disadvantages in NMT
        - less interpretable
        - difficult to control
    - evaluation in NMT
        - BLEU (Bilingual Evaluation Understudy)
            - it compares the translation of NMT with human translations.
            - the similarity score is based on: N-gram precision. e.g. 2-gram: [a,b] with [a_1,b_1].
            - it is useful but it is not a good evaluation for situations like when there are different ways of translation on one sentence.
    - difficulties in NMT
        - large vocabulary
            - simplify softmax: hierarchical softmax, negative sampling
            - reducing vocabulary: a naive way is to limit the vocabulary size use <UNK> to replace words that never seen in the vocabulary. another way is to partition the training data into subsets which have |V'| target words. (V'<V), treat every subset as a mini-batch subset. When testing, need to find targets out of subset vocabulary using the idea negative sampling.
        - out of vocabulary words: handling unknown words: **OOV problem**
            - happens under situations like using subset to reduce vocabulary. One solution is that to learn to copy words using Attention.
        - domain mismatch: like wiki words with oral words
        - maintaining context over longer text
        - low resources pairs
        - explainability
        - bias of translation
        - hard to capture common senses

<br>

### **Attention Mechanism**

- Definition
    - the problem on Seq2Seq: bottleneck problem
        - only input the last hidden state to the Decoder, the input may lose information of the whole source sentence.
    - solution: Attention
    - on each step of the Decoder, use direct connection to the Encoder to focus on a particular part of the source sequence.
- Details:

    ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%209.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%209.png)

    - for the first output in Decoder, dot product it with all hidden state in Encoder and pass the results through a softmax to get a probability distribution. The distribution is the attention weight and then the we weighted sum with all hidden outputs of Encoder to get the attention output. Finally, we concatenate (or do another operation) with the attention output and the Decoder output as the final output.
        - the attention output mostly contains information from the hidden states that received high attention.
    - in equations
        - we have hidden states in Encoder: $h_1,h_2...h_N$
        - at the time step t, we have the hidden output of Decoder $s_t$
        - the attention score at this time step is: $e^t=[s_t^Th_1,...,s_t^Th_N]$
        - the attention weight is: $a^t=softmax(e^t)$
        - the attention output: $a_t=\sum_{i=1}^N a^t_ih_i$
        - the final output: $y_t=[a_y;s_t]$
    - advantages
        - performance: focus on certain parts of Encoder, good on long sentences.
        - solves the bottleneck problem.
        - solves the gradient vanishing problem.
        - interpretable: actually learn the **alignments** by self.
            - the attention-based model learns to assign significance to differ-
            ent parts of the input for each step of the output. In the context of
            translation, attention can be thought of as "alignment.
    - more general definition of Attention
        - the query-key-value model: the query is the word vector in the target (Decoder), and we view the word vectors in the source (Encoder) is a key-value pair where the key equals to the value (set different name just to distinguish the function in attention calculation).
            - so the attention process is: use query and key to calculate the attention weight, and then multiply the result with the value to calculate the attention output.

        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2010.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2010.png)

        - self-attention: self-attention measures the importance of the other words in the source with the current word in the source. it can be also described as the query-key-value model.
            - the self-attention process is: use query and key to calculate the **self-attention** weight to represent the importance of the other words towards the key, and then multiply the result with the value to calculate the **self-attention** output.
            - there is a equation: $Attention=Softmax(\frac{QK^T}{d_k^{\frac12}})V$
                - when the calculating method is dot product and there is a scaling process before softmax, it is called **scaled dot-product attention.**
        - self-attention can be viewed as a specialty of attention where the target is the source so as to learn the self-representation well.
        - self-attention can be used to language representation, pre-trained language model, etc. attention can be used in tasks like NMT.
    - Bahdanau et al. NMT model, using attention in NMT task
        - paper: *Neural Machine Translation by Jointly Learning to Align and Translate*
- Variants
    - in conclusion: the attention process can be summarized as computing:
        - attention score
        - attention weight
        - attention output
        - concatenate as the output at this time step
    - change the calculation of attention score. from dot product to
        - multiplicative attention: $e_i=s^TWh_i$, where W is a learnable parameter matrix.
        - additive attention: $e_i=v^T*tanh(W_1h_i+W_2s)$, where v is a weight vector, this process may change the dimensionality, and the dimension is the hyper-parameter.

<br>

### **Material 1: other models in NMT using Attention**

- Luong et al. NMT model
    - paper: *Effective Approaches to Attention-based Neural Machine Translation*
    - different attention mechanisms
        - global attention: in the Bahdanau et al. model, the classical attention uses concatenate the attention output and the original Decoder output as the final output $\hat h_i=f([h_i,c_i])$ (where h_i is the hidden state in Decoder and c_i is the attention output), which exists a problem of coverage. To issue this, Luong et al. uses the method of input-feeding, that is, use the $\hat h_i$ as the input of Decoder rather than the final output.

            ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2011.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2011.png)

        - local attention: Luong et. al's model generate an align position (which means where the output word in Decoder may have alignments in the input sequence), and it uses a window to compute the context vector (only compute the attention weight&output within the window).
            - in general, only focus on a small subset, reduce the drawback of global attention which is expensive.

            ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2012.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2012.png)

            - the window: [position-D,position+D].

- Google's new NMT
    - multilingual NMT
    - rather than maintain a model for one particular language, google trained a model for translation between any two languages.
    - zero-shot translation: can do translation even if we don't have translation data.
    - the key idea is that, the process of decoding is not particular to a specific language but to learn a representation of sentence from input/output.
- more advanced papers using attention
    - *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*
        - 这篇论文是提出单词/图像对齐的方法
    - *Modeling Coverage for Neural Machine Translation*
        - 他们的模型使用覆盖向量，考虑到使用注意力的历史以帮助计算将来的注意力。
    - *Incorporating Structural Alignment Biases into an Attentional Neural Translation Model*
        - 这篇论文通过引入其他传统语言学的思想来提高注意力表现。

<br>

### **Material 2: word and character-based models**

- motivation: copy is not enough in solving out of vocabulary problem, we can turn to sub-word structure. one trend is to use Seq2Seq architecture based on characters, another is to use hybrid architecture of word and character.
    - OOV problem
- word segmentation
    - Sennrich et. al proposed a method that transform a rare or unknown word as a sequence of subwords. And apply it to MT.
    - paper: *Neural Machine Translation of Rare Word*
    - the key algorithm is called Byte Pair Encoding, it is a compression algorithm. it starts at character vocabulary, and make statistics on frequency n-gram character pairs and add to vocabulary.

        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2013.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2013.png)

- character-based model
    - Ling et. al proposed a model based on characters for open-vocabulary representations on words.
    - paper: *Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation*
    - the basic idea is: use the embedding of characters instead of words as the input of BiLSTM models. the output is a word whose word embedding vector is made up by an affine transformation of back and forward directions.
- hybrid NMT
    - Luong et al. proposed a hybrid word-character model to deal with unknown words and achieve open-vocabulary NMT.
    - paper: *Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models*
    - the system mainly translate at word level, and look up character-level information when there are rare words.
        - word-based translation as a backbone: a deep LSTM encoder-decoder that translates at the word level. We maintain a vocabulary of size |V| (which is smaller) per language and use <UNK> to represent out of vocabulary words.
        - source character-based representation: aims at the input OOV words, we learn a deep LSTM model over characters of rare words, and use the final hidden state of the LSTM as the representation for the rare word.
        - target character-level generation: aims at the generation of OOV words, we have a separate deep LSTM that "translates" at the character level given the current word-level state.
            - whenever the word- level NMT produces an <unk>, the character-level decoder is asked to recover the correct surface form of the unknown target word.

        ![Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2014.png](Lecture%208%20Machine%20Translation,%20Sequence-to-Sequenc%201180e68bf97d418a8b8bfe18d2ba3f86/Untitled%2014.png)