# Lecture 19&20: Bias in AI, Future in AI

### **Bias in Data**

<img src="pics_lecture19&20/Untitled.png" width = "600" height = "350" alt="d" vertical-align=center />

- data
    - reporting bias: what people shared is not the true reflection of the real world.
    - selection bias: selection doesn't mean the randomness of samples.
    - out-group homogeneiry bias: people tend to see outgroup members as more alike than ingroup members when comparing attitudes, values, personality traits, and other characteristics.
- interpretation
    - confirmation bias: people tend to find supportive information.
    - overgeneralization: overfitting
    - correlation fallacy: 相关性谬误, causal.
    - automation bias: human bias in automation process

        <img src="pics_lecture19&20/Untitled 1.png" width = "600" height = "350" alt="d" vertical-align=center />

- bias can be good bad and neutral
- measuring algorithmic bias
    - confusion matrix: recall and precision.
    - false positives might be better than false negatives

### **What we can DO**

- Data really really matters
    - understand your data: skews and correlations
    - abandon single training-set/test-set form simialr distribution.
    - combine inputs from multiple sources
    - leave held-out test-set
- mathine learning
    - de-biasing: remove signal for problematic output: racism etc.
    - inclusion:
        - multi-task learning to increase inclusion
    - equality of opportunity in supervised learning
        - e.g. toxic text, 'you are a gay' gets pretty higher toxic score because some text containing gay is toxic. But it's not actually right. we need to deal with these unintended bias.
        - measuring: synthetic datasets: bias madlibs
            - causal analysis
    - measuring model performances
        - How good is the model at distinguishing good from bad examples? (ROC-AUC)
        - low subgroup performance
- ethical AI

### **Future AI**

- big success these years
    - deep learning: the ability of scaling
    - big success in DL:
        - image recognition: imageNet: 14 million examples.
        - mathine translation: WMT: millions of sentence pairs.
        - game playing: atari games, alphago
- NLP datasets
    - even english, most tasks have 100k or less labeled data.
    - increasingly popular solution: use unlabeled data.
    - using **unlabeled** data in MT
        - pre-training technique
        - we can train two pre-trained model for the two languages in MT. as Encoder and Decoder.
        - problems: no interaction between the two languages.
        - sol: self-training: label unlabeled data to get training set with noise

            <img src="pics_lecture19&20/Untitled 2.png" width = "600" height = "350" alt="d" vertical-align=center />

        - better sol: back-translation.

            <img src="pics_lecture19&20/Untitled 3.png" width = "600" height = "350" alt="d" vertical-align=center />

            - no longer circular
            - models only receive bad inputs not bad (input, label) training data.
            - training details
                - train two model on labeled data
                - use one model to label unlabeled data
                - back translation it
                - repeat
            - results: transformer+back-translation better than transformer+positional embeddings
            - what if there is no bilingual data?
                - we do word translation instead of sentence translation.
    - unsupervised word translation
        - cross lingual word embeddings
            - two languages share the embedding space
            - in this space the English word and its corresponding German word is closed, so we just need to search the closest German word.
        - want to learn from monolingual (单语言的) corpora
            - we noticed that the embedding structure of different languages may be the same, so we can learn a transfoamtion function on one language embedding to another.
                - learning method: adversatial training.
                    - discriminator: predict if an embedding is from Y or is a transformed embedding Wx from x.
                    - the goal is to train our W so that the discriminator gets **confused**.
    - unsupervised MT
        - same Encoder-Decoder model
        - details

            <img src="pics_lecture19&20/Untitled 4.png" width = "600" height = "350" alt="d" vertical-align=center />

            - initialize with cross-lingual word embeddings
            - feed Encoder with a cross-lingual embedding (French or English)
            - for Decoder, use a tag to tell the model which language to output.
        - training: 2 training objectives
            - objective 1: de-noising auto encoder (DAE)

                <img src="pics_lecture19&20/Untitled 5.png" width = "600" height = "200" alt="d" vertical-align=center />

                - input a scrambled sentence(e.g. en), and output the original sentence.
                - no attention here, the AE learns the latent representations of the sentence.
            - objective 2: back translation (with unlabeled data)
                - first translate fr to en, as the label of the next step.
                - and back translate en to fr using supervised example
                - train the model so that even with bad inputs, we still can return the original sentence.
            - why does it work?

                <img src="pics_lecture19&20/Untitled 6.png" width = "600" height = "200" alt="d" vertical-align=center />

                - the starting point: cross-lingual embeddings for **initialization**
                    - cross-lingual makes two english-french word looks similar in embedding vector.
                - shared encoder
                    - the encoded representation of a french sentence should be very similar to the corresponding english sentence.
                    - so, when put the french cross-lingual representation to the encoder, the model should output the same result as the input is english.
            - intuitively, the latent output of the shared encoder is a language-free representations of semantics.
            - results of unsupervised MT

                <img src="pics_lecture19&20/Untitled 7.png" width = "600" height = "300" alt="d" vertical-align=center />

                - better than supervised MT when the amount of training examples is small.
                - attribute transfer

                    <img src="pics_lecture19&20/Untitled 8.png" width = "600" height = "250" alt="d" vertical-align=center />

                - not so fast, and not well on totally different languages.
            - cross-lingual BERT
                - details

                <img src="pics_lecture19&20/Untitled 9.png" width = "600" height = "350" alt="d" vertical-align=center />

                - original BERT: predict masked words
                - masked multi-lingual LM: mask some words in a (en, fr) translation pair.
        - huge model and GPT-2
            - scale up the unsupervised models

                <img src="pics_lecture19&20/Untitled 10.png" width = "600" height = "350" alt="d" vertical-align=center />

                - the trend of AI: scaling

                    <img src="pics_lecture19&20/Untitled 11.png" width = "600" height = "350" alt="d" vertical-align=center />

                    - better hardware
                    - data and model parallelism
            - openAI
            - huge models in CV: GPipe, etc.
            - GPT-2:
                - a big transformer LM
                - trained on 40GB of text
                - what can GPT do?
                    - LM
                    - zero-shot learning: no supervised training data, no intended training, ask the model to generate from a **prompt**.
                        - reading comprehension: input <context><question><notation of solving reading comprehension>
                        - summarization
                        - translation
                        - qa
            - applications
                - ML: only given english corpus
                    - because datasets contains translation examples
                - QA: 4% accuracy by 1% of simple baseline..
            - what if the model gets bigger? unclear.
    - social impact of NLP
        - high-impact decisions
        - chatbots
    - future research on NLP
        - BERT and next
            - size
            - death of architecture engineering?

                <img src="pics_lecture19&20/Untitled 12.png" width = "600" height = "350" alt="d" vertical-align=center />

        - harder NLU
            - reading comprehension
                - longer documents, multiple documnets
                - multi-hop reasoning
                - dialogue
            - key problem of reading comprehension dataset: people write questions based on the context of the corpus
                - tend to ask simple questions
            - QuAC: question answering in context

                <img src="pics_lecture19&20/Untitled 13.png" width = "600" height = "350" alt="d" vertical-align=center />

                - dialogue between a student who ask questions and a teacher who answers.
                - teacher sees the wiki docs, student doesn't
                - still far from human performance
            - HotPotQA

                <img src="pics_lecture19&20/Untitled 14.png" width = "500" height = "350" alt="d" vertical-align=center />

                - designed to require multi-hop reasoning: to look for multiple docs for answering a question.
                - still a gap between models and human performances.
        - multi-task learning: DecaNLP, etc.
            - BERT+multitask: train BERT in a multi-task way can also work.
        - low-resource setting
            - another area: models that don't require lots of compute power
            - especially important for mobile devices
        - interpreting and understanding models: XAI
            - method: diagnostic and probing classifiers: train a classifier to tell whether a representation is good. for example.
        - NLP in history: **dialogue, healthcare**