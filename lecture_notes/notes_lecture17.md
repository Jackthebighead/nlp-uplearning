# Lecture 17: Multitask Learning

- NLP history
    - ML with feature engineering
        - by hand
    - deep learning for feature engineering
        - word vectors, etc.
    - deep architecture engineering for single tasks
        - NER, NMT, etc.
    - multitask learning?
- multitask learning motivations
    - for more general AI, sharing knowledge is important. But models now are only partly pre-trained.
    - pre-training is great. In CV, most tasks are classification while NLP has no single blocking task. NLP needs induction on logics, languages, sentiments, visions, etc.
    - a **unified multi-task model** for NLP?
        - idea: the model should know how to do knowledge transfer (domain adaptation, share weights and zero-shot learning).
        - the objective of the model is to easily adapt to new tasks, lower the bar of solve new tasks, and potentially move towards continual learning. Also, to pretty complex tasks we still need to build thing around the core model.
            - It's sort of like the AI (NLP) core.
- multitask learning details
    - how to express NLP tasks in the same framework?
        - first we should category NLP tasks generally. NLP tasks can be classified into:
            - sequence tagging: NER, aspect specific sentiment.
            - text classification: sentiment classification, dialogue state tracking.
            - Seq2Seq: MT, summarization, qa.
        - next we should realize the super tasks in NLP:
            - LM: modeling on a language. predicting the next word.
            - QA system: input the question, output the answer.
                - may be the most important, and general TASK
            - Dialogue: generation.
    - The Natural Language Decathlon (decaNLP)

        <img src="pics_lecture17/Untitled.png" width = "600" height = "350" alt="d" vertical-align=center />


        <img src="pics_lecture17/Untitled 1.png" width = "600" height = "200" alt="d" vertical-align=center />

        - transfer 10 different tasks as **QA** **formulasm**, then train and test.
        - multitask learning as question answering
            - question-context-answer
            - meta supervised learning:
                - from {x,y} to {x,t,y}: t is the task.
                - use the question q as the natural description of the task.
                - y is the answer to q and x is the context necessary to answer q.
            - no task-specific modules, be able to adjust internally, and be open to zero-shot learning.
        - detail
            - start with a context
            - ask a question
            - generate the answer
                - pointer switch mechanism: for each output word, a pointer is choosing to point among {question, context, external information} to generate the answer.
        - MQAN (Multitask Question Answering Network): [www.decaNLP.com](http://www.decanlp.com/)

            <img src="pics_lecture17/Untitled 2.png" width = "600" height = "300" alt="d" vertical-align=center />

            - encoding:
                - fixed GloVe+character n-gram embedding
                - linear
                - shared Bi-LSTM with skip connection
            - co-attention: outer product between the two sentences and back again with skip connections.
                - to get questoin/context dependent contextual representation
            - Bi-LSTM (dimension reduction), 2\*Transformers, Bi-LSTM
                - only transformer layers is hard to optimize
                - have to have LSTMs
            - Auto-Regressive **Decoder**

                <img src="pics_lecture17/Untitled 3.png" width = "600" height = "400" alt="d" vertical-align=center />

                - 自回归编码器：一个句子的生成过程首先根据概率分布生成第一个词，然后根据第一个词生成第二个词，然后根据前两个词生成第三个词，以此类推，直到生成整个句子。
                - The autoregressive decoder uses fixed glove and character n-gram embedding as the initial answer input embedding. Then pass through two transformer layers and one LSTM layer to participate in the output of the last three layers of the encoder(i.e. the representation of the question and the context).
                - LSTM decoder state: the output of the decoder. then use the output to compute attention distributions with the context and question which are used as pointers.
                - there are three attentions: one for question attention, another for context attention and the other for (a softmax) the vocabulary of the context.
                - weighted sum then as the output attention (the weight is the pointer mechanism, i.e. select where should the pointer point)
                    - lambda decides whether to copy from context or question.
                    - gamma decides whether to copy from external vocabulary.
        - evaluation

            <img src="pics_lecture17/Untitled 4.png" width = "600" height = "400" alt="d" vertical-align=center />

            - performance

                <img src="pics_lecture17/Untitled 5.png" width = "600" height = "300" alt="d" vertical-align=center />

                - +self attention
                - +co attention
                - + question pointer == MQAN, pointing is important
                - transformer does help
                - the training results are bad at the beginning, but it will soon be better.
                - multitask helps zero shot learning
                - there are still gap between single model performance and multi version.
        - the training **strategies**
            - fully joint

                <img src="pics_lecture17/Untitled 6.png" width = "300" height = "200" alt="d" vertical-align=center />

                - works well
                - take a mini-batch on each task
            - anti-curriculum pre-training
                - works better but with difficulties
                - curriculumn learning: start training with simplest tasks and move on to the harder problems.
                - but in multi-task learning, we try to do the oppoiste: decreasing the order of difficulty.
                - it's actually intuitive because the harder tasks may let the model weights reach an optimal status and the simpler tasks can just fine-tune that. We can also veiw training the harder task as a kind of pre-training.(trainsfer learning as well).
                - performance
                    - improves a little but MT is still bad
            - closing the gap by improvoing the sys

                <img src="pics_lecture17/Untitled 7.png" width = "500" height = "500" alt="d" vertical-align=center />

                - CoVe: contextual vectors. using better word-representation model helps.
                - using more harder tasks (as the pre-training) in the first phase of anti-curriculumn training helps.
                - oversampling on the harder task (like ML) helps.
- performances
    - XAI
        - pointers

            <img src="pics_lecture17/Untitled 8.png" width = "600" height = "150" alt="d" vertical-align=center />

    - MQAN pretrained on decaNLP
        - outperforms on new tasks like ML, NER
    - zero-shot domain adaptation of pre-trained MQAN
        - without training, 80% accuracy on Amazon and Yelp reviews.
        - 62% on SNLI
    - zero-shot classification
        - no training on new tasks' dataset
- **decaNLP a benchmark for generalized NLP**
    - train single question answering model for multiple NLP tasks.
    - framework for tackling: NLU, multitask learning, domain adaption, transfer learning, pre-training, zero-shot learning...
- cross-task task?