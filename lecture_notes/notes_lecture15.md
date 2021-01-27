# Lecture 15: Natural Language Generation

### Recap on NLG

- Natural Language Generation: any setting in which we generate new text.
- NLG is the subcomponent of: MT, Summerization, Dialogue(chit-chat and task-based), Freedom QA, Creative writing(story-telling, poetry-generation), image captioning.
- LM: the task of predicting the next word given the words so far.
- conditional LM: the task of predicting the next word given the words so far and also some other input x.
    - MT: given source sentence and the target sentence.
    - Summerization: given input text and the summerarized text.
    - Dialogue: given the dialogue history and the next utterance.
- the training of a RNN-LM
    - teacher forcing: in training time, we just feed the ground truth (of course the hidden state passed through the Encoder) as input into decoder on every time step.

        ![Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled.png](Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled.png)

- the testing of a RNN-LM: the decoding algorithms
    - decoding algorithm: used to generate text from your trained LM.
    - we have learnt: greedy decoding, and beam searching.
        - greedy decoding: pick the most probable word as the next time-step input.
        - beam search: find a high-probability sequenced by tracking multiple possible sequences at once. stop and choose the sequence with the highest probability when reaching the stopping criterion.
            - factoring in some adjuistment for length.
            - k possible solutions each time
- beam size k
    - the effect of changing k
        - small k: bias, 一错再错
        - large k: reduce some bias but more computationally expensive. Moreover, for NMT, increasing k too much may decrease BELU score (probably because of the short translations). For open-ended tasks like chit-chat dialogue, large k can make the output more generic and less relevant.

### Decoding Algorithms

- greedy decoding
- beam search
- sampling based decoding
    - pure sampling: on step t, randomly sample from the probability distribution tp obtain the next word.
    - top-n sampling: on step t, randomly sample from the distribution but restricted to the top-n most probable words.
    - increasing n tends to get more diverse/risky output, but decrease n tends to get more genetic/safe output.
- softmax temperature
    - not a decoding algorithm, but a method that can be used in a decoding algorithm.
    - Knowledge Distillation: [https://zhuanlan.zhihu.com/p/90049906](https://zhuanlan.zhihu.com/p/90049906)
        - train the model at a higher temperature where the softmax output is 'soft', and let the model learn with the original data/model/...
        - test the model at a lower temperature.
        - the model is supposed to learn the useful specific (rather than general) knowledge/features.
    - the original softmax function in predicting the next word from LM: $P_t(w)=\frac{exp(S_w)}{\sum_{w\in V}exp(S_w)}$
    - apply a temperature hyperparameter to the softmax: $P_t(w)=\frac{exp(S_w/t)}{\sum_{w\in V}exp(S_w/t)}$
        - raise the t, the P distribution will become more uniform (1/n,1/n,1/n,...,1/n), thus more diverse output.
        - lower the t, the P distribution will become more spiky (0,0,...,1,0,0,...), thus the output is less diverse.

### NLG Tasks and NN approaches

- summarization
    - task: given input text x, write a summary y which is shorter and contains the main inforamtion of x.
    - category: single-document, multi-document.
        - single document summarization:
            - datasets with source documents of different lenghts and styles:
                - Gigaword: first one or two sentences of a news article—>headline style. (sentence comprehension)
                - LCSTS (Chinese microblogging): paragraph—> sentence summary.
                - NYT, CNN/DailyMail: news article—>(multi) sentence summary.
                - Wikihow: full how-to article—>summary sentence.
    - two main strategies:
        - extractive summarization: select parts of the oricginal text to form a summary.
            - easier
        - abstractive summarization: generate new text using NLG techs.
    - pre-neural summarization: the history
        - mostly extractive
        - a pipeline
            - content selection: choose some sentences to include.
                - sentence scoring functions
                    - presence of topic keywords (tfidf)
                    - features such as the sentence position in the doc.
                - graph based algorithms: view the document as a set of sentences (nodes).
                    - edge weight may be sentence similarity, or the position or sth.
            - information ordering: choose an ordering of those sentences.
            - sentence realiztion: edit the sequence of sentences, like simplify, remove parts, fix continuity issues. (to make it more like a sentence)
        - evaluation: ROUGE
        - Recall-Oriented Understudy for Gisting Evaluation

            ![Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%201.png](Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%201.png)

            - based on n-gram overlap
            - no brevity penalty
            - based on recall while BLUE (MT) is based on precision. summarization focus more on how much important infromation does your summarization contains. (Venn for understanding)
                - F1 version of ROUGE is also useful
            - reported as different scores varied from ROUGE-1 (unigrams), ROUGE-2, ROUGE-L(longest common subsequence overlap).
                - BLUE is reported as the combination of the precisions.
    - neural summarization
        - Seq2Seq summarization
            - *A Neural Attention Model for Abstractive Sentence Summarization, Rush et al, 2015* [https://arxiv.org/pdf/1509.00685.pdf](https://arxiv.org/pdf/1509.00685.pdf)
            - single-document abstractive summarization is a translation task!
            - apply standard Seq2Seq+attention method to it!
        - more developments
            - easier to copy, but not too much
            - hierarchical/multi-level attention mechanism
            - more global/high-level content selection
            - using RL to directly maximize the ROUGE or other metrics
            - improving pre-neural ideas such as graph algorithms.
        - copy mechanisms
            - motivation: Seq2Seq+attention systems are good at writing fluenet output but bad at copying over details. e.g. rare words.
            - idea: use attention to enable the system to copy words and phrases easier.
            - applying both copying and generating gives a hybrid of extractive and abstractive approach.
            - variants:
                - *Language as a Latent Variable: Discrete Generative Models for Sentence Compression, Miao et al, 2016* [https://arxiv.org/pdf/1609.07317.pdf](https://arxiv.org/pdf/1609.07317.pdf)
                - *Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond, Nallapati et al, 2016* [https://arxiv.org/pdf/1602.06023.pdf](https://arxiv.org/pdf/1602.06023.pdf)
                - *Incorporating Copying Mechanism in Sequence-to-Sequence Learning, Gu et al, 2016* [https://arxiv.org/pdf/1603.06393.pdf](https://arxiv.org/pdf/1603.06393.pdf)
            - take an example: in this structure, there is a scala P_gen, determined by the attention context and the current decoder output. And the P_gen is the ratio of whether the output is generated or copied, the copied text is just the attention distribution from the encoder at this time step. In all, the attention mechanism kind of does 2 duties.

                ![Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%202.png](Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%202.png)

            - problem: end up copying too much and bad at overall content selection (especially when the input doc is long).
                - *Get To The Point: Summarization with Pointer-Generator Networks, See et al, 2017* [https://arxiv.org/pdf/1704.04368.pdf](https://arxiv.org/pdf/1704.04368.pdf)
        - global content selection strategy
            - motivation: in pre-nueral, there are two steps called content selection and summerization realization. But in attention+Seq2Seq system the two step is mixed up.
            - idea: we first select content then do copy mechanism to avoid too many copying problem.
            - bottom-up summarization
                - content selection stage: neural sequence-tagging model to tag words as include or don't include.
                - bottom up attention stage: the don't include sentences are masked and attention system is built on these 'include'-tagged sentences from bottom to up to generate summerizations.
            - effective! overal content captured with less copying.
        - RL
            - RL to optimize ROUGE-L
            - *A Deep Reinforced Model for Abstractive Summarization, Paulus et al, 2017* [https://arxiv.org/pdf/1705.04304.pdf](https://arxiv.org/pdf/1705.04304.pdf)
                - Blog post: [https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/](https://www.salesforce.com/products/einstein/ai-research/tl-dr-reinforced-model-abstractive-summarization/)
            - finding: RL has better ROUGE but lower human-judged scores.
            - $R_{lcs}=\frac{LCS(X,Y)}{m}$, where LCS is the longest common subsequence.
            - finding: hybrid model is better. (RL+ML)
- sentence simplification: related task: rewrite the source text in a simpler (sometimes shorter) way.
    - simple wikipedia: standard wiki—>simple version.
    - newsela: news article—>version wrote for children.
- dialogue
    - tasks
        - task-oriented dialogue
            - assistive: QA, recommendation, etc.
            - co-operative: two agents solve a task together through dialogue.
            - adversarial: two agents compete in a task through dialogue.
        - social dialogue
            - chit-chat: for fun or company
            - therapy/mental wellbeing
    - pre and post neural dialogue
        - due to the open-ended freeform NLG, pre-neural dialogue systems more often used predefined templates or restrictions.
        - 2015, NMT makes open-ended dialogue system reburn.
        - Seq2Seq based dialogue
            - early papers
                - A Neural Conversational Model, Vinyals et al, 2015[https://arxiv.org/pdf/1506.05869.pdf](https://arxiv.org/pdf/1506.05869.pdf)Neural Responding Machine
                - for Short-Text Conversation, Shang et al, 2015[https://www.aclweb.org/anthology/P15-1152](https://www.aclweb.org/anthology/P15-1152)
            - disadvantages: boring reaction (genericness), irrelevant reaction, repeat reaction, lack-of-context reaction.
            - irrelevant response problem: generate words irrelevant to the users
                - how irrelevant: it's generic ('i don;t know') or changing into sth unrelated.
                - solution: optimize for MMI(Maximum Mutual Infromation).
                - $MMI=log\frac{p(S,T)}{p(S)p(T)}$$MMI=log\frac{p(S,T)}{p(Sp(T))}$
            - Genericness/boring response problem: lack of **variety**.
                - easy test-time fixes
                    - directly upweight rare words during beam search
                    - use a sampling decoding algorithm rather than beam search
                    - softmax temperature
                - conditioning fixes
                    - train with conditioning the decoder with some additional content
                    - train a retrioeve-and-refine model rather than a generated-from -scratch model.
                        - 从语料库中采样人类语言并加以编辑来适应当前的场景
                        - 能让模型说出更加多样化且有趣的话
            - repetition problem
                - solution
                    - abandon n-grams in beam search (check if the same)
                    - design a coverage mechanism, an objective to avoid attention attending to the same words and causing repetition.
                    - RL can help maybe, if the objective is a non-differentiable function.
            - lack of consistent persona problem
                - personality embedding
                - *A Persona-Based Neural Conversation Model, Li et al* 2016, [https://arxiv.org/pdf/1603.06155.pdf](https://arxiv.org/pdf/1603.06155.pdf)
                - *Personalizing Dialogue Agents: I have a dog, do you have pets too?, Zhang et al, 2018* [https://arxiv.org/pdf/1801.07243.pdf](https://arxiv.org/pdf/1801.07243.pdf)
            - negotiation dialogue
                - *Deal or No Deal? End-to-End Learning for Negotiation Dialogues, Lewis et al, 2017* [https://arxiv.org/pdf/1706.05125.pdf](https://arxiv.org/pdf/1706.05125.pdf)
                - traditional LM is good at producing fluent dialogue but not strategic.
                - RL may help
- story-telling
    - generated on the basis of prompt
        - given an image (image captioning), brief writing prompt, or a given pre-story (story continuation).
    - *Generating Stories about Images,* [https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed](https://medium.com/@samim/generating-stories-about-images-d163ba41e4ed)
        - how to get around the lack of parallel data? this is not the classical supervised image captioning problem, no related data-pair to learn.
            - using a common sentence-encoding space
            - **skip-thought vector**, a common sentence encoding method. learn the embdding by objective of predicting the surrounding words.
                - *Skip-Thought Vectors, Kiros 2015,* [https://arxiv.org/pdf/1506.06726v1.pdf](https://arxiv.org/pdf/1506.06726v1.pdf)
            - use COCO dataset (image-headline dataset) to learn the skip-thought embedding mapping from image to headline.
            - use sth like Taylor Swift lyrics to train RNN-LM to decode skip-thought vector to the Taylor Swift-style text.
    - *Hierarchical Neural Story Generation, Fan et al, 2018* [https://arxiv.org/pdf/1805.04833.pdf](https://arxiv.org/pdf/1805.04833.pdf)
        - generating a story from a writing prompt
        - dataset: Reddit’s WritingPrompts subreddit. every story has a brief writing hint.
        - the model: a complex Seq2Seq promp-to-story model.
            - convolutional based
            - gated multi-head multi-scale self attention
                - self-attention: capturing long-range context
                - gates: selective attention
                - heads and scales: varieties
            - model fusion: training strategy
                - pre-train a Seq2Seq model and train another Seq2Seq using the hidden state of the first.
                - the first LM learns the general information of the prompt and the second mode make stories based on that prompt.
            - but: lack of coherence
    - challenges in storytelling
        - coherence
        - event driven: event2event generation
            - *Event Representations for Automated Story Generation with Deep Neural Nets, Martin et al, 2018.* [https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17046/15769](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17046/15769)
        - structural storytelling
            - *Strategies for Structuring Story Generation, Fan et al, 2019* [https://arxiv.org/pdf/1902.01109.pdf](https://arxiv.org/pdf/1902.01109.pdf)
            - tracking events, entities, state, etc, also world state, etc.
- poetry generation: Hafes
    - Hafes: a poetry system
    - idea: FSA based RNN-LM
        - *Generating Topical Poetry, Ghazvininejad et al, 2016* [http://www.aclweb.org/anthology/D16-1126](http://www.aclweb.org/anthology/D16-1126)
        - *Hafez: an Interactive Poetry Generation System, Ghazvininejad et al, 2017* [http://www.aclweb.org/anthology/P17-4008](http://www.aclweb.org/anthology/P17-4008)
    - Deep-Speare
- Non-autoregressive generation for NMT
    - generate translation parallelly.

### NLG Evaluation

- automatic evaluation metrics for NLG
    - n-gram overlap based: BLEU, ROUGE, METROR, F1, etc.
    - not suitable for NMT, or summarizqtion, worse on open-ended tasks dialogue, storytelling.
    - word overlap metrics are not relevant to human-judgements.
        - *How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation, Liu et al, 2017* [https://arxiv.org/pdf/1603.08023.pdf](https://arxiv.org/pdf/1603.08023.pdf)
        - *Why We Need New Evaluation Metrics for NLG, Novikova et al, 2017* [https://arxiv.org/pdf/1707.06875.pdf](https://arxiv.org/pdf/1707.06875.pdf)
- perplexity? doesn't tell anything about generation
- word-embedding similarity? doesn't correlate well with human judgements.
- define more focused automatic metrics such as fluency, correct style and diversity, etc.
- human evaluation: gold standard but slow and expensive.
    - detailed human evaluation system

        ![Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%203.png](Lecture%2015%20Natural%20Language%20Generation%20ed48fd0d6dcc433396a4cc213ef1c01e/Untitled%203.png)

    - possible new avenues for NLG eval?
        - corpus-level metrics: verses the example-level metrics.
        - adversarial discriminator
        - human eval for free: gamification.
        - metrics that measure the diversity-safety tradeoff

### NLG Future

- current trends in NLG
    - incorporating discrete latent variables into NLG
        - discrete latent variable: applications in VAE, GAN
        - discrete latent variable: multi-model system.
    - alternatives to strict left-to-right generation
        - parallel generation, iterative refinement, top-down generation for loger pieces of text.
    - alternative to maximum likelihood training with teacher forcing.
        - more sentece-level objectives
- future
    - transfer NMT to NMG
- tips
    - if using LM, improving LM may improve the quality.
    - human judgement and automatic metrics
    - reproduction-open source.