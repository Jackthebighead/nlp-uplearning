# Lecture 16 Coreference Resolution

### Coreference Resolution

- 共指消解
- task: identify all mentions that refer to the same real world entity.
- applications
    - full text understanding
    - machine translation: gender recognition or sth.
    - dialogue systems
- two steps
    - detect the mentions: easy.
    - cluster the mentions: hard.
        - coreferece resolution it's actually a clustering task

### Mention Detection

- mention: span of text referring to some entity
- categories and ways of detection
    - pronouns(I, you, it, he): POS tagger 词性标注
    - named entities(prople, places): NER
    - noun phrases(a dog lying besides a tree): Parser(especially a constituency parser).
- making all of the categories as mentions over-generates mentions
- problems: bad mentions
    - solution
        - train a classifier on spurious mentions
        - keep all mentions as candidates, discard all singleton mentions(ones that are not marked as coreference with anything else).
- optimization: the above is a pipeline system
    - that is, we train a classifier specifically for different mention category detection
    - can we do it end-to-end? not only solve all kinds of detections together but jointly do mention detection and coreference resolution together?
- Linguistics on 2 different terms
    - coreference: when two mentions refer to the sanme entity in the world.
    - anaphora: when a term (anaphora) refers to another term (antecedent).
        - anaphora is a word that doesn't have independent reference, determined by antecedent.

            ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled.png)

    - comparison

        ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%201.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%201.png)

        - not all noun phrases have reference 不是所有的名次都有所指代（现实世界的物体）
            - e.g. No dancer twisted her knee
            - no dancer refers to nothing
        - not all anaphoric relations are coreferential 不是所有的anaphoric都是共指的。
            - we may loose the restriction as bridging anaphora. e.g. We went to see **a concert** last night. **The tickets** were really expensive.

            ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%202.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%202.png)

        - cataphora
            - usually the antecedent comes before the anaphoric, looking backwards.
            - as a complementary, cataphora refers to 'after', 'looking forward' situations.
    - coreference models
        - classical rule-based (pronominal anaphora resolution)
            - Hobbs' algorithm

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%203.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%203.png)

                - coreference for pronouns
            - example

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%204.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%204.png)

            - this is only a baseline, we should improve to be better than it.
                - but it's still a dumb algorithms, can't get right answers sometime.

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%205.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%205.png)

            - knowledge-based coreference resolution
                - Winograd Schema

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%206.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%206.png)

                - the improvement should be done in the area of knowledge establishing.
                - as a way of measuring intellegence through turing test
        - mention pair
            - idea: train a binary classifier that assigns every pair of mentions a probability of being coreferent: $P(m_i,m_j)$
            - in this example, 'she' looks for every possible candidate antecedents and decide which are coreferent with it.

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%207.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%207.png)

            - training
                - cross entropy loss

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%208.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%208.png)

                - negative samples
                - traverse every mentions then traverse every previous mentions
            - testing (a clustering task but we only got scores)
                - pick a threshold and add coreference links
                - take the transitive closure to get the clustering

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%209.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%209.png)

                - but if one link is mistaken, may cause two clusters mistakenly clustered.
            - disadvantages of mention pair models: the model may predict all their upper-threshold candidate and add coreference links.
                - train the model to predict only one antecedent

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2010.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2010.png)

        - mention ranking
            - idea:

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2011.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2011.png)

                - assign each mention its highest scoring candidate antecedent according to the model.
                - for those who has no candidate antecedent, link them with NA mention.
                - we only get every antecedents' probabilities and we want to choose the best. We apply a softmax over the socres for candidate antecedents so probabilities sum to 1.
                - optimization of mention pair modeling
            - training

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2012.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2012.png)

            - testing
                - pretty much the same as mention-pair model except each mention is assigned only on antecedent based on softmax.
            - modeling: how to calculate the probability
                - non-neural statistical classifier
                    - feature engineering: proper features

                        ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2013.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2013.png)

                - simple neural network
                    - FFNN

                        ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2014.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2014.png)

                        - input: word embeddings and other features used in the statistical models.
                            - embedding: previous two words, first word, last word, head word, of each mention.
                                - head word can be found using a parser（类似与核心主语的东西）
                            - additional features: distance, document genre, speaker information (knowledge).
                - more advanced end-to-end model using LSTMs, attention
                    - *Kenton Lee et al. from UW, EMNLP 2017*
                    - a mention ranking model
                    - detail
                        - LSTM, attention, end to end, no separate mention detection and coreference resolution.
                            - then how to detect? considering every span of text directly.
                        - model

                            ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2015.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2015.png)

                            - word and char embedding
                            - Bi-LSTM
                            - span representation(sub-sequence)

                                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2016.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2016.png)

                                - $g_i=[x_{start},x_{end},\hat x_i,\phi(i)]$, accoding to the first and last word representation from hidden states of Bi-LSTM, attention-based average of the word embeddings in the span (to find the head word), and additional features.
                            - score: on every span

                                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2017.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2017.png)

                            - train end to end, the whole coreference system
                        - problems:
                            - intractable to make scores to every pair of spans (time complexity O(T^4) where T is the word num and O(T^2) for the span numbers)
                                - lots of pruning needed: mention detection again?
                            - Attention learns which words are important in a mention
        - clustering-based
            - coreference resolution is actually a clustering mission
            - we can use agglomerative clustring, bottom up

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2018.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2018.png)

                - use models to score which clusters merging is preferred
            - idea: mention pair decision is difficult while cluster pair decision is easier since there is more information in clusters.
            - *Clark & Manning, 2016*
            - details

                ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2019.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2019.png)

                - generate a mention pair representation, like from the hidden state of the FFNN model.

                    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2020.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2020.png)

                - then using pooling operation to represent the cluster pair
                - score each cluster pair candidate based on: $s(merge[c_1,c_2]=u^Tr_c(c_1,c_2))$ where r is the matmul, u^t is the weight matrix.
                - merge the proper scored clusters
                    - this merge operation is based on the last step of merging, so RL is used.

### Evaluation

- many metrics: MUC, CEAF, LEA< B-CUBED, BLANC, or (often) averaging the metrics.
- B-CUBED: for each metion, compute a precision and a recall, then average the indivisual Ps and Rs.

    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2021.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2021.png)

    - this metric is kind of tricky because you can assign which cluster should be compared with the gold cluster 1. (bipartite graph algos....)
- OntoNotes dataset: 3000 documents labeled by humans
    - both EN and CH
- performance

    ![Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2022.png](Lecture%2016%20Coreference%20Resolution%2060ead4aec6eb4f2683ffe56fc824905a/Untitled%2022.png)

- conclusion: a useful, challenging task.
- Try out a coreference system yourself
    - [http://corenlp.run/](http://corenlp.run/) (ask for coref in Annotations)
    - [https://huggingface.co/coref/](https://huggingface.co/coref/)