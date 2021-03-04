# Lecture 16 Coreference Resolution

### **Coreference Resolution**

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

<br>

### **Mention Detection**

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

            <img src="pics_lecture16/Untitled.png" width = "300" height = "50" alt="d" vertical-align=center />

    - comparison

        <img src="pics_lecture16/Untitled 1.png" width = "500" height = "250" alt="d" vertical-align=center />


        - not all noun phrases have reference 不是所有的名次都有所指代（现实世界的物体）
            - e.g. No dancer twisted her knee
            - no dancer refers to nothing
        - not all anaphoric relations are coreferential 不是所有的anaphoric都是共指的。
            - we may loose the restriction as bridging anaphora. e.g. We went to see **a concert** last night. **The tickets** were really expensive.

            <img src="pics_lecture16/Untitled 2.png" width = "400" height = "250" alt="d" vertical-align=center />


        - cataphora
            - usually the antecedent comes before the anaphoric, looking backwards.
            - as a complementary, cataphora refers to 'after', 'looking forward' situations.
    - coreference models
        - classical rule-based (pronominal anaphora resolution)
            - Hobbs' algorithm

                <img src="pics_lecture16/Untitled 3.png" width = "500" height = "350" alt="d" vertical-align=center />


                - coreference for pronouns
            - example

                <img src="pics_lecture16/Untitled 4.png" width = "500" height = "300" alt="d" vertical-align=center />


            - this is only a baseline, we should improve to be better than it.
                - but it's still a dumb algorithms, can't get right answers sometime.

                    <img src="pics_lecture16/Untitled 5.png" width = "500" height = "100" alt="d" vertical-align=center />

            - knowledge-based coreference resolution
                - Winograd Schema

                    <img src="pics_lecture16/Untitled 6.png" width = "500" height = "100" alt="d" vertical-align=center />


                - the improvement should be done in the area of knowledge establishing.
                - as a way of measuring intellegence through turing test
        - mention pair
            - idea: train a binary classifier that assigns every pair of mentions a probability of being coreferent: $P(m_i,m_j)$
            - in this example, 'she' looks for every possible candidate antecedents and decide which are coreferent with it.

                <img src="pics_lecture16/Untitled 7.png" width = "400" height = "200" alt="d" vertical-align=center />


            - training
                - cross entropy loss

                    <img src="pics_lecture16/Untitled 8.png" width = "400" height = "200" alt="d" vertical-align=center />


                - negative samples
                - traverse every mentions then traverse every previous mentions
            - testing (a clustering task but we only got scores)
                - pick a threshold and add coreference links
                - take the transitive closure to get the clustering

                    <img src="pics_lecture16/Untitled 9.png" width = "400" height = "200" alt="d" vertical-align=center />


                - but if one link is mistaken, may cause two clusters mistakenly clustered.
            - disadvantages of mention pair models: the model may predict all their upper-threshold candidate and add coreference links.
                - train the model to predict only one antecedent

                    <img src="pics_lecture16/Untitled 10.png" width = "400" height = "200" alt="d" vertical-align=center />


        - mention ranking
            - idea:

                <img src="pics_lecture16/Untitled 11.png" width = "400" height = "300" alt="d" vertical-align=center />


                - assign each mention its highest scoring candidate antecedent according to the model.
                - for those who has no candidate antecedent, link them with NA mention.
                - we only get every antecedents' probabilities and we want to choose the best. We apply a softmax over the socres for candidate antecedents so probabilities sum to 1.
                - optimization of mention pair modeling
            - training

                <img src="pics_lecture16/Untitled 12.png" width = "400" height = "200" alt="d" vertical-align=center />


            - testing
                - pretty much the same as mention-pair model except each mention is assigned only on antecedent based on softmax.
            - modeling: how to calculate the probability
                - non-neural statistical classifier
                    - feature engineering: proper features

                        <img src="pics_lecture16/Untitled 13.png" width = "400" height = "250" alt="d" vertical-align=center />


                - simple neural network
                    - FFNN

                        <img src="pics_lecture16/Untitled 14.png" width = "500" height = "300" alt="d" vertical-align=center />


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

                            <img src="pics_lecture16/Untitled 15.png" width = "400" height = "250" alt="d" vertical-align=center />


                            - word and char embedding
                            - Bi-LSTM
                            - span representation(sub-sequence)

                                <img src="pics_lecture16/Untitled 16.png" width = "500" height = "100" alt="d" vertical-align=center />


                                - $g_i=[x_{start},x_{end},\hat x_i,\phi(i)]$, accoding to the first and last word representation from hidden states of Bi-LSTM, attention-based average of the word embeddings in the span (to find the head word), and additional features.
                            - score: on every span

                                <img src="pics_lecture16/Untitled 17.png" width = "300" height = "250" alt="d" vertical-align=center />


                            - train end to end, the whole coreference system
                        - problems:
                            - intractable to make scores to every pair of spans (time complexity O(T^4) where T is the word num and O(T^2) for the span numbers)
                                - lots of pruning needed: mention detection again?
                            - Attention learns which words are important in a mention
        - clustering-based
            - coreference resolution is actually a clustering mission
            - we can use agglomerative clustring, bottom up

                <img src="pics_lecture16/Untitled 18.png" width = "400" height = "300" alt="d" vertical-align=center />


                - use models to score which clusters merging is preferred
            - idea: mention pair decision is difficult while cluster pair decision is easier since there is more information in clusters.
            - *Clark & Manning, 2016*
            - details

                <img src="pics_lecture16/Untitled 19.png" width = "400" height = "300" alt="d" vertical-align=center />


                - generate a mention pair representation, like from the hidden state of the FFNN model.

                    <img src="pics_lecture16/Untitled 20.png" width = "300" height = "250" alt="d" vertical-align=center />


                - then using pooling operation to represent the cluster pair
                - score each cluster pair candidate based on: $s(merge[c_1,c_2]=u^Tr_c(c_1,c_2))$ where r is the matmul, u^t is the weight matrix.
                - merge the proper scored clusters
                    - this merge operation is based on the last step of merging, so RL is used.

<br>

### **Evaluation**

- many metrics: MUC, CEAF, LEA< B-CUBED, BLANC, or (often) averaging the metrics.
- B-CUBED: for each metion, compute a precision and a recall, then average the indivisual Ps and Rs.

    <img src="pics_lecture16/Untitled 21.png" width = "400" height = "300" alt="d" vertical-align=center />


    - this metric is kind of tricky because you can assign which cluster should be compared with the gold cluster 1. (bipartite graph algos....)
- OntoNotes dataset: 3000 documents labeled by humans
    - both EN and CH
- performance

    <img src="pics_lecture16/Untitled 22.png" width = "400" height = "300" alt="d" vertical-align=center />


- conclusion: a useful, challenging task.
- Try out a coreference system yourself
    - [http://corenlp.run/](http://corenlp.run/) (ask for coref in Annotations)
    - [https://huggingface.co/coref/](https://huggingface.co/coref/)