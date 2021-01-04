# Lecture 5: Dependency Parsing

- ### **Syntactic Structure**
    - why we need sentence structure
        - to interpret languages correctly
        - previously we have word vectors to represent word with similarity in vector distances, now we need more complex meanings are conveyed by sentence structure consists of words.
        - need to know what connects (modifies) to what.
    - two views of linguistic structure
        - phrase structure
            - aka constituency parsing, context free grammars (CFG), 短语结构树，上下文无关语法。
            - organizes words into nested constituents
            - Intuitively, CFG disassemble sentences into a tree structure using grammar symbolization (POS: part of speech 词性). To be specific, leaves in the tree correspond to every word in the sentence while nodes in the tree correspond to phrase.
            - The idea is that sentences are constructed using progressively nested units

            <img src="pics_lecture5/Untitled.png" width = "400" height = "150" alt="d" align=center />

        - dependency structure
            - shows the relationship of which words (modify or are arguments of) depend on which other words
            - 依存树
            - advantages: compared with phrase structure, dependency structure directly shows the relationship between words, what's more, in the tree structure, the nodes corresponding to relationships between words thus it is more simple than phrase structure.

           <img src="pics_lecture5/Untitled 1.png" width = "400" height = "150" alt="d" align=center />

        - challenges in sentence syntactic structures
            - prepositional phrase attachment ambiguity
                - 介词短语依附歧义
                - e.g. San Jose cops kill man with knife: man with knife or kill with knife?
            - prepositional attachment ambiguities multiply
                - 依附歧义叠加
                - the key idea is to make decisions on which part attaches which part.
                - it might be exponentially calculations on the possible structure (dependency assignment).

                    <img src="pics_lecture5/Untitled 2.png" width = "400" height = "150" alt="d" align=center />

            - coordination scope ambiguity
                - 对等范围歧义
                - e.g. Shuttle veteran and longtime NASA executive Fred Gregory appointed to board: 'and' may be interpreted to different meanings.
            - adjectival modifier ambiguity
                - 形容词修饰语歧义
                - e.g. Students get first hand job experience: is hand an adj (dependent) for experience or adj for job?
            - verb phrase attachment ambiguity
                - 动词依存歧义
                - Mutilated body washes up on Rio beach to be used for Olympic beach volleyball: is be used.. depends on body or beach?
            - conclusion: **dependency paths identity semantic relations.**

                <img src="pics_lecture5/Untitled 3.png" width = "400" height = "200" alt="d" align=center />

- ### **Dependency Parsing and Treebanks**
    - dependency parsing
        - dependency is the dependent relationship (normally binary asymmetric relations (arrows)) between lexical items (words).
        - dependency tree graph
        - start from the head and, point to the dependent (the pointed one depends on the origin, e.g. the objective depends on the verb so there is an arrow from the verb to the objective).
        - usually add a root to make every non-fake word depends on sth.
    - tree banks
        - Universal Dependency: the rise of annotated data
            - a general description/rule on dependency for every human language, **treebanks.**
            - building a treebank is not good in practice
            - however, it provides dataset for the later ML related work
    - on building a dependency parser
        - characteristics on dependency
            - bilexical affinities: the possibility of two words have dependency
            - dependency distance: the distance of word dependency
            - intervening material: e.g. ','
            - valency of heads: the number of dependencies
        - constituting process: attaching each word to its dependents.
        - constraints: root has only one dependent and no cycles.
        - whether arrows can cross or not: **projectivity**
            - projectivity: there are no crossing dependency arcs when the words are laid out in their linear order, with all arcs above the words.
            - CFG doesn't allow it, while it is allowed in the dependency structure to capture complex information.
    - methods of dependency parsing
        - dynamic programming
            - O(n^3), heads at the end rather than in the middle.
        - graph algorithms
            - MSTParser
            - capture global information
            - measure the relationship with words to a value as the edge weight in a tree and find the MST. the final path is the path of dependency parser.
        - constraint satisfaction
            - pruning on hard constraints.
        - transition-based parsing
            - or called deterministic dependency parsing
            - locally based
- ### **Transition-based dependency parsing**
    - greedy transition-based parsing: Nivre's algo

        <img src="pics_lecture5/Untitled 4.png" width = "300" height = "150" alt="d" align=center />

        - a simple form of greedy discriminative dependency parser
        - the parser does a sequence of bottom up actions
            - kinda like shift&reduce parser, the reduce here is to choose an action from a set of actions (left-arc, right-arc, etc.).
        - the parser has
            - a stack to store words being processing at the moment with a ROOT in the bottom
            - a buffer to store the sentence, starts with the input sentence
            - a set of actions and a set of dependency arcs.
        - the algorithm

            <img src="pics_lecture5/Untitled 5.png" width = "300" height = "100" alt="d" align=center />

            - when fetching one from the buffer and compared it with the one in the top of the stack, there are 3 transition between states: shift means there may be no dependency so we move on to the next, left-arc and right-arc means there are dependency.
            - it can be regarded as the multiple classification task
            - the algorithm stops when the buffer is empty and only ROOT in the stack
    - arc-standard transition-based parser

        <img src="pics_lecture5/Untitled 6.png" width = "300" height = "200" alt="d" align=center />

        - another transaction solution
        - **only build projective dependency trees**
            - the algorithm can't capture non-projective arcs.
            - solve non-projectivity: add post-processor, add transitions to make model at least most non-projective(sort or sth), moving to mechanism that has no constraints on projectivity like graph-based.
    - MaltParser: ML way of dependency parsing
        - how to decide which action to perform in the next step?
            - **Machine Learning**
            - features: top of the stack word, POS, first in the buffer word, POS, etc.
            - no searches like beam search
                - beam search: find b candidate for each step. but in the next round there are b*(size) data.
                - can keep k good parse results as prefixes at each step, it is helpful for the quality but there is a tradeoff between quality and the (speed, storage, and computation resources).
            - it provides very fast linear time parsing, achieve great performance.
        - how to represent the features?

            <img src="pics_lecture5/Untitled 7.png" width = "300" height = "200" alt="d" align=center />

            - indicator features: many characteristics, and combinations of characteristics.
            - map indicator features to binary sparse numeral vectors
            - for ML classification: LR, SVM, etc.
    - evaluations

        <img src="pics_lecture5/Untitled 8.png" width = "300" height = "150" alt="d" align=center />

        - simply count the correct arcs
        - use unlabeled attachment score: UAS = #correct_dependencies/#dependencies
        - use labeled attachment score: LAS = #correct_dependencies/#dependencies considering labels (dependency type).
- Neural way of dependency parsing
    - why NN way?
        - traditional ML creates sparse data
        - indicator features is incomplete
        - the computation is complex
    - solution: NN
        - using distributed representation
            - word embedding on words
            - embedding on data like POS and dependency labels
        - details
            - extract a set of tokens based on the stack and buffer positions
            - vectorize them (feature, word, POS, dependency labels)
            - concatenate as the final feature map
        - NN only controls what decisions to make next, the parser structure is remained.
        - NN structure

            <img src="pics_lecture5/Untitled 9.png" width = "400" height = "200" alt="d" align=center />

        - improvements/optimizations
            - go wide and deep on NN and fine-tuning
            - beam search
            - global CRF to interfere decision sequence
        - SyntaxNet, the Parsey McParseFace model
        - graph-based dependency parsers
            - compute a score for every possible dependency for each edge, add an edge from each word to the highest scored candidate head. repeat til convergence.
            - **A Neural graph-based dependency parser,** Manning
                - great results but slower
                - neural sequence-based scoring model