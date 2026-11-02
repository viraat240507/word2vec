import numpy as np
import re
class word2vecDataset:
    def __init__(self, window_size=2, k=5):

        if not isinstance(window_size,int) or window_size<1: 
            raise ValueError("Window size must be a positive integer")

        if not isinstance(k,int) or k<1:
            raise ValueError("k must be a positive integer")
        
        self.window_size=window_size
        self.k=k

        self.word2idx={}
        self.idx2word={}
        self.vocab_size=0
        self.idx_corpus=[]
        self.prob_dist=None
        self.training_pairs=[]
        
    # step 1: text processing and vocabulary mapping
    def preprocess(self,corpus):
        # cleaning the corpus to remove all punctuations using regular expressions
        cleaned_corpus=[re.sub(r'[^a-zA-Z\s]',' ',sentence) for sentence in corpus]
        
        # produces list all words of the cleaned corpus in the original order as the corpus and converts them to lowercase
        vocab=" ".join(cleaned_corpus).lower().split() 
        
        uniq_vocab=list(set(vocab)) # extract the unique words only
        self.vocab_size=len(uniq_vocab)

        if self.vocab_size==0:
            raise ValueError("Corpus contains no valid words")
    
        # create two dictionaries: word2idx which maps unique words to indices and idx2word for the reverse mapping
        self.word2idx={word:idx for idx,word in enumerate(uniq_vocab)}
        self.idx2word={idx:word for idx,word in enumerate(uniq_vocab)}
    
        # now convert text corpus to integer IDs
        self.idx_corpus=[[self.word2idx[word] for word in sentence.lower().split()] for sentence in cleaned_corpus]
        
    
    # step 2: generating all pairs of words (centre_word,context_word) as training examples
    def generate_training_pair(self):
        # consider a window size of say, 2 to look atmost 2 words to the left and right of the centre word to capture the true relationship
        self.training_pairs=[]
        
        for sentence in self.idx_corpus:
            sent_len=len(sentence)
            
            for i in range(sent_len):
                start=max(0,i-self.window_size)
                end=min(sent_len,i+self.window_size+1)
    
                for j in range(start,end,1):
                    # capture the pair as (centre_word_idx,context_word_idx)
                    if j!=i: self.training_pairs.append((sentence[i],sentence[j]))
        
    
    # step 3: building the negative sampling distribution
    def prob_distribution(self):
    
        # counting frequencies of all words across the entire text corpus
        counts=np.zeros(self.vocab_size)
        for idx_sentence in self.idx_corpus:
            for word_idx in idx_sentence: counts[word_idx]+=1
    
        # applying the exponential scaling factor to the counts
        scaling_factor=0.75
        scaled_counts=np.power(counts,scaling_factor)
    
        # finding the probability distribution from the scaled counts
        self.prob_dist=scaled_counts/np.sum(scaled_counts)

    
    # this function generates k random noise samples for the centre word which are different from the exclude_idx word
    def k_negative_samples(self, exclude_idx):
        if self.prob_dist is None:
            raise ValueError("No probability distribution created yet, first call prob_distribution().")

        if exclude_idx<0 or exclude_idx>=self.vocab_size:
            raise ValueError("exclude_idx is outside the range of the vocabulary.")

        # Create a copy of the probability distribution created from the prob_distribution() function
        p=self.prob_dist.copy()

        # since the exclude_idx cannot be sampled as negative, assign zero probability to it
        p[exclude_idx]=0.0

        total=p.sum()
        if total<=0:
            raise ValueError("No eligible negative words to sample.")

        # normalize the remaining probabilities by dividing by the sum of all remaining probabilities
        p/=total

        return np.random.choice(
            self.vocab_size,
            size=self.k,
            replace=True,
            p=p
        )