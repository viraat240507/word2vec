def similarity(data, network, word1, word2):
    word1_l,word2_l=word1.lower(),word2.lower()

    if word1_l not in data.word2idx or word2_l not in data.word2idx: 
        raise ValueError("Both words are not present in the vocabulary.")

    word1_idx,word2_idx=data.word2idx[word1_l],data.word2idx[word2_l]
    
    def cosine_similarity(v1,v2):
        dot_product=torch.dot(v1,v2)
        norm1=torch.linalg.vector_norm(v1)
        norm2=torch.linalg.vector_norm(v2)
    
        return dot_product/(norm1*norm2).clamp_min(1e-12)

    # consider centre of word1 and context of word2
    w1_centre=network.W_centre.weight[word1_idx]
    w2_context=network.W_context.weight[word2_idx]
    score1=cosine_similarity(w1_centre,w2_context)

    # consider context of word1 and centre of word2
    w1_context=network.W_context.weight[word1_idx]
    w2_centre=network.W_centre.weight[word2_idx]
    score2=cosine_similarity(w1_context,w2_centre)

    cross_role_score=max(score1,score2)
    centre_score=cosine_similarity(w1_centre,w2_centre)

    return centre_score, cross_role_score