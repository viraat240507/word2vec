import torch
from torch import nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    # step 4: generate the vector embeddings for all words in the text corpus in the constructor for the ANN
    def __init__(self,vocab_size,embedding_dim=16):
        super().__init__()

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vocab_size<2:
            raise ValueError("The text corpus must have atleast two unique words, currently it does not.")

        if not isinstance(embedding_dim,int) or embedding_dim<1: 
            raise ValueError("Embedding dimension must be a positive integer.")

        # the table for vector embeddings has dimensions (n,d), where n is the number of words in the vocabulary and d is the dimension of the vector embedding for each word
        self.W_centre=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        self.W_context=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        with torch.no_grad():
            self.W_centre.weight.normal_(mean=0.0, std=0.01)       
            self.W_context.weight.normal_(mean=0.0, std=0.01)

    def forward(self, centre_id, context_id, negative_ids):
        centre=self.W_centre(centre_id) # dimensions:(,embedding_dim)
        context=self.W_context(context_id) # dimensions:(,embedding_dim)
        negatives=self.W_context(negative_ids) # dimensions: (k,embedding_dim)

        positive_score=torch.dot(centre,context)
        negative_score=negatives@centre

        positive_loss=-F.logsigmoid(positive_score)
        negative_loss=-F.logsigmoid(-negative_score).sum()

        return positive_loss+negative_loss

    # here, data parameter represents the entire word2vec class instance
    def fit(self,data,epochs=50,learning_rate=0.01):
        if not data.training_pairs:
            raise ValueError("Generate training pairs first.")

        if data.prob_dist is None:
            raise ValueError("Create the probability distribution first.")

        if epochs<1 or learning_rate<=0:
            raise ValueError("Enter valid positive values for learning rate and number of epochs.")

        optimizer=torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

        history=[]
        self.train()

        for epoch in range(epochs):
            total_loss=0.0

            # shuffle the indices of training pairs visiting order at every epoch
            order=np.random.permutation(len(data.training_pairs))

            for pair_idx in order:
                # extract training pairs one by one from data.training pairs
                centre_idx,context_idx=data.training_pairs[pair_idx]

                negative_indices=data.k_negative_samples(context_idx)

                # convert everything to PyTorch tensors for compatibility with PyTorch embedded layers
                centre_id=torch.tensor(
                    int(centre_idx), dtype=torch.long, device=self.device
                )

                context_id=torch.tensor(
                    int(context_idx), dtype=torch.long, device=self.device
                )

                negative_ids=torch.tensor(
                    negative_indices, dtype=torch.long, device=self.device
                )

                # clear the gradients from the previous step so that they do not interfere in training of the current batch
                optimizer.zero_grad()

                # calculate the loss
                loss=self.forward(centre_id, context_id, negative_ids)
                # since we have passed PyTorch tensors in the forward() function as input, the loss returned will also by a PyTorch tensor, hence we need to extract the num

                # calculate the gradients
                loss.backward()

                # update the word embeddings
                optimizer.step()

                total_loss+=loss.item()
                
            average_loss=total_loss/len(data.training_pairs)
            history.append(average_loss)

            print(f"Epoch: {epoch+1} : Loss: {average_loss:.4f}")

        return history
