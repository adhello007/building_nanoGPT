import torch 
import torch.nn as nn 
from torch.nn import functional as F

"""
If you take a token index, convert it into a one-hot vector 
(a vector of zeros with a "1" at the token's position),
 and pass it through an nn.Linear layer (with no bias), 
 the result is identical to nn.Embedding.$$OneHot(index) 
 \times WeightMatrix = Row(index)$$The reason we use nn.Embedding 
 for token tables instead of nn.Linear is efficiency. 
 In a vocabulary of 50,000 words, a one-hot vector has 49,999 zeros.
 Multiplying by those zeros is a waste of computer power. nn.Embedding skips the math and just fetches the row.
"""
#18000 MiB GPU VRAM
batch_size = 64 
context_size = 256 
max_iterations = 5000
eval_interval = 500
learning_rate = 1e-5
eval_iterations = 200
n_embed=384 
n_head=6 #number of self attention heads in parallel.   
n_layer = 16 #number of transformer layers. 
dropout = 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read()

torch.manual_seed(133)
print(f"A glimpse of our text: {text[:100]}")
print("===" * 20)
#lets get the unique characters and start building out the library 
chars = sorted(list(set(text)))
print(f"Total Unique Characters are: {chars}\n")
vocab_size = len(chars)
print(f"vocabulary size: {vocab_size}")

#Now lets have a simple mappoing of the characters and map 
#it against a value. 
stoi = {s:i for i,s in enumerate(chars)} #This acts as a lookup table. 
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] #Takes in a string, and outputs its integers 
decode = lambda l: ''.join([itos[i] for i in l]) 

#Lets use this representation and build our text in a tensor 

data = torch.tensor(encode(text), dtype=torch.long)
print("===" * 20)
print("\nThe input text seen as converted into a tensor data\n")
print(data.shape)
print(data[:100])
#Lets do a train-test split 
split_size = int(0.9*(len(data))) 
train_data =  data[:split_size]
val_data = data[split_size:]

#Create a function that randomly grabs a batch of data 
#from the training dataset. 


def get_batch(split: str, gpu=False): 
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,)) 
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    if gpu==True: 
        x, y = x.to(device=device), y.to(device=device)
    return x, y 


"""
So essentially, we are only mapping any input character to predict the next 
character that is in the dataset. This is what our entire goal would be. 

Now this relationship's name would be a Bigram model, because we look at the 
prev character to predict the next one. 
""" 
class Head(nn.Module): 
    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size)))  

        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        B,T,C = x.shape  

        k = self.key(x) #B,T,C
        q = self.query(x) #B,T,C

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #We are using head_size here instead of C because q and k are projected down. 
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) #we might have a smaller T than context-size 
        wei = F.softmax(wei, dim=-1) #against the channel dimension
        wei = self.dropout(wei) 
        #perform weighted aggregation of the values 
        v = self.value(x) # (B, T, head_size)
        out = wei @ v 
        return out 
    
class MultiHeadedAttention(nn.Module): 
    def __init__(self, num_heads, head_size): 
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) #This is the Wo Output Matrix.
        self.dropout = nn.Dropout(dropout) 
    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate against the channel dimension. 
        out = self.dropout(self.proj(out)) # instead of simply passing on concatenated individual head info, 
        #we use a simple projection layer so that the model can combine these individual infos more coherently. 
        return out

class FeedForward(nn.Module): 
    def __init__(self, n_embed): 
        super().__init__()  
        self.ff_head = nn.Sequential(
            nn.Linear(n_embed, n_embed*4), #in the paper its mentioned that the inner hidden layer is 4 times the input feature length.
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed), #This is also the projection layer(Wo) like our MultiHeadedAttention proj layer. 
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        return self.ff_head(x)

class TransformerBlock(nn.Module): 
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head 
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
    def forward(self, x): 
        x = x + self.sa(self.layer_norm1(x))   
        x = x + self.ffwd(self.layer_norm2(x))
        return x 
    """
    when i write x = x + self. sa(x), i create a two diverging paths:
        path a (Residual): pass x directly to the output unchanged.
        path b (Sub-Network): pass x through attention/linear layers to get updated values. 
    """
    
class BigramLanguageModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #B , T , C
        self.position_embedding_table = nn.Embedding(context_size, n_embed) #position matters only for the entire context_size. 
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) #B, T, vocab_size
        

    def forward(self, idx, targets=None):   
        """
        previously we used the nn.Embedding layer directly to calculate the raw scores, 
        which were then passed to cross_entropy to cal loss bw the pred and the target. 

        We are moving from a "Lookup Table" model to a "Neural Network" model.
        Earlier we calculated the next prediction of a character via the embedding column outputs (65)
        on which we would perform softmax. 

        Even after training, the current character can only predict its next character 
        as the weights from the training dataset (more like pattern recognition or statistical frequency.)

        It has no context of the previous characters that came before it, it simply 
        shouts, THIS IS THE NEXT CHARACTER GIVEN WE INPUT THIS CHARACTER, even when we train it for this task. 

        Now, we have split the process:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
            self.lm_head = nn.Linear(n_embed, vocab_size)

        When 43 ('a') comes in, we look up a vector of size 32. These 32 numbers are NOT predictions. 
        They are Attributes (or Features) of the character 'a'.

        This vector (B, T, n_embed) is the "Hidden State" (nn.embedding)
        It is the model's internal representation of the data before it tries to guess the answer.

        we have lm_head (Language Modeling Head). 
        Its job is to take those 32 attributes and translate them into 65 probabilities.

        We want to use Self-Attention,so we need a "workspace" where the tokens can talk to each other, 
        which we do in the hidden state of tok_embd (nn.Embedding) (internal features or representations.)
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #ints from 0 to T-1. #(T,C)

        x = tok_emb + pos_emb #(B,T,C) broadcasting rules work
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) #language modeling head. This is where they predict the next token. 

        if targets is None: 
            loss = None
        else: 
            B,T, C = logits.shape  
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets) #(B,C)
            
        return logits, loss


    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:] #We crop the idx to the context size because we 
            #only have the position embedding of the context_size
            logits, _ = self(idx_cond) 

            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B , C)

            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)

            idx = torch.cat((idx,idx_next), dim=1) #(B, T+1)
        return idx

#Lets run inference on it to find out how random initializations 
#of the embeddings give a loss. 

xb, yb = get_batch('train', gpu=True)
m = BigramLanguageModel()
m = m.to(device=device)
logits, loss =  m(xb, yb) 
print("===" * 20)
print(f"\nThe LOSS for a standard UNTRAINED MODEL OF RANDOM INITILAIZATIONS: \n")
print(logits[0].shape)
print(logits[0])
print(loss)


print("===" * 20)
print("Lets try and generate some text from it. ")
print("Generating the first 100 characters from the above model\n")
print("We start from a single character which means batch =1, means one single story that we want from the author")
print(",the first character from our vocab itos[0]")
print("===" * 20)
idx = torch.zeros((1,1), dtype=torch.long) #(B = how many decoding texts in parallel do we want to decode. T = What start character from itos[index] should index be.)
#print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
idx = idx.to(device=device)
generated_batch = m.generate(idx, max_new_tokens=100)

#itterate through each row in the batch
for i in range(generated_batch.shape[0]):
    row = generated_batch[i].tolist()
    print(f"--- STORY {i+1} ---")
    print(decode(row))

print("===" * 20)
print("its gibberish because the lm_heads, attention heads etc are all randomly initialized right now, but we can train the model ")
print("===" * 20)
print(f"Now its time to TRAIN THE ACTUAL MODEL AND LOWEST BOUND FOR IT IS THE STATISTICAL FREQUENCY OF IT.(count each occurence)")

@torch.no_grad() #No storing a computational graph for it since we wont call backprop on these variables. 
def estimate_loss(): 
    out = {}
    m.eval()
    for split in ['train','val']: 
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations): 
            X, Y = get_batch(split, gpu=True)
            logits, loss = m(X,Y)
            losses[k] =  loss.item()
        out[split] = losses.mean()
    m.train()
    return out

final_loss = 100 #usually it should be MAX_INT

def train_bigram(m): 
   # m = m.to(device=device) #it is already now being shifted to the gpu post initialization
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    global final_loss
    print("===" * 20)
    print("The LOSS IS ")

    for step in range(max_iterations): 
        #cal and display loss on train and val data. 
        if step % eval_interval ==0: 
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < final_loss: 
                final_loss = losses['val']                
        
        #sampling from the train  data  
        xb,yb = get_batch('train', gpu=True)

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True) #make optimizer's gradients stored from prev loop = 0.  
        #if this line is followed by a backward pass, .grads are guaranteed to be None for params that did not receive a gradient.
        #it  skips the update for gradients whose update is none. PyTorch simply deletes the gradient tensor
        loss.backward()
        optimizer.step()
choice = input("IF YOU WANT TO TRAIN THE MODEL, PRESS 1. OR PRESS 0 TO EXIT\n")
if choice==0: 
    print("EXITING THE PROCESS")
else: 
    print("TRAINING THE MODEL ====================")
    train_bigram(m)


print("===" * 20)
print("LETS RUN INFERENCE ON THIS")
idx = torch.zeros((1, 1), dtype=torch.long)
idx = idx.to(device=device) #imp since model is in the GPU.
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))
print("===" * 20)
print("Lets save this model if we feel its loss is good.")

if final_loss < 2.7:
    torch.save(m.state_dict(), f'trained_model/bigram_model_loss_{final_loss:.3f}.pth')
    print(f"Saved a model to disk because its loss is {final_loss}")

