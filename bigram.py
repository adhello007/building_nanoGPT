import torch 


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
print("\nThe input text converted into a tensor data\n")
print(data.shape)
print(data[:100])

#Lets do a train-test split 
split_size = int(0.9*(len(data))) 
train_data =  data[:split_size]
val_data = data[split_size:]

#Create a function that randomly grabs a batch of data 
#from the training dataset. 
batch_size = 32 
context_size = 8 

def get_batch(split: str, gpu=False): 
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,)) 
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    if gpu==True: 
        x, y = x.to(device='cuda'), y.to(device='cuda')
    return x, y 


"""
So essentially, we are only mapping any input character to predict the next 
input character that is in the dataset. This is what our entire goal would be. 

Now this relationship's name would be a Bigram model, because we look at the 
prev character to predict the next one. 
"""

import torch.nn as nn 
from torch.nn import functional as F

class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):   
        """
        idx is not a single character. 
        idx is exactly the tensor x (or xb) that comes out
         of your get_batch function.

         tensor([[24, 43, 58,  5, 57,  1, 46, 43],   # Row 0
        [44, 53, 56,  1, 58, 46, 39, 58],   # Row 1
        [52, 58,  1, 58, 46, 39, 58,  1],   # Row 2
        [25, 17, 27, 10,  0, 21,  1, 54]])

        B (Batch) = 4: There are 4 rows.

        T (Time/Context) = 8: There are 8 columns (integers) in each row.

        They are 32 independent prediction problems
          packed into a (4, 8) grid for convenience.
        """
        logits = self.token_embedding_table(idx)

        if targets is None: 
            loss = None
        else: 
            B,T, C = logits.shape #We get the 
            logits = logits.view(B*T, C) #32 X 65 
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets) #(B,C)
            
        return logits, loss


    def generate(self, idx, max_new_tokens): 
        for _ in range(max_new_tokens): 
            logits, _ = self(idx) 

            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B , C)

            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)

            idx = torch.cat((idx,idx_next), dim=1) 
        return idx

#Lets run inference on it to find out how random initializations 
#of the embeddings give a loss. 

xb, yb = get_batch('train')
m = BigramLanguageModel(vocab_size)
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
print(", the first character from our vocab itos[0]")
print("===" * 20)
idx = torch.zeros((1,1), dtype=torch.long) #(B = how many decoding texts in parallel do we want to decode. T = What start character from itos[index] should index be.)
#print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
generated_batch = m.generate(idx, max_new_tokens=100)

#itterate through each row in the batch
for i in range(generated_batch.shape[0]):
    row = generated_batch[i].tolist()
    print(f"--- STORY {i+1} ---")
    print(decode(row))

print("===" * 20)
print("its gibberish because a bigram is a dumb model, but we can train the model ")
print("===" * 20)
print(f"Now its time to TRAIN THE ACTUAL MODEL AND LOWEST BOUND FOR IT IS THE STATISTICAL FREQUENCY OF IT.(count each occurence)")
choice = input("IF YOU WANT TO TRAIN THE MODEL, PRESS 1. OR PRESS 0 TO EXIT\n")
if choice==0: 
    print("EXITING THE PROCESS")
else: 
    print("TRAINING THE MODEL ====================")
    m = m.to(device="cuda")
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size =32
    print("===" * 20)
    print("The LOSS IS ")

    for steps in range(10000): 
        #sampling from the train  data  
        xb, yb = get_batch('train',gpu=True)

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True) #make optimizer's gradients stored from prev loop = 0.  
        #if this line is followed by a backward pass, .grads are guaranteed to be None for params that did not receive a gradient.
        #it doesnt do torch.optim for skips the update for gradients whose update is none. PyTorch simply deletes the gradient tensor
        loss.backward()
        optimizer.step()
        if steps % 100 == 0:
            print(loss.item())


print("===" * 20)
print("LETS RUN INFERENCE ON THIS")
idx = torch.zeros((1, 1), dtype=torch.long)
idx = idx.to(device='cuda') #imp since model is in the GPU.
print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))
print("===" * 20)
print("Lets save this model if we feel its loss is good.")

current_loss = loss.item()
if current_loss < 2.7:
    torch.save(m.state_dict(), 'trained_model/bigram_model_shakespeare.pth')
    print(f"Saved a model to disk because its loss is {current_loss}")

