# Tranformer from scratch

this is a step by step implementation of Attention is all you need paper issued in 2017

# What did i add ?
1. Automatic mixed precision for the forward loop
2. The ability to subset the data and try on smaller subset not the entire dataset -> config ['data_subset_ratio']
3. added Pin_memory in datalaoder to True to enable faster data transfer to CUDA-enabled GPUs. -> reference ['https://pytorch.org/docs/stable/data.html']
4. in validation loop no_inference context manager was used to squeeze every bit of performance -> reference ['https://pytorch.org/docs/stable/notes/autograd.html#inference-mode']
5. With Adam optimizer i used fused kernel flag to speed up computation -> reference ['https://pytorch.org/docs/stable/optim.html']
