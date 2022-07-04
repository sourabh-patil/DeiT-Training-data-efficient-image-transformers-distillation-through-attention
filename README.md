# DeiT-Training-data-efficient-image-transformers-distillation-through-attention
PyTorch implementaion of Data-eifficient Image Transformers. I have written code for student network and teacher network sperately and then combined them together. Like to the paper: https://arxiv.org/pdf/2012.12877.pdf
The student network is a vision transformer which learns itself as well as takes valueable lessons from the teacher network. Techer network is a convolutional network which is pre-trained. Following is the overall block diagram shown in the paper. 


![deit_block](https://user-images.githubusercontent.com/53788836/177100007-0e5edf2e-70d9-4440-9bc0-eff20e1ba867.png)

