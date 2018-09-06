# Variational autoencoder for protein sequences

This repository provides code to accompany the paper:

Greener JG, Moffat L and Jones DT, Design of metalloproteins and novel protein folds using variational autoencoders, 2018 - [link to pre-print](https://arxiv.org/abs/1806.09900)

The work describes a variational autoencoder that can add metal binding sites to protein sequences, or generate protein sequences for a given protein topology.

## Getting Started
This instructions will get a copy of this repo on your system and running so you can produce sequences with our models or expand on our work. These files have been cleaned and minimized so they can be run using python with as few dependencies as possible.

### Dependencies
The follow are the packages needed to run our software. Our software uses python and leverages several python 
* [python 3.6](https://www.python.org/)
* [pytorch 4.0](https://pytorch.org/)
* [sklearn 0.19.1](http://scikit-learn.org/stable/)
* [numpy 1.14.1](http://www.numpy.org/)
* [lark-parser 0.4.1](https://github.com/lark-parser/lark)

Follow the links to install the correct version of all of the packages. Note that installing pytorch can be a bit tricky but all of these packages can be easily install with [Anaconda](https://anaconda.org/). Using the `conda install` command from the Anaconda package is highly recommended. The lark-parser package is only needed if you wish to generate protein sequences using the model that takes in a strucure grammar to produce sequences (described more below). The neural networks are built using pytorch and some utility functions from sklearn are used for metrics, making proper installation critical. 

If you wish to extend our work note that we have configured the general use scripts to run solely on cpu as they do not require the computational power of a GPU. However, the example training scripts provided can be run on GPU and contain switches and/or command line arguments to enable this. 

### Installation

Installing our software is incredibly easy. Navigate to the directory you wish to store the software in and simply run the following from the command line. 
```console
user@computer:~$ git clone git@github.com:psipred/protein-vae.git
user@computer:~$ cd protein-vae/produce_sequences/
```
You are now in the directory to run the scripts to produce sequences. Producing sequences is also very simple and is described below. 

## Running the software
Before using our software please read the paper as linked to at the top of these instructions. There are three different methods of producing sequences that this software provides:
1. **Seq-to-Seq**: Providing an initial sequence to return a similar sequence with some variation
2. **Seq-to-Metal-Seq:** Providing an initial sequence that does not bind to a metal and returning a the same sequence with variation that is more likely to bind a specified metal. 
3. **Grammar-to-Seq:** Providing a grammar string (see the paper) and producing a sequence that is likely to fold to the topology described in the grammar string. 

Each one of these methods is run with its own python script. Before going any further make sure you are in the [produce_sequences](produce_sequences/). Below we have provided examples for running each one of the scripts and what the input and output should look like. We have also provided example sequences (as found in the [examples](produce_sequences/examples/) directory). 

If you run one of the scripts without providing a sequence it will default use one of the example sequences. Below are examples of how to use each one of the scripts with the default examples. 

### 1. Seq-to-Seq Example
**1. Input File Preparation**

Make sure you have a text file or fasta file with a single sequence in it. For example, in the [seq2seq_example.txt
](produce_sequences/examples/seq2seq_example.txt) you'll find the following: 
```
AEVPSGEQLFNSNCSACHIGGNNVIISHKTLRKEALEKYAMNSLEAIRYQVVNGKNAMPAFGGRLNEEEIDAIATYVLGQAELD--------------------------------------------------------
```
Only provide one sequence in a given file (you can still output multiple based on the one). If you wish you can pad the sequence up to 140 characters long with a '-' symbol as above however this is not necessary (see the Seq-to-Metal-Seq example). Sequences longer than 140 will be chopped. If you do not provide a sequence the default example will be used. 

**2. Run the Script**

Run the following command to produce sequences from the input file (the default has been used)
```console
user@computer:~$ python seq_to_seq.py -infile examples/seq2seq_example.txt -numout 10
```
The `-numout` argument is an integer input for the number of sequences you want to produce. Here we are producing 10 examples. If you do not provide values for the two command line arguments the script defaults to producing 10 sequences from the default example. 

**3. Output Example**

Running the script in step 2 outputs 10 sequences and provides the average sequence identity between the sequences produced and the original input sequence. This is outputed to `stdout` Here is an example:
```
Average Sequence Identity to Input: 60.4%
ADLEAGEQIFSANCAACHGGGNNIIMPEKTLKKDALEENGMKSVEAITYQVTNGKNAMPAFGGRLSDEDIEDVANYVLSQAEKGW
ADLEHGAQIFSANCAACHAGGNNVIMPDKTLKKDALEKNGMNSIEAITYQVTNGKNAMPAFGGRLSDEDIEDVANYVLSQAEKGW
ADLENGGKVFSGACAACHIGGENIVRPEKTLKKDALEEGGMDSIEAITAQVTNGKNAAPAFGERLVDEDIEDVAEYVL
ADLAAGEQIFSANCAACHAGGNNVVMPDKTLKKDALEKYGMNSIEAITTQVTNGKNAMPAFGGRLEAEDIEDVAAYVLSQAEG
ADLEHGEQIFSANCAACHAGGNNVIMPEKTLKKDALEKYGMNSVEAITTQVTNGKNAMPAFGGRLEDEQIEDVANYVLSQSEW
ADIEHGEKIFSANCAACHAGGNNAIMRNKTLKKEALEPNGMNSIEAITYQVTNGKNAMPAFGGRLSDEDIEDVANYVLKQAEKGW
ADLAAGEQIFSANCAACHAGGNNIIMPEKTLKKEALEKYSMNSIEAITTQVTNGKNAAPAFGGRLSDEDIEDVANYVLSQAEKGW
ADIITGEQIFSANCAACHIGGNNAIRPEKTLKKPALETNGMNSVDAITTQVVNPKNAMPAFGGRLEDEDIEDVANYVLSQAEK
GDLEKGKGIFKFNCVACHSNGKNVIIIEKTLKKDALKANGMFSIDAITSQIANGKNAMPAFAGRLKDDLIELVAYYVLEKAEQW
ADLANGAKIFSANCAACHAGGGNAIMPTKTLKKNALEKNGMNSIEAITYQVTNGKNAMPAFKGRLSEEDIEDVAAYVLEQSEKGW

```

### 2. Seq-to-Metal-Seq

**1. Input File Preparation**

Note this is very similar to Seq-to-Seq example. Make sure you have a text file or fasta file with a single sequence in it. For example, in the [seq2metalseq_example.txt](produce_sequences/examples/seq2metalseq_example.txt) you'll find the following: 
```
DTDSEEEIKEAFKVFDKDGNGYISAAELRHVMTNLGEKLSDNEVDEMIREADVDGDGQINYEEFVKMMLSK
```
Only provide one sequence in a given file (you can still output multiple based on the one). If you wish you can pad the sequence up to 140 characters long with a '-' symbol however this is not necessary. Sequences longer than 140 will be chopped. If you do not provide a sequence the default example will be used. 

You also need to decide which metal you want the model to try and insert a binding site for. For example, if you choose Iron it will produce sequences more likely to bind the metal specified. The 8 metals you can choose from are: 
* Fe 
* Zn
* Ca
* Na
* Cu
* Mg
* Cd
* Ni 

**2. Run the Script**

Run the following command to produce sequences from the input file (the default has been used)
```console
user@computer:~$ python seq_to_metalseq.py -infile examples/seq2metalseq_example.txt -numout 10 -metal Fe
```
The `-numout` argument is an integer input for the number of sequences you want to produce. Here we are producing 10 examples. The `-metal` argument is the two letter atomic code (one of the above 8) of the one metal you wish to use. If you do not provide values for the three command line arguments the script defaults to producing 10 sequences from the default example inserting Fe binding. 

**3. Output Example**

Running the script in step 2 outputs 10 sequences and provides the average sequence identity between the sequences produced and the original input sequence. This is outputed to `stdout` Here is an example:
```
Average Sequence Identity to Input: 80.1%
DTDREEEIREAFRVFDKDGNGFISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVKMMEAK
MTDTEEEIDEAFRVFDKDGNGYDSAAELRHVMTNLGEKLTDEEVDEMIREADIAGDGQVNYEEFVTMMTAK
DTSSEEEIDEAFRVFDKDGNGFISAAELRHVMTNLGEKLTDEEVDEMIREADNAGDGQDNYEEFVTMMTVK
DTDEEEKIREAFRDFDKSDNEFDSAAELRHVMTAGGEKLTDEEVDEMIDGADMDDDGQDFDEEFDGMMTAK
DSDTEEEIKEAFRVFDKDGNGYISAAELRHVMTNVGEKLTDEEVDEMIREADIDGDGQVNYEEFVVMMTAK
DTDSEEEIREAFRVFDKDGNGFISAAELRHVMTNLGEKLTDEEVDEMIREADIAGDGQVNYEEFVKMMTAK
DTDSEEEIREAFRVFDKSDNGFISAAELRHLMTNLGEKLTDEEVDEMIREADIDGDGQINYEEFVKMMLAK
DTDREEEIREAFRVFKKSGNELISAAELRHVMTPLGEKLTDEEVDEMIREAIIDGDGQVNYEEFVGMMKDK
DTDSEEEIREAFRVFDKDGNGFISAAELRHVMTNLGEKLTDEEVDEMIREADIAGDGQVNYEEFVGMMTAK
DTDSESELKEAFRVADKDRNGPDSACKLRHVMLNGIEKLTDKEVDEMIREADIAEDGQVNYEEFVMT
```


### 3. Grammer to Seq

**1. Input File Preparation**

Make sure you have a text file that contains a single grammar string in it as defined in the linked paper. For example, in the [gram2seq_example.txt](produce_sequences/examples/gram2seq_example.txt) you'll find the following: 
```
+B+0-C+0+B+2-B+1
```
Only provide one grammar string in a given file (you can still output multiple sequences based on the one grammar string). If you do not provide a grammar string the default example will be used. 

**2. Run the Script**

Run the following command to produce sequences from the input file (the default has been used)
```console
user@computer:~$ python gram_to_seq.py -infile examples/gram2seq_example.txt -numout 10
```
The `-numout` argument is an integer input for the number of sequences you want to produce. Here we are producing 10 examples. If you do not provide values for the two command line arguments the script defaults to producing 10 sequences from the default example. 

**3. Output Example**

Running the script in step 2 outputs 10 sequences and provides these outputed to `stdout` Here is an example:
```
ESGYAVVCDTTCSYDGECNNECTCCCLKVKQKGNDGGYCWLWECGCLCLGAPVLVPEDTKCK
KKGCLVSRGTGCGSGCSNNNCAKGLKISNGAKGKEGHRGYKCGCGCFCWPDR
CDGYLVESKTGCGFCGLNNSCCNLCCNKNGAKAGYCACGYKCKCECLPLPLPN
RDGYPVHDKGCKISCFGNNYCWKECKKKGKSKGYCYCWWLACWCYGLPDPEKVWDYA
KKGYPVVSDDCCKYCCLNNKYCNYCCNKCGAKSGYCAWCCKSGCACWCLDLPK
ERDGYIADPTNCGYTCANNSCCNGLCTKNGAKAGYCAWIGPYGKACWCIPLPDKVP
KDYYPKDDKTCCSCCFNNNYCNKECKKEGKASGYCYGWCPACWCWCLPDDE
KKGKYINDGTNCKYTCANNAKNNCCDKKCGAKGGYGHWGYPFGKACWCFPLPE
KRGYLVVKNTNCKYSCFNLGYCNYCCTKCGAKSGYCSWGYCYGNACWCKPLPDKVPIRPPGKC
DRGYLVVSDTGCKYVCYNNSYNKYCDRKCKNKAEYYGFGWLFGYGCWCLPLPEPVWIKIVDC

```

## Editing the Software

If you'd like to edit the software we have provided training scripts that are self contained, along with trained models that can be loaded in. The directories [metal_gen](/metal_gen/) and [fold_gen](/fold_gen/) contain the scripts for experiments involving generating metal binding proteins and experiments producing sequences for a given fold, respectively. These scripts also contain the ability to perform inference however this much be specified by altering the script. Instructions are contained with the script.  






<!--
The two tasks this work approaches are adding a metal binding site to a protein sequence and generating a protein sequence for a given topology string. The first is described as Task 1 and the second as Task 2. Below are descriptions for using the trained models.

### Task 1
This is adding a metal binding site to a protein sequence. The files for this task are located in the metal_gen folder. Aside from the model file in the folder folder, the main script is the `metal_VAE_pytorch.py`. This file can be run from the command line with several arguments that can be seen by looking at the source. The code itself contains explanations for its use, but more specifically, it can be used for training a model or producing samples of a protein similar to another protein provided.

Note the model provided and thus inference was trained and can be used with the `nostruc` dataset.

<b> Training </b>

In the case of training please get in contact for the datasets, however if you have correctly formated data you should be able to use that instead. A model has been provided for a network with a latent dimension of 16 which was used in the paper. The dataset can be specified by command line args as either `struc` or `nostruc`. These are numpy `.npy` files of size `L x 4353` and `L x 3088` where `L` is the number of examples in the dataset. Both datasets have one-hot encoded sequences as the first 3080 dimensions and the next 8 as the binary switches for metal binding. The `struc` dataset contains an extra 1265 dimensions that describe the input data.  

Read through the command line arguments in the files in order to understand what parameters of the network can be changed, but you can change them by changing the defaults if running in an IDE. The current defaults are what was used to train the 16 dimensional network. To train the network you also need to change the switches specified in the script so that `cuda=True` and `train=True`.

<b> Sampling a new metal binder </b>

First make sure that that `cuda=False` and `train=False` in the script. The network is capable of running inference very quickly on just a cpu. From there make sure that `new_metal=True`. At the bottom of the script you will see

```Python
if new_metal:
    name="prots_nomet"
    ...
```
The `name` is the name not including `.npy` of the file being used (that needs to be placed on the current directory) of protein(s) you wish to alter. So the input file will be a numpy binary of size `L x 3088` where the last 8 metal binding flags are changed in accordance with what you want metal binding you want to add. If you've got that sorted then just run the script.

<b> Encoding and decoding separately</b>

In order to do these just make a function that is a part of the network module that looks like the forward pass but only uses the layers you intend of using i.e. for decoding or encoding. From there add whatever you like at the end of the script to call these functions and encode or decode data.

### Task 2
This is generating a protein sequence for a given topology string and the scripts are located in the fold_gen folder.--->
