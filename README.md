# Variational autoencoder for protein sequences

This repository provides code to accompany the paper:

Greener JG, Moffat L and Jones DT, Design of metalloproteins and novel protein folds using variational autoencoders, *Scientific Reports* 8:16189, 2018 - [link](https://www.nature.com/articles/s41598-018-34533-1)

The work describes a variational autoencoder that can add metal binding sites to protein sequences, or generate protein sequences for a given protein topology.

## Getting Started
These instructions will get a copy of this repo on your system and running so you can produce sequences with our models or expand on our work. These files have been cleaned and minimized so they can be run using python with as few dependencies as possible.

### Dependencies
The follow are the packages needed to run our software. Our software uses python and leverages several python packages:
* [python 3.6](https://www.python.org/)
* [pytorch 1.10](https://pytorch.org/)
* [numpy 1.19](http://www.numpy.org/)
* [lark-parser 0.4.1](https://github.com/lark-parser/lark)

Install either through [Anaconda](https://anaconda.org/), or run with the dockerfile `make build bash` and then `python vae.py`.

If you wish to extend our work note that we have configured the general use scripts to run solely on CPU as they do not require the computational power of a GPU. However, the example training scripts provided can be run on GPU and contain switches and/or command line arguments to enable this.

### Installation

Navigate to the directory you wish to store the software in and simply run the following from the command line.
```console
user@computer:~$ git clone git@github.com:psipred/protein-vae.git
user@computer:~$ cd protein-vae
```
You are now in the directory to run the scripts to produce sequences. Producing sequences is described below.

## Running the software
Before using our software please read the paper as linked to at the top of these instructions. There are three different methods of producing sequences that this software provides:
1. **Seq-to-Seq**: Providing an initial sequence to return a similar sequence with some variation.
2. **Seq-to-Metal-Seq:** Providing an initial sequence that does not bind to a metal and returning a the same sequence with variation that is more likely to bind a specified metal.
3. **Grammar-to-Seq:** Providing a grammar string (see the paper) and producing a sequence that is likely to fold to the topology described in the grammar string.

Each one of these methods is run with its own python script. Below we have provided examples for running each one of the scripts and what the input and output should look like. We have also provided example sequences (as found in the [examples](examples/) directory).

If you run one of the scripts without providing a sequence it will default use one of the example sequences. Below are examples of how to use each one of the scripts with the default examples.

### 1. Seq-to-Seq Example
**1. Input File Preparation**

Make sure you have a text file or fasta file with a single sequence in it. For example, in the [seq2seq_example.txt
](examples/seq2seq_example.txt) you'll find the following:
```
AEVPSGEQLFNSNCSACHIGGNNVIISHKTLRKEALEKYAMNSLEAIRYQVVNGKNAMPAFGGRLNEEEIDAIATYVLGQAELD--------------------------------------------------------
```
Only provide one sequence in a given file (you can still output multiple based on the one). If you wish you can pad the sequence up to 140 characters long with a '-' symbol as above however this is not necessary (see the Seq-to-Metal-Seq example). Sequences longer than 140 will be chopped. If you do not provide a sequence the default example will be used.

**2. Run the Script**

Run the following command to produce sequences from the input file (the default has been used):
```console
user@computer:~$ python seq_to_seq.py -infile examples/seq2seq_example.txt --numout 10
```
The `--numout` argument is an integer input for the number of sequences you want to produce. Here we are producing 10 examples. If you do not provide values for the two command line arguments the script defaults to producing 10 sequences from the default example.

**3. Output Example**

Running the script in step 2 outputs 10 sequences and provides the average sequence identity between the sequences produced and the original input sequence. This is outputed to `stdout`. Here is an example:
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

Note this is very similar to Seq-to-Seq example. Make sure you have a text file or fasta file with a single sequence in it. For example, in the [seq2metalseq_example.txt](examples/seq2metalseq_example.txt) you'll find the following:
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
user@computer:~$ python seq_to_metalseq.py -infile examples/seq2metalseq_example.txt --numout 10 --metal Fe
```
The `--numout` argument is an integer input for the number of sequences you want to produce. Here we are producing 10 examples. The `--metal` argument is the two letter atomic code (one of the above 8) of the one metal you wish to use. If you do not provide values for the three command line arguments the script defaults to producing 10 sequences from the default example inserting Fe binding.

**3. Output Example**

Running the script in step 2 outputs 10 sequences and provides the average sequence identity between the sequences produced and the original input sequence. This is outputed to `stdout`. Here is an example:
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

### 3. Grammar to Seq

**1. Input File Preparation**

Make sure you have a text file that contains a single grammar string in it as defined in the linked paper. For example, in the [gram2seq_example.txt](examples/gram2seq_example.txt) you'll find the following:
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

Running the script in step 2 outputs 10 sequences and provides these outputed to `stdout`. Here is an example:
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

The grammar is specified using the Lark package and can be found [here](https://github.com/psipred/protein-vae/blob/master/gram_to_seq.py#L27-L116).

## Editing the Software

If you'd like to edit the software we have provided training scripts that are self contained, along with trained models that can be loaded in. The [vae](/vae.py) script contains the model definition and training method for metal binding and given fold, respectively.

## Datasets

The datasets used for training are available as [numpy binaries](/data/) to be read in with `numpy.load`:
* `assembled_data_mb_csr.npz` is the metal binding training data, size (147842, 3088). Each row is an example consisting of 22x140=3080 values for the sequence and 8 values for the metal binding flags (order as above).
* `assembled_data_fold_csr.npz` is the topology training data, size (104845, 4353). Each row is an example consisting of 22x140=3080 values for the sequence, 23x55=1265 values for the topology encoding and 8 values for the metal binding flags (order as above).

The assignments of Taylor topology strings to SCOP folds we used can be found in [topology_data](/topology_data/) along with the 3,785 PDB chain IDs used in the dataset.
