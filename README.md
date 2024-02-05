# ReadMe



# T5_Model


T5Untitled6.ipynb is main file and , summary_audio.mp3 is result of summary and T5_Model.mkv is video of implementing model


This set of code demonstrates the process of setting up a Python environment, installing necessary packages, and utilizing the transformers library along with T5 (Text-To-Text Transfer Transformer) model to generate a summary of a given text. The summary is then converted into audio using the Google Text-to-Speech (gTTS) library.

## Setting up the Environment

The code begins by ensuring the correct Python version is available and setting the PYTHONPATH environment variable. It then installs the virtualenv package, creates a virtual environment named 'myenv', and activates it.

```bash
!which python
!python --version

%env PYTHONPATH=

!pip install virtualenv

!virtualenv myenv
```

## Installing Miniconda

The code proceeds to install Miniconda, a minimal version of the Anaconda distribution, and configures it to use Python 3.8 along with installing the 'ujson' package.

```bash
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

!chmod +x Miniconda3-latest-Linux-x86_64.sh

!./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

!conda install -q -y --prefix /usr/local python=3.8 ujson
```

## Installing Transformers and Torch

Next, the code installs specific versions of the transformers and torch libraries.

```bash
!pip install transformers==2.8.0
!pip install torch==1.4.0
!pip install gtts
```

## Utilizing T5 for Text Summarization

The T5 model is then loaded, and a sample text is provided for summarization. The summary is generated and converted into audio using gTTS.

```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from gtts import gTTS

# ... (previous code)

# T5 model and tokenizer initialization
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

# Sample text for summarization
text = """
This text discusses a judgment from the Supreme Court of India
regarding a complaint filed under Section 138 of the Negotiable Instruments Act.
The case involves a dispute over a cheque
issued by the respondent, which was returned due to insufficient funds.
The Trial Court initially dismissed the complaint,
but the Supreme Court upheld it,
finding that the cheque was indeed issued
by the respondent.
"""

# Preprocessing and tokenization
preprocessed_text = text.strip().replace('\n','')
t5_input_text = 'summarize: ' + preprocessed_text

# Tokenizing the text
tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt', max_length=512, truncation=True, truncation_strategy='longest_first').to(device)

# Generating summary
summary_ids = model.generate(tokenized_text, min_length=1, max_length=200)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Converting summary to audio
language = 'en'
tts = gTTS(text=summary, lang=language, slow=False)
tts.save("summary_audio.mp3")
```

This code provides an end-to-end example of using the T5 model for text summarization and converting the generated summary into audio.
