Undergraduate dissertation about sequence to sequence translation and ocr,written in Chinese.

# Abstract 
This dissertation shows an neural model to do sequence to sequence translation,more specifically,it uses LSTM + CTC for handwriting English character recognition.
It features an end to end structure,that is a whole sentence is processed,so that there is no segmentation of sentence into individual alphabets.

## result:

![result](https://raw.githubusercontent.com/doofin/ocr/master/ocrResult.png?token=ABOC6CLFIU7FR4VZLR6J7O27X5T44)

Look at the original and decoded(result) line,you can see that although the performance is not perfect,it is comparatively feasible giving the obstrusiveness of the handwriting style.

# The architecture of the neural network

fixed height ,variable length image of a sentence  -> unrolled dynamic LSTM by variable length -> CTC(Connectionist temporal classification) loss -> linear layer -> result

## Requirements
Attention: The code is written in 2017 and has become outdated .please consider the new pytorch framework for its simplicity of dynamic nets.

- Python 2.7+
- Tensorflow 1.x
- numpy
- scipy

# run

run wordOcr.py  with python , this will train the model and save it.

# data format:

    #--- sentences.txt -----------------------------------------------------------#
    #
    # iam database sentence information
    #     A|MOVE|to|stop|Mr.|Gaitskell|from
    #                     -> transcription for this part of the sentence. word
    #                        tokens are separated by the character |
    
# files:

thesisnew.pdf : the thesis in Chinese

model/*.py : the AI model 

visual/*.hs : haskell code for visualization

data folder contains sentence images for training
