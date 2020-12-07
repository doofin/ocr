# LSTM + CTC for handwriting character recognization(graduation thesis for undergraduate,written in Chinese,based on tensorflow)

End to end training and recognization of whole sentence handwriting character recognization,no process of segmentation into individual alphabets is involved!

Overview of the architecture : 

fixed height ,variable length image of a sentence  -> unrolled dynamic LSTM by variable length -> CTC(Connectionist temporal classification) loss -> linear layer -> result

## result:

![result](https://raw.githubusercontent.com/doofin/ocr/master/ocrResult.png?token=ABOC6CLFIU7FR4VZLR6J7O27X5T44)


## Requirements
Attention: The archetecture is set up at about 2017 and has become outdated ,it's better to reimplement with the new pytorch framework

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
    #
    # format: a01-000u-s0-00 0 ok 154 19 408 746 1663 91 A|MOVE|to|stop|Mr.|Gaitskell|from
    #
    #     a01-000u-s0-00  -> sentence/line id for form a01-000u
    #     0               -> sentence number within this form
    #     ok              -> result of word segmentation
    #                            ok: line is correctly segmented
    #                            er: segmentation of line has one or more errors
    #
    #                        warning: if a sentence starts or ends in the middle of
    #                                 a line which is not correctly segmeted, a
    #                                 correct extraction of the sentence can fail.
    #
    #     154             -> graylevel to binarize line
    #     19              -> number of components for this part of the sentence
    #     408 746 1663 91 -> bounding box around for this part of the sentence
    #                        in the x,y,w,h format
    #
    #     A|MOVE|to|stop|Mr.|Gaitskell|from
    #                     -> transcription for this part of the sentence. word
    #                        tokens are separated by the character |
    
# files:

thesisnew.pdf : the thesis in Chinese

.py files under model folder : the AI model 

.hs files under visual folder : haskell code for visualization

data folder contains sentence images for training
