# LSTM + CTC for handwriting recognization

end to end training,no segmentation into individual alphabets!

big picture : fixed height sentence picture (variable length) -> unrolled lstm by variable length -> ctc loss


## Requirements

- Python 2.7+
- Tensorflow 1.0+
- numpy
- scipy

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
#