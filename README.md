# set_card_finder_EDGETPU

Disclaimer and warning: This is my first GitHub project and I am a beginner programmer.

This will find sets from an image of set cards. I am trying to get it to run quickly on a rasberry pi. I train tensorflow models to classify number, shape, and fill. I convert those to tflite, and also quantize them and then compile them with edgetpu compiler. Thus there are two 'set-finder' files, one for tflite and one that uses EDGETPU.  Note that I use Opencv to find color instead of tensorflow.



I plan to add something to grab an image of cards on the fly and find sets, rather than having to load an image.
