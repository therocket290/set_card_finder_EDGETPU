# set_card_finder_EDGETPU

This will find sets from an image of set cards. I am trying to get it to run quickly on a rasberry pi. I trained tensorflow models to classify number, shape, and fill. I convert those to tflite, and also quantize them and compile them with edgetpu compiler.  I use Opencv to find color.
So far, I have two parts. One uses tensorflow lite models and the other uses EDGETPU models.

I plan to add something to grab an image of cards on the fly and find sets, rather than having to load an image.
