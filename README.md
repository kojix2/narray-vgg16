# narray-vgg16

VGG-16 is a famous image classification model.

## Requirements
- Ruby
- Numo::NArray (latest version from github)
- oily_png
- If you want to show GUI Window 
  - Ruby/Tk + Plotchart is required

## External data
I get weights from this chainer model using "numpy.ndarray.tofile" manually
[chainer-imagenet-vgg](https://github.com/mitmul/chainer-imagenet-vgg)
Don't forget convert chainer variable to numpy or cupy using "x.data"
The original caffemodel is provided [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

## Run
```
$ ruby predict.rb [png_image_file_path]
```
Note: Image size must be 224 x 224

Show Tk Window
```
$ ruby predict.rb -d [png_image_file_path]
```
![screenshot](https://raw.githubusercontent.com/kojix2/narray-vgg16/images/vggcat.png)

Sample outputs:
```
conv1..
conv2..
conv3..
conv4..
conv5..
kit fox, Vulpes macrotis : 22.374089062213898
red fox, Vulpes vulpes : 18.137173354625702
Egyptian cat : 15.885089337825775
tiger cat : 15.635280311107635
tabby, tabby cat : 11.92069873213768
```

## Contribute
Please contribute to Ruby science and visualization packages.

[SciRuby](https://github.com/SciRuby)

[sciruby-jp](https://github.com/sciruby-jp)
