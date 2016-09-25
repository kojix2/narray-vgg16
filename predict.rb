# VGG0-16 (forward only) with Ruby
# pretrained parameters from https://github.com/mitmul/chainer-imagenet-vgg

require 'numo/narray'
require 'oily_png'
require 'optparse'
N = Numo::SFloat

tk = false
OptionParser.new do |opt|
  opt.on('-d', '--display', 'Show Tk GUI Window'){tk = true}
  opt.parse!(ARGV)
end
raise "No image error" if ARGV[0].nil?

# Add methods for CNN to NArray::SFloat
class N
  @@k = []
  3.times{|i| 3.times {|j| @@k << [true, i..-3+i, j..-3+j] }}

  def conv(weight, bias)
    c, b, a = shape
    # zero padding
    na = N.zeros(c, b+2, a+2)
    na[true, 1..-2, 1..-2] = self
    c2 = weight.shape[0]
    r = N.zeros(c2, b, a)
    c2.times do |i|
      w = weight[i, true, true, true]
      r[i, true, true] = @@k.map { |k|
        (na[*k] * w[*k]).sum(0)
      }.inject(&:+) + bias[i]
    end
    r 
  end

  def pool
    c,b,a = shape
    reshape(c, b/2, 2, a/2, 2).max(2,4)
  end

  def line(weight,bias)
    (self * weight).sum(1) + bias
  end

  def relu
    self[self<0] = 0
    self
  end

  def softmax
    e = Numo::NMath.exp(self - self.max)
    e / e.sum
  end
end

def _load(w,s=nil)
  na = N.from_string File.binread(File.join("weights", w)), s
end

# load weights from binary files.
W01 = _load "W01", [ 64,  3,3,3]
W02 = _load "W02", [ 64, 64,3,3]
W03 = _load "W03", [128, 64,3,3]
W04 = _load "W04", [128,128,3,3]
W05 = _load "W05", [256,128,3,3]
W06 = _load "W06", [256,256,3,3]
W07 = _load "W07", [256,256,3,3]
W08 = _load "W08", [512,256,3,3]
W09 = _load "W09", [512,512,3,3]
W10 = _load "W10", [512,512,3,3]
W11 = _load "W11", [512,512,3,3]
W12 = _load "W12", [512,512,3,3]
W13 = _load "W13", [512,512,3,3]
W14 = _load "W14", [4096, 25088]
W15 = _load "W15", [4096,  4096]
W16 = _load "W16", [1000,  4096]

# load biases from binary files.
B01 = _load "B01", [64]
B02 = _load "B02", [64]
B03 = _load "B03", [128]
B04 = _load "B04", [128]
B05 = _load "B05", [256]
B06 = _load "B06", [256]
B07 = _load "B07", [256]
B08 = _load "B08", [512]
B09 = _load "B09", [512]
B10 = _load "B10", [512]
B11 = _load "B11", [512]
B12 = _load "B12", [512]
B13 = _load "B13", [512]
B14 = _load "B14", [4096]
B15 = _load "B15", [4096]
B16 = _load "B16", [1000]  

# load labels
labels = File.readlines('imagenet_classes.txt').map(&:chomp)

# Todo : change file path
_path = ARGV[0]
image  = ChunkyPNG::Image.from_file(_path)

height = image.height
width  = image.width

if height =! 224 or width != 224
  raise "image size must be 224 x 224"
end

# Convert image to narray
nimage = Numo::UInt8.from_string(image.to_rgb_stream, [224,224,3])
nimage = nimage.reverse(2) # rgb →  bgr

mean   = N[103.939, 116.779, 123.68]
nimage -= mean
nimage = nimage.transpose(2,0,1)


# VGG16
result = nimage 
  .conv(W01, B01).relu
  .conv(W02, B02).relu
  .pool                      .tap{ puts "conv1.." }
  .conv(W03, B03).relu
  .conv(W04, B04).relu
  .pool                      .tap{ puts "conv2.." }
  .conv(W05, B05).relu
  .conv(W06, B06).relu
  .conv(W07, B07).relu
  .pool                      .tap{ puts "conv3.." }
  .conv(W08, B08).relu
  .conv(W09, B09).relu
  .conv(W10, B10).relu
  .pool                      .tap{ puts "conv4.." }
  .conv(W11, B11).relu
  .conv(W12, B12).relu
  .conv(W13, B13).relu
  .pool                      .tap{ puts "conv5.." }
  .flatten
  .line(W14, B14).relu
  .line(W15, B15).relu
  .line(W16, B16).softmax

result = labels.zip(result.to_a.flatten).to_a
top5 = result.sort_by{|r| -r[1]}[0..4]
top5.each do |name, prob|
  puts "#{name} : #{prob * 100}"
end

###### GUI (Ruby/Tk) ######
if tk 
  names, probs = top5.reverse.transpose
  probs = probs.map{|_p| _p*100} # percentage

  require 'tkextlib/tcllib/plotchart'
  require 'tkextlib/tkimg/png'

  TkRoot.new(title: "VGG16 with Ruby + Numo::NArray") do |r|
    TkLabel.new(r) do |l|
      image TkPhotoImage.new(file: _path)
      pack(side: :left)
    end
    TkCanvas.new(r) do |c|
      width 363
      height 224
      Tk::Tcllib::Plotchart::HorizontalBarchart.new(c, [0,100,20], names, 1) do
        title _path
        plot 'prob', probs, 'blue'
        bg "white"
        pack
      end
      pack(side: :right)
    end
  end

  Tk.mainloop
end
