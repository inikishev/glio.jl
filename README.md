# glio.jl
WIP library for Flux to further speed up experimenting and fiddling around

I am working on figuring out how to make proper modules in Julia... 

For now it adds some useful stuff that no one else seems to have made for julia and flux, in particular for making datasets. Read below or look at a quick example https://github.com/stunlocked1/glio_julia/blob/main/example.ipynb

## Importing
```julia
include("./data.jl")
include("./transforms.jl")
```
dw this will change

## Creating a dataset
I am going to create a dataloader for  https://www.kaggle.com/competitions/it1244-brain-tumor-dataset


```julia
import FileIO

# `DatasetLabeled` struct makes creating classification datasets extremely simple.
dataset = DatasetLabeled()

# We can define a loader using Compose which is basically like `torchvision.transforms.Compose`. And we can use those transforms:
# Resize(256, 256) - a parametrized wrapper around `Images.imresize`. Crop would be better but whatever...
# to_image_arr - this takes RGB array that images are loaded as and turns into a channel-last float32 array, ready to use with Flux.
# ToChannels(3) - Most samples have 3 channels but few have 1, so this will duplicate those to make sure it is always 3.
loader = Compose(FileIO.load, Resize(256, 256), to_image_arr, ToChannels(3))

# add all samples
path = "D:/datasets/it1244-brain-tumor-dataset/data"
open("$path/train/data.csv") do f
    while !eof(f)

        # a line has something like `76,benign`
        num, label = split(replace(readline(f), "\ufeff" => ""), ",")

        # add sample to the dataset with its label. The label will be automatically one-hot encoded when indexing into the dataset, so you can just pass a string.
        add_sample!(dataset, "$path/train/$num.jpg", loader=loader, label=label)
    end
end
```

Thats it! We now have a dataset with all samples and their labels.
```julia
dataset.classes
>> 2-element Vector{Any}:
   "benign"
   "malignant"
```

Indexing into the dataset returns a tuple with the image and one-hot encoded label
```julia
image, label = dataset[1]

size(image)
>> (256, 256, 3)

label
>> 2-element OneHotVector(::UInt32) with eltype Bool:
   1
   ⋅
```

## Normalizing
Z-normalization means making sure each channel in your dataset has `mean = 0` and `std = 1`, which helps with training. And hereby I present to you - Quite possibly the the laziest way to do it:
```julia
meanstd = mean_std(dataset, 32)
add_loader!(dataset, Normalize(meanstd))
```
`mean_std` calculates channel-wise mean and standard deviation in batches. It supports arbitrary number of dimensions, as long as all samples are channel-last and have the same number of dimensions and channels. 

`add_loader!` adds a new function to the end of the loader, using `Compose` or appending to existing one. 

`Normalize` does channel-wise normalization, like `torchvision.transforms.Normalize`, but last dimension is considered the channel (thats what Flux uses)

## Preloading
Reading and decoding those images takes a lot of time. Since there aren't many of them, preloading them into RAM is gonna speed things up. This is how to do it:
```
preload!(dataset)
```
This applies `loader` to all samples and replaces their `loader` with `identity`. And if you need random transformations, put them into `transform` when adding your samples, which will still be evaluated each time.

## Splitting
Thats right, there is also the laziest solution to splitting:
```julia
ds_train, ds_test, ds_val = split_dataset(ds, [0.7, 0.2, 0.1])
```

## DataLoader
Actually the entire thing is done. We can now create a DataLoader for the dataset using MLUtils.DataLoader. Make sure to set `collate` to `true` for proper sample and label collating.
```julia
import MLUtils
dl_train = MLUtils.DataLoader(ds_train, batchsize=16, shuffle=true, collate=true)
dl_test = MLUtils.DataLoader(ds_test, batchsize=64, shuffle=false, collate=true)

# training loop
for (images, labels) in dl_train
  ...
```

