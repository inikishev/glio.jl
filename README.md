# glio_julia
WIP library for Flux to further speed up experimenting and fiddling around

I am working on figuring out how to make proper modules in Julia...

For now it adds a little bit of useful stuff that no one else seems to have made for julia/Flux:

### data.jl:
`DatasetLabeled` - dataset class that makes creating datasets for classification extremely easy
```julia
ds = DatasetLabeled()
loader(path::String) = resize(to_channels(read_image(path), 3), (32, 32))
add_folder!(ds, raw"C:\Users\nkshv\Pictures", loader = loader, label = last_folder, extensions = [".jpg", ".png", ".jpeg"])
dl = MLUtils.DataLoader(ds, batchsize = 8, shuffle = true, collate = true)
```
Label functiom, `last_folder`, is like `torchvision.datasets.ImageFolder` - for `root/dog/xxx.png` it returns `dog`. DatasetLabel it automatically saves labels into `ds.labels`, and one-hot encodes them when iterating. Simply use with MLUtils DataLoader and set collate to true.

Finally, if you have enough RAM, preload all samples using `preload!(ds)`, leads to insane speed ups.

### image.jl:
- `read_image` just goes ahead and reads an image into a WHC format that Flux uses, no matter if it is colored or black/white or whatever. Ready to use with your Flux models.
- `to_channels` converts that image into specified amount of channels. If you want 1 channel, it will take the average; If you set to 3 channels, black and white images will have their channels repeated; images with 4 channels (alpha channel) will have the last channel discarded.
- `resize` - resizes that image using Images.imresize but also makes sure that function can actually resize it
