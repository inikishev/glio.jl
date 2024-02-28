import FileIO, Flux, Images
using Statistics


"Returns channel-first float32 image in WHC format (width - height - channels). Black and white images will have one channel as well."
function read_image_arr(path::String)
    arr = Array{Float32}(Images.channelview(FileIO.load(path)))
    if ndims(arr) == 2
        return Flux.unsqueeze(arr, 3)
    end
    return permutedims(arr, (2, 3, 1))

end

"Converts an image to specificed amount of channels. Expects and returns WHC format."
function to_channels(image::Array{Float32, 3}, channels::Int)
    n_channels = size(image, 3)
    if n_channels == channels return image
    elseif n_channels == 1 return repeat(image, outer = [1,1,channels]) 
    elseif channels == 1 return mean(image, dims = 3)
    elseif n_channels > channels return image[:, :, 1:channels]
    end
    throw("I haven't implemented that yet")
end

"Resizes the image to specified size. Does not preserve aspect ratio. Expects and returns WHC format."
function resize(image::Array{Float32, 3}, new_size::Tuple{Int, Int})
    if size(image, 3) == 1
        return Images.imresize(repeat(image, outer = [1,1,3]), new_size)[:,:,1:1]
    end
    return Images.imresize(image, new_size)
end