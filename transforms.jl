import Images
import Statistics
import ColorTypes
import FileIO

abstract type Transform end
abstract type TransformReturns <: Transform end
abstract type TransformInplace <: Transform end

# __________ Compose __________
struct Compose <: TransformReturns
    transforms::Base.AbstractVecOrTuple{Any}
end
Compose(transforms...) = Compose(transforms)
Compose(comp1::Compose, comp2::Compose) = Compose(comp1.transforms..., comp2.transforms...)
Compose(comp::Compose, transforms...) = Compose(comp.transforms..., transforms...)
Compose(transform::Any, comp::Compose) = Compose(transform, comp.transforms...)
function (transform::Compose)(sample::Any)
    for t in transform.transforms
        if t isa TransformInplace
            t(sample)
        else
            sample = t(sample)
        end
    end
    return sample
end

# __________ Normalize __________
"Normalizes an array according to the mean and standard deviation of each channel"
struct Normalize <: TransformInplace
    mean::Base.AbstractVecOrTuple{AbstractFloat}
    std::Base.AbstractVecOrTuple{AbstractFloat}
end
Normalize(mean::AbstractFloat, std::AbstractFloat) = Normalize([mean], [std])
Normalize(mean_std::Base.AbstractVecOrTuple{Base.AbstractVecOrTuple{AbstractFloat}}) = Normalize(mean_std[1], mean_std[2])
function (transform::Normalize)(sample::AbstractArray{<:AbstractFloat})
    for (i,v) in enumerate(eachslice(sample, dims=ndims(sample)))
        v .-= transform.mean[i]
        v ./= transform.std[i]
    end
end

# __________ Resize __________
struct Resize <: Transform
    size::Base.AbstractVecOrTuple{Int}
end
Resize(size...) = Resize(size)
(transform::Resize)(sample::Matrix{<:ColorTypes.Colorant}) = Images.imresize(sample, transform.size)

# __________ ToChannels __________
struct ToChannels <: Transform
    channels::Int
end
"Converts your sample to specificed amount of channels. Your sample must have last dimension as the channel."
function (transform::ToChannels)(sample::Array{Float32})
    n_channels = last(size(sample))
    if n_channels == transform.channels return sample
    elseif n_channels == 1 return repeat(sample, outer = [ones(Int64, ndims(sample)-1)..., transform.channels]) 
    elseif transform.channels == 1 return Statistics.mean(sample, dims = ndims(sample))
    elseif n_channels > transform.channels return selectdim(sample, ndims(sample), 1:transform.channels)
    end
    throw("I haven't implemented that yet")
end

# __________ unsqueeze __________
function unsqueeze(sample::Array, dim::Int)
    sample_size = size(sample)
    insert!(sample_size, dim, 1)
    return reshape(sample, sample_size)
end

# __________ ensure_channel_last __________
"Ensures the sample is in channel-last format. Note that if size[1] < size[end], it is considered channel-first. Make sure your samples already have a channel dimension!"
function ensure_channel_last(sample::Array)
    sample_size = size(sample)
    if first(sample_size) >= last(sample_size)
        return sample
    end
    sample_dims = Vector(range(1, ndims(sample), step=1))
    sample_dims[1], sample_dims[end] = sample_dims[end], sample_dims[1]
    return permutedims(sample, sample_dims)
end
"Uses Images.channelview and permutes to channel-last. Adds channel dim to grayscale images."
function ensure_channel_last(sample_image::Matrix{<:ColorTypes.Colorant}) 
    sample = Images.channelview(sample_image)
    if ndims(sample) == 2
        return reshape(sample, (size(sample)..., 1))
    end
    return permutedims(sample, [2, 3, 1])
end

# __________ ensure_channel_first __________
"Ensures the sample is in channel-first format. Note that if size[1] > size[end], it is considered channel-last. Make sure your samples already have a channel dimension!"
function ensure_channel_first(sample::Array)
    sample_size = size(sample)
    if first(sample_size) <= last(sample_size)
        return sample
    end
    sample_dims = Vector(range(1, ndims(sample), step=1))
    sample_dims[1], sample_dims[end] = sample_dims[end], sample_dims[1]
    return permutedims(sample, sample_dims)
end
"Returns Images.channelview, and adds channel dim to grayscale images."
function ensure_channel_first(sample_image::Matrix{<:ColorTypes.Colorant}) 
    sample = Images.channelview(sample_image)
    if ndims(sample) == 2
        return reshape(sample, (1, size(sample)...))
    end
    return sample
end

# __________ to_image __________
"Creates a 3D array{Float32}, where 3rd dim is the channel."
to_image_arr(sample::Matrix{<:ColorTypes.Colorant}) = sample |> ensure_channel_last |> Array{Float32}
to_image_arr(sample::Array) = Array{Float32}(ensure_channel_last(sample))
to_image_arr(sample::Array{Float32}) = ensure_channel_last(sample)
to_image_arr(sample::String) = sample |> FileIO.load |> ensure_channel_last |> Array{Float32}

