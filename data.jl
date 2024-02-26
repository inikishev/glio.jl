import FileIO, Images, OneHotArrays, MLUtils

"Always returns True"
always_true(a::Any) = true

struct Sample
    sample::Any
    loader::Function
    label::Any
    Sample(sample::Any, loader::Function, label::Function) = new(sample, loader, label(sample))
    Sample(sample::Any, loader::Function, label::Any) = new(sample, loader, label)
end

abstract type Dataset end
mutable struct DatasetLabeled <: Dataset
    samples::Vector{Sample}
    labels::Vector{Any}
    DatasetLabeled() = new([], [])
end

function Base.getindex(dataset::DatasetLabeled, index::Int64)
    sample = dataset.samples[index]
    return (sample.loader(sample.sample), OneHotArrays.onehot(sample.label, dataset.labels))
end

function Base.getindex(dataset::DatasetLabeled, indexes::Vector{Int64})
    return MLUtils.batch([Base.getindex(dataset, index) for index in indexes]) # ignore
end

function Base.length(dataset::Dataset)
    return length(dataset.samples)::Int
end

function add_sample!(dataset::DatasetLabeled, sample::Any, loader::Function, label::Any)
    sample = Sample(sample, loader, label)
    push!(dataset.samples, sample)
    if sample.label ∉ dataset.labels
        push!(dataset.labels, sample.label)
    end
end

"Add folder to a dataset."
function add_folder!(dataset::DatasetLabeled, path::String; loader::Function, label::Any, filter::Function = always_true, extensions::Vector{String} = [])
    for (root, _, files) in walkdir(path)
        for file in files
            p = joinpath(root, file)
            # if filter(p) and has one of the extensions / extensions list is empty
            if filter(p) && (!isempty(extensions) ? in(splitext(file)[2], extensions) : true)
                add_sample!(dataset, p, loader, label)
            end
        end
    end
end

"Get the last folder in a path. For example `datasets/images/car/car4.jpg` >> `car`"
function last_folder(path::String)
    return splitdir(splitdir(path)[1])[2]
end

"Pre-applies loader to all samples in the dataset."
function preload!(dataset::Dataset)
    dataset.samples = [Sample(sample.loader(sample.sample), identity, sample.label) for sample in dataset.samples]
end
# ---------------

# include("./image.jl")

# loader(path::String) = resize(to_channels(read_image(path), 3), (32, 32))

# ds = DatasetLabeled()

# add_folder!(ds, raw"C:\Users\nkshv\Pictures", loader = loader, label = last_folder, extensions = [".jpg", ".png", ".jpeg"])

# getindex(ds, [1,2,3])

# import MLUtils

# dl = MLUtils.DataLoader(ds, batchsize = 8, shuffle = true, collate = true)
