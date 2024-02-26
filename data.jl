import FileIO, Images, OneHotArrays, MLUtils

"Always returns True"
always_true(a::Any) = true

struct Sample
    sample::Any
    loader::Function
    transform::Function
    label::Any
end
Sample(sample::Any, loader::Function, transform::Function, label::Function) = Sample(sample, loader, transform, label(sample))

abstract type Dataset end
mutable struct DatasetLabeled <: Dataset
    samples::Vector{Sample}
    labels::Vector{Any}
end
DatasetLabeled() = DatasetLabeled([], [])

function Base.getindex(dataset::DatasetLabeled, index::Int64)
    sample = dataset.samples[index]
    return (sample.sample |> sample.loader |> sample.transform,  OneHotArrays.onehot(sample.label, dataset.labels))
end

function Base.getindex(dataset::DatasetLabeled, indexes::Vector{Int64})
    return MLUtils.batch([Base.getindex(dataset, index) for index in indexes]) # ignore
end

function Base.length(collection::Dataset)
    return length(collection.samples)::Int
end

function Base.copy(x::DatasetLabeled)
    return DatasetLabeled(copy(x.samples), copy(x.labels))
end

"Add sample to a dataset"
function add_sample!(dataset::DatasetLabeled, sample::Any; loader::Function, transform::Function = identity, label::Any)
    sample = Sample(sample, loader, transform, label)
    push!(dataset.samples, sample)
    if sample.label ∉ dataset.labels
        push!(dataset.labels, sample.label)
    end
end

"Add folder to a dataset."
function add_folder!(dataset::DatasetLabeled, path_to_folder::String; loader::Function, transform::Function = identity, label::Any, filter::Function = always_true, extensions::Vector{String} = [])
    for (root, _, files) in walkdir(path_to_folder)
        for file in files
            p = joinpath(root, file)
            # if filter(p) and has one of the extensions / extensions list is empty
            if filter(p) && (!isempty(extensions) ? in(splitext(file)[2], extensions) : true)
                add_sample!(dataset, p, loader=loader, transform=transform, label=label)
            end
        end
    end
end

"Create a dataset from a folder."
function DatasetLabeled(path_to_folder::String; loader::Function, transform::Function = identity, label::Any, filter::Function = always_true, extensions::Vector{String} = [])
    ds = DatasetLabeled([], [])
    add_folder!(ds, path_to_folder, loader = loader, transform = transform, label = label, filter = filter, extensions = extensions)
    return ds
end

"Get the last folder in a path. For example `datasets/images/car/car4.jpg` -> `car`"
function last_folder(path::String)
    return splitdir(splitdir(path)[1])[2]
end

"Pre-applies loader to all samples in the dataset."
function preload!(dataset::Dataset)
    dataset.samples = [Sample(sample.loader(sample.sample), identity, sample.transform, sample.label) for sample in dataset.samples]
end

function split_dataset(dataset::Dataset, splits::Vector{Int})
    dataset_length = length(dataset)

    # make sure splits sum up to dataset length
    if sum(splits) != length(dataset)
        throw("Splits don't sum up to dataset length, which is $(length(dataset)), and your splits sum up to $(sum(splits))")
    end

    split_datasets = []
    cur = 1

    # get samples
    for i in splits
        # get samples from current index to next
        println(cur, ", ", i)
        samples = dataset.samples[cur:min(cur+i, dataset_length)]
        cur += i
        ds = copy(dataset)
        ds.samples = samples
        push!(split_datasets, ds)
    end
    return split_datasets
end

function split_dataset(dataset::Dataset, splits::Vector{AbstractFloat})
    if sum(splits) != 1
        throw("Splits sum up to $(sum(splits)), they should sum up to 1")
    end
    dataset_length = length(dataset)
    split_dataset(dataset, (floor.(Int64, splits .* dataset_length)))
end

function split_dataset(dataset::Dataset, split::AbstractFloat)
    splits = [split, 1 - split]
    split_dataset(dataset, splits)
end

function split_dataset(dataset::Dataset, split::Int)
    dataset_length = length(dataset)
    splits = [split, dataset_length - split]
    split_dataset(dataset, splits)
end


# ---------------

# include("./image.jl")

# loader(path::String) = resize(to_channels(read_image(path), 3), (32, 32))

# ds = DatasetLabeled()

# add_folder!(ds, raw"C:\Users\nkshv\Pictures", loader = loader, label = last_folder, extensions = [".jpg", ".png", ".jpeg"])

# getindex(ds, [1,2,3])

# import MLUtils

# dl = MLUtils.DataLoader(ds, batchsize = 8, shuffle = true, collate = true)
