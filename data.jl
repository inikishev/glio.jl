import FileIO, Images, OneHotArrays, MLUtils, Statistics

"Always returns True"
always_true(a) = true

struct Sample
    sample
    loader
    transform
    label
end
Sample(sample, loader, transform, label::Function) = Sample(sample, loader, transform, label(sample))

abstract type Dataset end
mutable struct DatasetLabeled <: Dataset
    samples::Vector{Sample}
    classes::Vector
end
DatasetLabeled() = DatasetLabeled([], [])

function Base.getindex(dataset::DatasetLabeled, index::Int)
    sample = dataset.samples[index]
    return (sample.sample |> sample.loader |> sample.transform, OneHotArrays.onehot(sample.label, dataset.classes))
end

function Base.getindex(dataset::DatasetLabeled, indexes::Vector{Int})
    return MLUtils.batch([Base.getindex(dataset, index) for index in indexes]) # ignore
end

function Base.iterate(dataset::DatasetLabeled)
    getindex(dataset, 1), 1
end

function Base.iterate(dataset::DatasetLabeled, state)
    state == length(dataset) ? nothing : (getindex(dataset, state + 1), state + 1)
end

function Base.length(collection::Dataset)
    return length(collection.samples)
end

function Base.copy(x::DatasetLabeled)
    return DatasetLabeled(copy(x.samples), copy(x.classes))
end

"Add sample to a dataset"
function add_sample!(dataset::DatasetLabeled, sample; loader, transform=identity, label)
    sample = Sample(sample, loader, transform, label)
    push!(dataset.samples, sample)
    if sample.label ∉ dataset.classes
        push!(dataset.classes, sample.label)
    end
end

"Add folder to a dataset."
function add_folder!(dataset::DatasetLabeled, path_to_folder::String; loader, transform=identity, label, filter=always_true, extensions::Vector=[])
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

include("./transforms.jl")
"Adds a loader function to the end of existing loader pipeline."
function add_loader!(dataset::Dataset, loader, sample_filter=always_true)
    dataset.samples = [Sample(sample.sample, Compose(sample.loader, loader), sample.transform, sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
end
"Replaces the loader function"
function set_loader!(dataset::Dataset, loader, sample_filter=always_true)
    if loader isa TransformInplace
        dataset.samples = [Sample(sample.sample, Compose(loader), sample.transform, sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
    else
        dataset.samples = [Sample(sample.sample, loader, sample.transform, sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
    end
end
"Adds a transform function to the end of existing transform pipeline."
function add_transform!(dataset::Dataset, transform, sample_filter=always_true)
    dataset.samples = [Sample(sample.sample, sample.loader, Compose(sample.transform, transform), sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
end
"Replaces the transform function"
function set_transform!(dataset::Dataset, transform, sample_filter=always_true)
    if transform isa TransformInplace
        dataset.samples = [Sample(sample.sample, sample.loader, Compose(transform), sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
    else
        dataset.samples = [Sample(sample.sample, sample.loader, transform, sample.label) for sample in dataset.samples if sample_filter(sample.sample)]
    end
end

"Create a dataset from a folder."
function DatasetLabeled(path_to_folder::String; loader, transform=identity, label, filter=always_true, extensions::Vector=[])
    ds = DatasetLabeled([], [])
    add_folder!(ds, path_to_folder, loader=loader, transform=transform, label=label, filter=filter, extensions=extensions)
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

"Split dataset into according to splits, for example 150-long dataset into [100, 50]. Returns a list of datasets."
function split_dataset(dataset::Dataset, splits::Base.AbstractVecOrTuple{Int})
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
        samples = dataset.samples[cur:min(cur + i, dataset_length)]
        cur += i
        ds = copy(dataset)
        ds.samples = samples
        push!(split_datasets, ds)
    end
    return split_datasets
end

"Split dataset into according to splits, for example [0.7, 0.2, 0.1] splits into 70%. 20% and 10%. Returns a list of datasets."
function split_dataset(dataset::Dataset, splits::Base.AbstractVecOrTuple{AbstractFloat})
    if sum(splits) != 1
        throw("Splits sum up to $(sum(splits)), they should sum up to 1")
    end
    dataset_length = length(dataset)
    splits = floor.(Int64, splits .* dataset_length)
    splits[end] += dataset_length - sum(splits)
    split_dataset(dataset, splits)
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

function mean_std(dataset::DatasetLabeled, batch_size::Int=32, n_samples=Inf)
    sample_size = size(getindex(dataset, 1)[1])
    n_channels = last(sample_size)
    mean_sum = zeros(n_channels)
    std_sum = zeros(n_channels)
    dl = MLUtils.DataLoader(dataset, batchsize=batch_size, shuffle=true, collate=true)
    total_values = 0
    for (sample, label) in dl
        dims = Vector(range(1, ndims(sample), step=1))
        mean_dims = filter!(!=(dims[end-1]), dims)
        mean_sum .+= vec(Statistics.mean(sample, dims=mean_dims))
        std_sum .+= vec(Statistics.std(sample, dims=mean_dims))
        total_values += 1

        n_samples -= batch_size
        if n_samples <= 0
            break
        end
    end

    return mean_sum ./ total_values, std_sum ./ total_values

end

# ---------------

# include("./image.jl")

# loader(path::String) = resize(to_channels(read_image(path), 3), (32, 32))

# ds = DatasetLabeled()

# add_folder!(ds, raw"C:\Users\nkshv\Pictures", loader = loader, label = last_folder, extensions = [".jpg", ".png", ".jpeg"])

# getindex(ds, [1,2,3])

# import MLUtils

# dl = MLUtils.DataLoader(ds, batchsize = 8, shuffle = true, collate = true)
