include("./image.jl")
include("./data.jl")

loader(path::String) = resize(to_channels(read_image(path), 3), (32, 32))
ds = DatasetLabeled()
add_folder!(ds, raw"C:\Users\nkshv\Pictures", loader = loader, label = last_folder, extensions = [".jpg", ".png", ".jpeg"])

preload!(ds)
import MLUtils
dl = MLUtils.DataLoader(ds, batchsize = 8, shuffle = true, collate = true)

using Flux, CUDA, Statistics, ProgressMeter
model = Chain(
    Conv((3, 3), 3=>16, relu), # 29?
    MaxPool((2, 2)), # 14?
    Conv((3, 3), 16=>32, relu), # 12?
    MaxPool((2, 2)), # 6x6
    Flux.flatten,
    Dense(32 * 6 * 6, length(ds.labels)),
    softmax
) |> gpu

batch = first(dl)

out1 = model(batch[1] |> gpu) |> cpu

losses = []

optim = Flux.setup(Flux.Adam(0.01), model)
@showprogress for epoch in 1:10
    for (images, labels) in dl
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(images |> gpu)
            Flux.crossentropy(y_hat, labels |> gpu)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
        #println(loss)
    end
end

optim

out1 = model(batch[1] |> gpu) |> cpu

using Plots
plot(losses)