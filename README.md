# Conjugate Gradient examples

## Running on the CPU
To run the code on the CPU you need to setup your environment with
```bash
julia --project=@. -e "using Pkg; Pkg.instantiate()"
```
then the examples can be run as
```bash
julia --color=yes --project=@. FD_example.jl
```

## Running on NVIDIA GPUs
To run the code on the NVIDIA GPUs you need to setup your environment with
```bash
julia --project=env/cuda -e "using Pkg; Pkg.instantiate()"
```
then the examples can be run as
```bash
julia --color=yes --project=env/cuda FD_example.jl
```

## Running in the REPL
The code can also be run in the REPL with Julia launched using one of the
following:
```bash
julia --project=@.
julia --project=env/gpu
```
The first time you run the code need to setup the environment:
```julia
]instantiate
```
and subsequently the code can be run with
```julia
include("FD_example.jl")
```

Note: this has been tested with Julia 1.2.0
