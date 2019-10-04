using LinearAlgebra
using GPUifyLoops
using Logging
using Printf

# Based on CG algorithm from
"""
    cg_iteration!(x, d, g, w, y_Ax!, gTg)

Compute one more iteration of the cg algorithm `x` is the current solution, `g`
is the current residual, `w` is some scratch space, `d` is the step direction.
The function `y_Ax!(y, x)` computes the multiplication calculation `y .= A*x`.
`gTg` is the current values of `dot(g,g)`.

The arrays `x`, `d`, `w`, and `g` are all modified by the function and new value
of `dot(g,g)` is returned.

@book{RauberRüber2013,
  title={Parallel Programming},
  author={Rauber, Thomas and Rünger, Gudula},
  year={2013},
  publisher={Springer},
  doi={10.1007/978-3-642-04818-0}
}
"""
function cg_iteration!(x, d, g, w, y_Ax!, gTg)
  y_Ax!(w, d)
  dTw = dot(d, w)
  alpha = gTg / dTw
  x .+= alpha .* d
  g .+= alpha .* w
  g1Tg1 = dot(g, g)
  beta = g1Tg1 / gTg
  d .= .-g .+ beta .* d
  x .= x
  return g1Tg1
end

function cg!(x, b, y_Ax!, tol, maxiteration=length(x))
  g = similar(x)
  d = similar(x)
  w = similar(x)

  g .= 0
  d .= 0
  w .= 0

  y_Ax!(g, x)
  g .-= b
  d .= -g

  gTg = dot(g, g)

  start_time = time()
  for k = 1:maxiteration
    if gTg < tol^2
      end_time = time()
      total_time = end_time - start_time
      return gTg, k, total_time
    end

    gTg = cg_iteration!(x, d, g, w, y_Ax!, gTg)
  end

  end_time = time()
  total_time = end_time - start_time
  return gTg, maxiteration+1, total_time
end

# Test with a simple matrix
let
  tol = 1e-6

  # matrix size
  N = 10

  # create random postive definite matrix
  (Q, _) = qr(rand(N, N))
  A = Q' * Diagonal(rand(N)) * Q

  # matrix vector multiply function
  y_Ax!(y, x) = y .= A*x

  # random RHS vector
  b = rand(N)

  # initial condition
  x = zeros(N)

  # do the cg iteration
  cg!(x, b, y_Ax!, tol)

  # check that the solution converged
  @assert norm(A*x - b) < tol
end

using Pkg
@static if haskey(Pkg.installed(), "CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

"""
    k_fd!(Au, u, ::Val{N}) where N

Computes the 3D, 2nd order central difference stencil with strong enforcement of
boundary data which is already set of u
"""
function k_fd!(Au, u, ::Val{N}, ::Val{Δ}) where {N, Δ}
  # Loop over interion and apply update
  # threads shifted by one on the GPU to match the bounds

  # The @loop macro translates the for loop to different code on the CPU and
  # GPU.
  #
  # On the CPU this translates to:
  #
  #    for k = 2:N[3]
  #      {rest of the code}
  #     end
  #
  # On the GPU this translates to:
  #
  #    k = (blockIdx().z-1) * blockDim().z + threadIdx().z+1
  #    if k in 2:N[3]
  #      {rest of the code}
  #     end
  #
  # Note that the +1 at the end is for shifting the threads to start at 2 and
  # not 1
  @inbounds @loop for k in (2:N[3];
                            (blockIdx().z-1) * blockDim().z + threadIdx().z+1)
    @loop for j in (2:N[2];
                    (blockIdx().y-1) * blockDim().y + threadIdx().y+1)
      @loop for i in (2:N[1];
                      (blockIdx().x-1) * blockDim().x + threadIdx().x+1)
        # Compute the centeral FD stencil assuming h = 1 and that the Dirchlet
        # boundary data is already set in u
        tmp = -zero(eltype(Au))
        tmp += (u[i-1, j, k] - 2u[i, j, k] + u[i+1, j, k]) / Δ[1]^2
        tmp += (u[i, j-1, k] - 2u[i, j, k] + u[i, j+1, k]) / Δ[2]^2
        tmp += (u[i, j, k-1] - 2u[i, j, k] + u[i, j, k+1]) / Δ[3]^2
        Au[i, j, k] = tmp
      end
    end
  end
end

function fd!(Au, u, Δ)
  device = typeof(u) <: Array ? CPU() : CUDA()
  N = size(Au) .- 1

  threads = (32, 16, 1)
  blocks = div.((N .- 1) .+ threads .- 1, threads)
  @launch(device, threads=threads, blocks=blocks, k_fd!(Au, u, Val(N), Val(Δ)))
end

# Test with a simple matrix
let
  tol = 1e-6

  # number of levels determined by size of error array
  err = zeros(8)
  rate = zeros(length(err)-1)
  for k = 1:length(err)
    # problem size
    N = (1, 2, 3) .* (2, 2, 2).^k

    # grid spacing
    Δ = 1 ./ N

    # create the mesh
    x = kron(ones(N[3]+1), ones(N[2]+1), range(0; stop = 1, length = N[1]+1))
    y = kron(ones(N[3]+1), range(0; stop = 1, length = N[2]+1), ones(N[1]+1))
    z = kron(range(0; stop = 1, length = N[3]+1), ones(N[2]+1), ones(N[1]+1))

    x, y, z = reshape.((x, y, z), N[1]+1, N[2]+1, N[3]+1)

    # Exact solution
    uex = exp.(x) .* exp.(y) .* cos.(√2z)

    # Exact boundary data
    u = copy(uex)
    u[2:end-1, 2:end-1, 2:end-1] .= 0

    # compute the influence of the boundary conditions
    b = zeros(size(uex))
    fd!(b, u, Δ)
    @. b = -b

    # Set the initial condition
    u .= 0

    # Copy the arrays to the device
    d_u = ArrayType(u)
    d_b = ArrayType(b)

    # Compute conjugate gradient algorithm
    _, iterations, total_time = cg!(d_u, d_b, (y, x)->fd!(y, x, Δ), tol)

    # Copy the solution back from the device
    u .= d_u

    # fill in the boundary data
    u[  1,   :,   :] = uex[  1,   :,   :]
    u[end,   :,   :] = uex[end,   :,   :]
    u[  :,   1,   :] = uex[  :,   1,   :]
    u[  :, end,   :] = uex[  :, end,   :]
    u[  :,   :,   1] = uex[  :,   :,   1]
    u[  :,   :, end] = uex[  :,   :, end]

    # check the error
    err[k] = √prod(Δ) * norm(u - uex)

    if k > 1
      rate[k-1] = log2(err[k-1]./ err[k])
      @info @sprintf """
      mesh size (%d, %d, %d)
        error              = %e
        rate               = %e
        elapsed time       = %e s
        time per iteration = %e s
      """ N[1]  N[2]  N[3] err[k] rate[k-1] total_time total_time / iterations
    else
      @info @sprintf """
      mesh size (%d, %d, %d)
        error              = %e
        elapsed time       = %e s
        time per iteration = %e s
      """ N[1]  N[2]  N[3] err[k] total_time total_time / iterations
    end
  end

  # Since we use grid doubling this estimates the rate

  @show err
  @show rate
end

nothing
