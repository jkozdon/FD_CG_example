using LinearAlgebra
using GPUifyLoops

using Pkg
@static if haskey(Pkg.installed(), "CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

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

  y_Ax!(g, x)
  g .-= b
  d .= -g

  gTg = dot(g, g)

  for k = 1:maxiteration
    if gTg < tol^2
      return gTg, k
    end

    gTg = cg_iteration!(x, d, g, w, y_Ax!, gTg)
  end

  return gTg, maxiteration+1
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
