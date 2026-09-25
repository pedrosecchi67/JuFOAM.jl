# JuFOAM.jl

A Field Operation and Manipulation package for highly-parallel, GPU-accelerated computation of fluid dynamics solutions.

## Domain construction

JuFOAM allows you to build finite volume domains from virtually any data structure with face geometry and connectivity information. You can specify it with:

```julia
dom = Domain(
    face_owners, # vector of indices of connecting cells
    face_neighbors,
    face_centers, # matrix, each row a face center
    face_neighbors, # similar, each row a face normal, norm = area
    "wall" => wall_face_indices, # vector of face indices
)

@show dom.face_normals # face normals
@show dom.face_centers # face centers
@show dom.face_areas # face areas
@show dom.centers # cell centers
@show dom.volumes # cell volumes
```

One may also specify immersed boundary method/non-body-fitted boundaries with:

```julia
dom = Domain(
    face_owners,
    face_neighbors,
    face_centers,
    face_normals,
    "wall" => wall_faces,
    "immersed_boundary" => (
        ib_faces, # vector of indices
        ib_face_projections # matrix, each row a face center proj. point on the surface
    ), # image points are located at twice the distance to the wall as the face centers
    "ib_with_distance" => (
        ib_faces2,
        ib_face_projections2,
        ib_image_point_distances # vector of floats
    )
)
```

Check out the Immersed Boundary Method implementation by [Francesco Capizzano](https://doi.org/10.2514/1.J050466?_gl=1*j9gcnw*_gcl_au*NDE5NjIxMDYuMTc4NjM5MDAwNw..*_ga*MTY4Nzk0ODcwNy4xNzcwNjAyNjQy*_ga_BFMKMMYM72*czE3OTAzMzQ3ODckbzIzJGcxJHQxNzkwMzM1MDA1JGo2MCRsMCRoMA..*_ga_GKPDRCLRFH*czE3OTAzMzQ3ODckbzIzJGcxJHQxNzkwMzM1MDA1JGo2MCRsMCRoMA..).

Partitioned domains may also be constructed with the p-METIS algorithm:

```julia
using Distributed

pdom = PartitionedDomain(
    args...;
    max_partition_size = 100_000, # default
    workers = procs() # optionally, run in parallel
)

# arbitrary field property:
u = rand(length(pdom))
# arbitrary property at a family:
uw = rand(length(pdom, "wall"))

# note that all field property arrays should have
# the first dimension as an index identifying a 
# cell or face.

pdom(u, "wall" => uw) do subdomain, u, (bname, uw)
    # here, we can calculate anything at the 
    # subdomain level, and edit the passed 
    # arrays in-place

    r # return values are also captured and returned in a vector
end
```

One may also easily port domain and field property array information to a custom array backend, such as `CuArray` for CUDA GPU acceleration:

```julia
using CUDA

pdom = PartitionedDomain(
    args...;
    conv_to_backend = x -> CuArray(x),
    conv_from_backend = x -> Array(x),
    lazy_conversion = false
)

u = rand(length(pdom))

pdom(u) do subdom, u
    @show typeof(u)
    # CuArray
    @show typeof(subdom.centers)
    # also CuArray :)
end
```

If `lazy_conversion = true`, the domain's geometric information is converted to the backend at each residual calculation function call and discarded at its end to avoid overloading GPU VRAM in single-GPU systems.

With serial, non-partitioned domains, one may also perform backend conversion and port the entire domain to the backend at once:

```julia
dom = to_backend(
    dom, x -> CuArray(x)
)

u = rand(length(dom))
u = to_backend(u, x -> CuArray(x))
```

### Unstructured (polyhedral) grids

I'm sure you don't want to implement your own face normal calculation function every time, so let's work with a `PolyhedralMesh` data structure.

```julia
using JuFOAM.UnstructuredGrids

points = [
    0.0 0.0;
    1.0 0.0;
    1.0 1.0;
    0.0 1.0
] # corner points of a square
faces = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 1],
    [1, 3]
] # edges and diagonal
cells = [
    [1, 2, 5],
    [3, 4, 5]
] # two triangles subdividing a square
families = Dict(
    "wall" => [1, 2] # families as face indices
)

msh = PolyhedralMesh(
    points, faces, cells, families
)

dom = Domain(msh) # other constructors are also supported
```

Immersed boundary information may also be specified:

```julia
set_boundary_projections!(
    msh,
    "family_name",
    face_center_projections # matrix, each row a face
)

set_image_distances!(
    msh,
    "family_name",
    image_point_distances # vector with distances between boundary and image points
)
```

You can easily create block-structured grids and join them:

```julia
msh1 = PolyhedralMesh(
    x, y, z; # grid arrays with coordinates
    families = [
        "inlet" => [ # block faces:
            (1, false), # x-axis, back
            (2, false), # y-axis, left
            (2, true), # y-axis, right
            (3, false), # z-axis, bottom
            (3, true) # z-axis, top
        ]
    ]
)
msh2 = PolyhedralMesh(
    [1.0, 0.0, 0.0], # hypercube origin
    [1.0, 1.0, 1.0], # hypercube widths along each axis
    (100, 100, 100); # grid size along each axis
    families = [
        "inlet" => [
            (2, false), (2, true),
            (3, false), (3, true)
        ],
        "outlet" => [
            (1, true)
        ] # other faces are set as ORPHAN
    ]
)

msh = PolyhedralMesh(
    msh1, msh2; tolerance = 1f-7 # for point and face merging
)
```

## Residual calculation and BC imposition

```julia
u = rand(length(dom), 5) # arbitrary tensor
uw = rand(length(dom, "wall"), 5) # its values at a family

r = similar(u)
r .= 0 # let's store residuals

pdom(r, u, "wall" => uw) do dom, r, u, (bname, uw)
    bdry = dom.boundaries[bname]

    # at face owners and neighbors:
    uo = at_owners(dom, u)
    un = at_neighbors(dom, u)

    # at faces (interpolated):
    uf = at_faces(dom, u)

    # let's impose BCs:
    uimage = at_images(bdry, u)
    # from cell data, calculate image point values

    # calculate arbitrary Neumann condition
    du!dn = 1.0f0
    ubdry = uimage .- du!dn .* bdry.distances

    #=
    other properties at boundaries:

    bdry.normals # matrix
    bdry.distances # distance to boundary
    bdry.projections # matrix, projs. on boundary
    =#

    # now we impose values at boundary faces
    # by getting a view to them and 
    # altering it in-place:
    at_boundary(bdry, uf) .= ubdry

    # for immersed boundaries, let's interpolate
    # properties between image points and 
    # the boundary:
    at_boundary(bdry, uf) .= interp2boundary(
        bdry, ubdry, uimage
    )

    # from arbitrary flow velocities at cells:
    uvw = rand(length(dom), ndims(dom))
    # calculate face fluxes:
    fluxes = sum(
        uvw .* dom.face_normals; dims = 2
    ) |> vec

    # calculate advective fluxes for tensor u:
    F = uf .* fluxes
    # obtain Green-Gauss divergent:
    divF = green_gauss(dom, F)

    # we also have unsigned face integrations for
    # CFL number calculations:
    CFL = green_gauss(dom, fluxes; signed = false) |> maximum

    # cell gradients:
    grad_u = gradient(dom, uf) # returns tuple
    du!dy = gradient(dom, uf) # along specific dimension

    # (note that we calculate from face properties with BC information!!!)

    # face gradients, corrected by owner
    # and neighbor information:
    
    # (hypothetical Dirichlet BC at boundary)
    at_boundary(bdry, uo) .= 0
    grad_u_at_faces = face_gradient(
        dom, grad_u, uo, un
    ) # (also returns tuple)

    # divergent() is a shorthand for the divergent of 
    # a vector field, given as a tuple of 
    # values at faces
    laplacian_u = divergent(dom, grad_u_at_faces)

    # store residuals in-place:
    r .= laplacian_u

    ; # be careful not to accidentally transport vector parts around as return values
end
```

## Multigrid (Full-Approximation Scheme)

Non-linear, fixed-point iterations with multigrid agglomeration may be performed with:

```julia
using JuFOAM.Solver

dom, coarse_domains, coarseners, prolongators = MultigridDomain(
    n_levels, # number of multigrid levels
    msh;
    n_coarsening_iters = 2, # defaults to ndim(domain)
    partitioned = true, # def. false
    kwargs... # all other kwargs passed to PartitionedDomain
)

f = (level, u) -> begin
    # at multigrid level "level"
    # (may be 0 for domain `dom`)
    # calculate r such that we may perform
    # the fixed-point iteration:
    # u += ω * r

    (r, ω)
end

FAS!(
    f, u;
    coarseners = coarseners,
    prolongators = prolongators,
    n_cycles = 4, n_iter = 10,
    rtol = 1f-2, atol = 1f-7
)
```

## CFD utilities

Check out the docstrings for the following functions and structs:

```julia
using JuFOAM.CFD

Fluid
speed_of_sound
dynamic_viscosity
heat_conductivity
primitive2state
state2primitive
FlowBC
ISA_atmosphere
streamwise_direction
pressure_coefficient
inviscid_fluxes
viscous_fluxes
Reynolds_number
adjust_Reynolds
TimeAverage

using JuFOAM.Turbulence

wall_function
shear_rate
Smagorinsky_νSGS
WALE_νSGS
Wray_Agarwal
standard_kϵ
Ducros_sensor
```
