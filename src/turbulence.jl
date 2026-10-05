module Turbulence

    using DocStringExtensions

    export wall_function

    """
    $TYPEDSIGNATURES

    Von-Karman law of the wall
    """
    von_Karman(
        y⁺;
        κ::Real = 0.41f0, C::Real = 4.9f0
    ) = (
        @. min(log(max(y⁺, 1.0f0)) / κ + C, y⁺)
    )

    """
    $TYPEDSIGNATURES

    Run fixed-point iteration and figure out BL data
    from `Rey = y × u / ν = y⁺ × u⁺`.

    Returns tuple with fields `y⁺, u⁺, μ⁺, k⁺, du⁺!dy⁺`.
    """
    function wall_function(
        Rey::AbstractVector;
        κ::Real = 0.41f0, C::Real = 4.9f0, A::Real = 19.0f0,
        β::Real = 0.075f0, βstar::Real = 0.09f0,
        D::Real = 4.2f0, A⁺::Real = 360.0f0,
        ω_fixed_point::Real = 0.5f0, n_iter::Int = 20,
    )
        ϵ = eps(eltype(Rey))
        Rey = @. clamp(abs(Rey), ϵ, Inf32)

        y⁺ = sqrt.(Rey) # start from laminar
        u⁺ = similar(y⁺)

        for _ = 1:n_iter
            u⁺ .= von_Karman(y⁺; κ = κ, C = C)
            @. y⁺ = ω_fixed_point * (Rey / u⁺) + (1.0f0 - ω_fixed_point) * y⁺
        end

        u⁺ .= Rey ./ y⁺

        # from van Driest
        μ⁺ = @. κ * y⁺ * (1.0f0 - exp(- y⁺ / A)) ^ 2
        du⁺!dy⁺ = @. 1.0f0 / (1.0f0 + μ⁺)

        # from Nakagawa-Nezu
        k⁺ = @. min(
            y⁺ ^ 2 / (6.0f0 * βstar / β - 2.0f0),
            D * exp(
                - y⁺ / A⁺
            )
        )

        (
            y⁺ = y⁺, 
            u⁺ = u⁺, 
            μ⁺ = μ⁺,
            k⁺ = k⁺,
            du⁺!dy⁺ = du⁺!dy⁺,
        )
    end

    """
    $TYPEDSIGNATURES

    Obtain named tuple with `uτ, νₜ, k, ω, ϵ, du!dn` given a set of `y, u, ν` 
    values. Takes the same kwargs as other methods for the same function. 
    """
    function wall_function(
        y::AbstractVector, u::AbstractVector, ν::AbstractVector;
        βstar::Real = 0.09f0,
        kwargs...
    )
        nt = wall_function(u .* y ./ ν; kwargs...)

        uτ = u ./ nt.u⁺

        νₜ = nt.μ⁺ .* ν
        k = nt.k⁺ .* uτ .^ 2
        ω = k ./ νₜ
        ϵ = @. βstar * ω * k

        du!dn = @. nt.du⁺!dy⁺ * uτ ^ 2 / ν

        (
            uτ = uτ,
            νₜ = νₜ,
            k = k,
            ω = ω,
            ϵ = ϵ,
            du!dn = du!dn,
        )
    end

    export shear_rate

    """
    $TYPEDSIGNATURES

    Obtain `sqrt(2 * SijSij)`, if `Sij = (∇u + ∇uᵀ) / 2` is the strain-rate tensor.

    `velocity_gradient` is a matrix such that `velocity_gradient[i, j]`
    indicates the gradient of vel. component `i` along dimension `j`.
    """
    function shear_rate(
        velocity_gradient::AbstractMatrix
    )
        SijSij = similar(velocity_gradient[1, 1])
        SijSij .= 0
        for i = 1:size(velocity_gradient, 1)
            for j = 1:size(velocity_gradient, 2)
                SijSij .+= (
                    (velocity_gradient[i, j] .+ velocity_gradient[j, i]) ./ 2
                ) .^ 2
            end
        end

        @. sqrt(2 * SijSij)
    end

    export vorticity_magnitude

    """
    $TYPEDSIGNATURES

    Obtain `sqrt(2 * WijWij)`, if `Wij = (∇u - ∇uᵀ) / 2`.

    `velocity_gradient` is a matrix such that `velocity_gradient[i, j]`
    indicates the gradient of vel. component `i` along dimension `j`.
    """
    function vorticity_magnitude(
        velocity_gradient::AbstractMatrix
    )
        WijWij = similar(velocity_gradient[1, 1])
        WijWij .= 0
        for i = 1:size(velocity_gradient, 1)
            for j = 1:size(velocity_gradient, 2)
                WijWij .+= (
                    (velocity_gradient[i, j] .- velocity_gradient[j, i]) ./ 2
                ) .^ 2
            end
        end

        @. sqrt(2 * WijWij)
    end

    export Smagorinsky_νSGS

    """
    $TYPEDSIGNATURES

    Obtain `νSGS` as per the Smagorinsky turbulence model.
    `S` is the norm of vorticity, `sqrt(2 * SijSij)`
    """
    Smagorinsky_νSGS(
        Δ::AbstractVector, S::AbstractVector;
        Cₛ::Real = 0.17f0,
    ) = (@. (Cₛ * Δ) ^ 2 * S)

    export standard_kϵ

    """
    $TYPEDSIGNATURES

    Closure for the standard k-ϵ turbulence model.

    Returns named tuple with fields:

    ```
    (
        Sk = (source term for k),
        Sϵ = (source term for ϵ),
        νk = (dissipation rate for k),
        νϵ = (dissipation rate for ϵ),
        νₜ = (eddy viscosity)
    )
    ```

    Such that:

    ```
    kₜ = - ∇⋅(uk) + ∇⋅[(ν + νk) ∇k] + Sk
    ϵₜ = - ∇⋅(uϵ) + ∇⋅[(ν + νϵ) ∇ϵ] + Sϵ
    ```

    Does not account for Buoyancy.

    Remember: apply `k = ϵ = 0` at walls, and
    ````
    k∞ = 3 * (U∞ Tu) ^ 2 / 2
    ϵ∞ = Cμ * k∞ ^ 2 / (3 * ν∞)
    Cμ = 0.09
    Tu ≈ 0.10
    ```
    """
    function standard_kϵ(
        k::Union{AbstractVector, Real}, ϵ::Union{AbstractVector, Real}, S::Union{AbstractVector, Real};
        Cμ::Real = 0.09f0, σk::Real = 1.0f0, σϵ::Real = 1.3f0,
        C1ϵ::Real = 1.44f0, C2ϵ::Real = 1.92f0,
    )
        νₜ = @. Cμ * k ^ 2 / max(ϵ, 1f-14)

        Pk = @. νₜ * S ^ 2

        Sk = @. Pk - ϵ
        Sϵ = @. (C1ϵ * Pk * ϵ - C2ϵ * ϵ ^ 2) / max(k, 1f-14)

        (
            νk = νₜ ./ σk,
            νϵ = νₜ ./ σϵ,
            Sk = Sk,
            Sϵ = Sϵ,
            νₜ = νₜ,
        )
    end

    export Wray_Agarwal

    """
    $TYPEDSIGNATURES

    Obtain closure for a 'simplified' Wray-Agarwal turbulence model
    which collapses all constants to the `k-ω` values.

    Remember: BCs involve using `R∞ = 3ν` and `R=0` at walls!

    The return value is a tuple with entries:

    ```
    (
        νₜ = (eddy viscosity),
        νR = (dissipation rate for R),
        S = (source term)
    )
    ```

    Such that:

    ```
    Rₜ = - ∇⋅(uR) + ∇⋅[νR ∇R] + S
    ```
    """
    function Wray_Agarwal(
        ν::AbstractVector,
        R::AbstractVector, S::AbstractVector,
        ∇R::AbstractMatrix, ∇S::AbstractMatrix;
        σR::Real = 0.72f0, C₁::Real = 0.0829f0, κ::Real = 0.41f0,
        Cω::Real = 8.54f0,
    )
        ϵ = eps(eltype(R))

        χ = @. R / ν

        C₂ = σR + C₁ / κ ^ 2

        S = let ∇R∇S = sum(∇R .* ∇S; dims = 2) |> vec
            @. C₁ * R * S + C₂ * ∇R∇S * (R / (S + ϵ))
        end
        @. S = min(S, 10.0f0 * R)

        νₜ = @. χ ^ 3 / (χ ^ 3 + Cω ^ 3) * R

        (
            νₜ = νₜ,
            νR = R .* σR .+ ν,
            S = S
        )
    end

    export Spallart_Allmaras

    """
    $TYPEDSIGNATURES

    Spallart-Allmaras one-equation turbulence model.

    `ν` is the kinematic viscosity, `d` is the nearest wall distance,
    `ν̂`` is the working variable, and `Ω` is the vorticity magnitude.

    The return value is a tuple with entries:

    ```
    (
        νₜ = (eddy viscosity),
        νSA = (dissipation rate),
        S = (source term)
    )
    ```

    Such that the model can be written as:

    ```
    ν̂ₜ = - ∇⋅(uν̂ ) + ∇⋅[νSA ∇ν̂ ] + S
    ```

    If cell size `Δ` is provided, the destruction term is modified to its DES form.
    """
    function Spallart_Allmaras(
        ν::AbstractVector, d::AbstractVector, Ω::AbstractVector,
        ν̂::AbstractVector, ∇ν̂::AbstractMatrix, 
        Δ::Union{Nothing, AbstractVector} = nothing;
        σ::Real = 2.0f0 / 3, 
        Cb1::Real = 0.1355f0, Cb2::Real = 0.622f0,
        κ::Real = 0.41f0,
        Cw2::Real = 0.3f0, Cw3::Real = 2.0f0,
        Cv1::Real = 7.1f0, Ct3::Real = 1.2f0, Ct4::Real = 0.5f0,
        CDES::Real = 0.65f0,
    )
        Cw1 = Cb1 / κ ^ 2 + (1.0f0 + Cb2) / σ

        χ = @. ν̂  / ν

        fv1 = @. χ ^ 3 / (χ ^ 3 + Cv1 ^ 3)
        fv2 = @. 1.0f0 - χ / (1.0f0 + χ * fv1)
        ft2 = @. Ct3 * exp(-Ct4 * χ ^ 2)

        Ŝ = @. min(Ω + ν̂  * fv2 / (d * κ) ^ 2, 0.3f0 * Ω)

        r = @. min(ν̂  / (Ŝ * (d * κ) ^ 2), 10.0f0)
        g = @. r + Cw2 * (r ^ 6 - r)

        fw = @. g * ( (1.0f0 + Cw3 ^ 6) / (g ^ 6 + Cw3 ^ 6) ) ^ (1.0f0 / 6)

        mod2∇ν̃  = sum(∇ν̂ .* ∇ν̂; dims = 2) |> vec

        νₜ = @. ν̂  * fv1
        νSA = @. (ν + ν̂ ) / σ

        # source term
        S = @. Cb1 * (1.0f0 - ft2) * Ŝ * ν̂  + mod2∇ν̃  * Cb2 / σ
        # destruction term
        d̃ = d
        if !isnothing(Δ)
            d̃ = @. min(d, CDES * Δ)
        end
        @. S += (- (Cw1 * fw - Cb1 * ft2 / κ ^ 2) * (ν̂  / d̃) ^ 2)

        (
            νₜ = νₜ,
            νSA = νSA,
            S = S
        )
    end

    export Ducros_sensor

    """
    $TYPEDSIGNATURES

    Ducros shock sensor based on matrix of velocity gradients.

    `velocity_gradient` is a matrix such that `velocity_gradient[i, j]`
    indicates the gradient of vel. component `i` along dimension `j`.
    """
    function Ducros_sensor(
        velocity_gradient::AbstractMatrix
    )
        ϵ = eps(eltype(velocity_gradient[1, 1]))

        curl2 = similar(velocity_gradient[1, 1])
        div2 = similar(velocity_gradient[1, 1])
        nd = size(velocity_gradient, 1)

        div2 .= 0
        for i = 1:nd
            div2 .+= velocity_gradient[i, i]
        end
        div2 .^= 2

        if nd == 2
            curl2 .= (
                velocity_gradient[2, 1] .- velocity_gradient[1, 2]
            ) .^ 2
        elseif nd == 3
            curl2 .= (
                (velocity_gradient[3, 2] .- velocity_gradient[2, 3]) .^ 2 .+
                (velocity_gradient[1, 3] .- velocity_gradient[3, 1]) .^ 2 .+
                (velocity_gradient[2, 1] .- velocity_gradient[1, 2]) .^ 2
            )
        else
            error("Ducros sensor only implemented for 2D and 3D")
        end

        @. (div2 + ϵ) / (div2 + curl2 + ϵ)
    end

    export WALE_νSGS

    """
    $TYPEDSIGNATURES

    Wall-Adapting Local Eddy-viscosity (WALE) model for subgrid-scale viscosity.
    """
    function WALE_νSGS(
        Δ::AbstractVector, velocity_gradient::AbstractMatrix;
        Cw::Real = 0.325f0,
    )
        ϵ = eps(eltype(velocity_gradient[1, 1]))
        nd = size(velocity_gradient, 1)

        @assert nd == 3 "WALE model only implemented for 3D"

        g = velocity_gradient
        g2 = Matrix{AbstractVector}(undef, nd, nd)
        for i = 1:nd
            for j = 1:nd
                s = similar(g[i, j])
                s .= 0

                for k = 1:nd
                    s .+= g[i, k] .* g[k, j]
                end
                g2[i, j] = s
            end
        end

        SijSij = similar(velocity_gradient[1, 1])
        SijSij .= 0
        for i = 1:nd
            for j = 1:nd
                SijSij .+= (
                    (g[i, j] .+ g[j, i]) ./ 2
                ) .^ 2
            end
        end

        SdijSdij = similar(velocity_gradient[1, 1])
        SdijSdij .= 0
        for i = 1:nd
            for j = 1:nd
                δ = (i == j)
                SdijSdij .+= (
                    (g2[i, j] .+ g2[j, i]) ./ 2 .- g2[i, j] .* (δ / 3)
                ) .^ 2
            end
        end

        @. Cw * Δ ^ 2 * (SdijSdij ^ 1.5f0) / (SijSij ^ 2.5f0 + SdijSdij ^ 1.25f0 + ϵ)
    end

end
