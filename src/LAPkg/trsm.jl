"""
    trsm!(
        side::Char, uplo::Char, tran::Char, diag::Char,
        m::Integer, n::Integer, a::Float64,
        A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64},
        lda::Integer, ldb::Integer
    )

where `A ∈ ℝ (lda × k)`, `B ∈ ℝ (ldb × n)`. `trsm!` solves one of the matrix equations
- `op(A) ⋅ X = aB`, or
- `X ⋅ op(A) = aB`,

where `a` is a scalar, `X ∈ ℝ (m × n)`, and `B ∈ ℝ (m × n)`, `A` is a unit, or non-unit, upper or lower triangular matrix, and `op(A)` is one of
- `op(A) = A` or
- `op(A) = Aᵀ`.

The matrix `X` is overwritten on `B`.

- `side`: On entry, `side` specifies whether `op(A)` appears on the left or right of `X` as follows:
    - `side = 'L'`: `op(A) ⋅ X = aB`.
    - `side = 'R'`: `X ⋅ op(A) = aB`.

- `uplo`: On entry, `uplo` specifies whether the matrix `A` is an upper or lower triangular matrix as follows:
    - `uplo = 'U'`: `A` is an upper triangular matrix.
    - `uplo = 'L'`: `A` is a lower triangular matrix.

- `tran`: On entry, `tran` specifies the form of `op(A)` to be used in the matrix multiplication as follows:
    - `tran = 'N'`: `op(A) = A`.
    - `tran = 'T'`: `op(A) = Aᵀ`.

- `diag`: On entry, `diag` specifies whether or not `A` is unit triangular as follows:
    - `diag = 'U'`: `A` is assumed to be unit triangular.
    - `diag = 'N'`: `A` is not assumed to be unit triangular.

- `m`: On entry, `m` specifies the number of rows of `B`. `m` must be at least zero.

- `n`: On entry, `n` specifies the number of columns of `B`. `n` must be at least zero.

- `a`: On entry, `a` specifies the scalar `a`. When `a` is zero, then `A` is not referenced, and `B` need not be set before entry.

- `A`, `size(A) = (lda, k)` where
    - `k` is `m` when `side = 'L'`
    - `k` is `n` when `side = 'R'`
    - Before entry with `uplo = 'U'`, the leading `k × k` upper triangular part of the array `A` must contain the upper triangular matrix, and the strictly lower triangular part of `A` is not referenced.
    - Before entry with `uplo = 'L'`, the leading `k × k` lower triangular part of the array `A` must contain the lower triangular matrix, and the strictly upper triangular part of `A` is not referenced.
    - Note that when `diag = 'U'`, the diagonal elements of `A` are not referenced either but are assumed to be unity.

- `lda`: On entry, `lda` specifies the first dimension of `A` as declared in the calling (sub) program. When `side = 'L'` then `lda` must be at least `max(1, m)`, when `side = 'R'`then `LDA` must be at least `max(1, n)`.

- `B`, `size(B) = (ldb, n)`: Before entry, the leading `m × n` part of the array `B` must contain the right-hand side matrix `B`, and on exit is overwritten by the solution matrix `X`.

- `ldb`: On entry, `ldb` specifies the first dimension of `B` as declared in the calling (sub) program. `ldb` must be at least `max(1, m)`.
"""
function trsm!(
        side::Char, uplo::Char, tran::Char, diag::Char,
        m::Integer, n::Integer, a::Float64,
        A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64},
        lda::Integer, ldb::Integer
    )
    lside = side ≡ 'L'
    nrowa = lside ? m : n
    nounit = diag ≡ 'N'
    upper  = uplo ≡ 'U'

    info = 0
    if !lside && !(side ≡ 'R')
        info = 1
    elseif !upper && !(uplo ≡ 'L')
        info = 2
    elseif !(tran ≡ 'N') && !(tran ≡ 'T') && !(tran ≡ 'C')
        info = 3
    elseif !(diag ≡ 'U') && !(diag ≡ 'N')
        info = 4
    elseif m < 0
        info = 5
    elseif n < 0
        info = 6
    elseif lda < max(1, nrowa)
        info = 9
    elseif ldb < max(1, m)
        info = 11
    end

    !iszero(info) && error("trsm! ", info)

    # Quick return if possible.
    (iszero(m) || iszero(n)) && error("")

    # a == 0.0
    if iszero(a)
        @inbounds for j in 1:n, i in 1:m
            B[i,j] = 0.0
        end
        return B
    end

    # Start the operations.
    if lside
        if tran ≡ 'N' # Form B = inv(A) * aB.
            if upper
                @inbounds for j in 1:n
                    if !isone(a)
                        for i in 1:m
                            B[i,j] *= a
                        end
                    end
                    for k in m:-1:1
                        if !iszero(B[k,j])
                            if nounit
                                B[k,j] /= A[k,k]
                            end
                            for i in 1:k-1
                                B[i,j] -= B[k,j] * A[i,k]
                            end
                        end
                    end
                end
            else
                @inbounds for j in 1:n
                    if !isone(a)
                        for i in 1:m
                            B[i,j] *= a
                        end
                    end
                    for k in 1:m
                        if !iszero(B[k,j])
                            if nounit
                                B[k,j] /= A[k,k]
                            end
                            for i in k+1:m
                                B[i,j] -= B[k,j] * A[i,k]
                            end
                        end
                    end
                end
            end
        else # Form B = inv(Aᵀ) * aB
            if upper
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = a * B[i,j]
                        for k in 1:i-1
                            temp -= A[k,i] * B[k,j]
                        end
                        if nounit
                            temp /= A[i,i]
                        end
                        B[i,j] = temp
                    end
                end
            else
                @inbounds for j in 1:n
                    for i in m:-1:1
                        temp = a * B[i,j]
                        for k in i+1:m
                            temp -= A[k,i] * B[k,j]
                        end
                        if nounit
                            temp /= A[i,i]
                        end
                        B[i,j] = temp
                    end
                end
            end
        end
    else
        if tran ≡ 'N' # Form B = aB * inv(A).
            if upper
                @inbounds for j in 1:n
                    if !isone(a)
                        for i in 1:m
                            B[i,j] *= a
                        end
                    end
                    for k in 1:j-1
                        if !iszero(A[k,j])
                            for i in 1:m
                                B[i,j] -= A[k,j] * B[i,k]
                            end
                        end
                    end
                    if nounit
                        temp = inv(A[j,j])
                        for i in 1:m
                            B[i,j] *= temp
                        end
                    end
                end
            else
                @inbounds for j in n:-1:1
                    if !isone(a)
                        for i in 1:m
                            B[i,j] *= a
                        end
                    end
                    for k in j+1:n
                        if !iszero(A[k,j])
                            for i in 1:m
                                B[i,j] -= A[k,j] * B[i,k]
                            end
                        end
                    end
                    if nounit
                        temp = inv(A[j,j])
                        for i in 1:m
                            B[i,j] *= temp
                        end
                    end
                end
            end
        else # Form B = aB * inv(Aᵀ).
            if upper
                @inbounds for k in n:-1:1
                    if nounit
                        temp = inv(A[k,k])
                        for i in 1:m
                            B[i,k] *= temp
                        end
                    end
                    for j in 1:k-1
                        if !iszero(A[j,k])
                            temp = A[j,k]
                            for i in 1:m
                                B[i,j] -= temp * B[i,k]
                            end
                        end
                    end
                    if !isone(a)
                        for i in 1:m
                            B[i,k] *= a
                        end
                    end
                end
            else
                @inbounds for k in 1:n
                    if nounit
                        temp = inv(A[k,k])
                        for i in 1:m
                            B[i,k] *= temp
                        end
                    end
                    for j in k+1:n
                        if !iszero(A[j,k])
                            temp = A[j,k]
                            for i in 1:m
                                B[i,j] -= temp * B[i,k]
                            end
                        end
                    end
                    if !isone(a)
                        for i in 1:m
                            B[i,k] *= a
                        end
                    end
                end
            end
        end
    end

    return B
end
