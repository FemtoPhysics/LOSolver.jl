"""
    syrk!(
        uplo::Char, tran::Char,
        n::Integer, k::Integer, a::Float64,
        A::AbstractMatrix{Float64}, lda::Integer,
        b::Float64, C::AbstractMatrix{Float64}, ldc::Integer
    )
    

where `A ∈ ℝ (lda × *)`, `C ∈ ℝ (ldc × *)`. `syrk!` performs one of the symmetric rank `k` operations
- `C = a⋅A⋅Aᵀ + b⋅C`, or
- `C = a⋅Aᵀ⋅A + b⋅C`,

where `a` and `b` are scalars, `C ∈ ℝ (n × n)` is a symmetric matrix, and `A ∈ ℝ (n × k)` in the first case and `A ∈ ℝ (k × n)` in the second case.

- `uplo`: On entry, `uplo` specifies whether the upper or lower triangular part of the array `C` is to be referenced as follows:
    - `uplo = 'U'`: Only the upper triangular part of `C` will be referenced.
    - `uplo = 'L'`: Only the lower triangular part of `C` will be referenced.

- `tran`: On entry, `tran` specifies the operation to be performed as follows:
    - `tran = 'N'`: `C = a⋅A⋅Aᵀ + b⋅C`.
    - `tran = 'T'`: `C = a⋅Aᵀ⋅A + b⋅C`.

- `n`: On entry, `n` specifies the order of the matrix `C`. `n` must be at least zero.

- `k`: On entry with `tran = 'N'`, `k` specifies the number of columns of the matrix `A`, and on entry with `tran = 'T'`, `k` specifies the number of rows of the matrix `A`. `k` must be at least zero.

- `a`: On entry, `a` specifies the scalar `a`.

- `A`, `size(A) = (lda, ka)` where
    - `ka` is `k` when `tran = 'N'`
    - `ka` is `n` when `tran = 'T'`
    - Before entry with `tran = 'N'`, the leading `n × k` part of the array `A` must contain the matrix `A`.
    - Before entry with `tran = 'T'`, the leading `k × n` part of the array `A` must contain the matrix `A`.

- `lda`: On entry, `lda` specifies the first dimension of `A` as declared in the calling (sub) program. When `tran = 'N'`, then `lda` must be at least `max(1, n)`; otherwise `lda` must be at least `max(1, k)`.

- `b`: On entry, `b` specifies the scalar `b`.

- `C`, `size(C) = (ldc, n)` where
    - Before entry with `uplo = 'U'`, the leading `n × n` upper triangular part of the array `C` must contain the upper triangular part of the symmetric matrix, and the strictly lower triangular part of `C` is not referenced. On exit, the upper triangular part of the array `C` is overwritten by the upper triangular part of the updated matrix.
    - Before entry with `uplo = 'L'` , the leading `n × n` lower triangular part of the array `C` must contain the lower triangular part of the symmetric matrix, and the strictly upper triangular part of `C` is not referenced. On exit, the lower triangular part of the array `C` is overwritten by the lower triangular part of the updated matrix.

- `ldc`: On entry, `ldc` specifies the first dimension of `C` as declared in the calling (sub) program. `ldc` must be at least `max(1, n)`.
"""
function syrk!(
        uplo::Char, tran::Char,
        n::Integer, k::Integer, a::Float64,
        A::AbstractMatrix{Float64}, lda::Integer,
        b::Float64, C::AbstractMatrix{Float64}, ldc::Integer
    )
    nrowa = tran ≡ 'N' ? n : k
    upper = uplo ≡ 'U'
    
    info = 0
    if !upper && !(uplo ≡ 'L')
        info = 1
    elseif !(tran ≡ 'N') && !(tran ≡ 'T')
        info = 2
    elseif (n < 0)
        info = 3
    elseif (k < 0)
        info = 4
    elseif lda < max(1, nrowa)
        info = 7
    elseif ldc < max(1, n)
        info = 10
    end

    !iszero(info) && error("syrk! ", info)
    # Quick return if possible.
    (iszero(n) || (iszero(a) || iszero(k)) && isone(b)) && error("")
    # And when a = 0.
    if iszero(a)
        if upper
            if iszero(b)
                @inbounds for j in 1:n, i in 1:j
                    C[i,j] = 0.0
                end
            else
                @inbounds for j in 1:n, i in 1:j
                    C[i,j] *= b
                end
            end
        else
            if iszero(b)
                @inbounds for j in 1:n, i in j:n
                    C[i,j] = 0.0
                end
            else
                @inbounds for j in 1:n, i in j:n
                    C[i,j] *= b
                end
            end
        end
        return C
    end
    # Start the operations.
    if tran ≡ 'N'
        # Form  C = a⋅A⋅Aᵀ + b⋅C.
        if upper
            for j in 1:n
                @inbounds if iszero(b)
                    for i in 1:j
                        C[i,j] = 0.0
                    end
                elseif !isone(b)
                    for i in 1:j
                        C[i,j] *= b
                    end
                end
                @inbounds for l in 1:k
                    if !iszero(A[j,l])
                        temp = a * A[j,l]
                        for i in 1:j
                            C[i,j] += temp * A[i,l]
                        end
                    end
                end
            end
        else
            for j in 1:n
                @inbounds if iszero(b)
                    for i in j:n
                        C[i,j] = 0.0
                    end
                elseif !isone(b)
                    for i in j:n
                        C[i,j] *= b
                    end
                end
                @inbounds for l in 1:k
                    if !iszero(A[j,l])
                        temp = a * A[j,l]
                        for i in j:n
                            C[i,j] += temp * A[i,l]
                        end
                    end
                end
            end
        end
    else
        # Form  C = a⋅Aᵀ⋅A + b⋅C.
        if upper
            for j in 1:n
                @inbounds for i in 1:j
                    temp = 0.0
                    for l in 1:k
                        temp += A[l,i] * A[l,j]
                    end
                    if iszero(b)
                        C[i,j] = a * temp
                    else
                        C[i,j] = a * temp + b * C[i,j]
                    end
                end
            end
        else
            for j in 1:n
                @inbounds for i in j:n
                    temp = 0.0
                    for l in 1:k
                        temp += A[l,i] * A[l,j]
                    end
                    if iszero(b)
                        C[i,j] = a * temp
                    else
                        C[i,j] = a * temp + b * C[i,j]
                    end
                end
            end
        end
    end
    
    return C
end
