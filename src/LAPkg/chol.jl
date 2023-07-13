"""
    chol!(
        uplo::Char, n::Integer,
        A::AbstractMatrix{Float64},
        lda::Integer, info::Integer 
    )

`chol!` computes the Cholesky factorization of a real symmetric positive definite matrix `A` using the recursive algorithm.

The factorization has the form
- `A = Uᵀ⋅U` if `uplo = 'U'`, or
- `A = L⋅Lᵀ` if `uplo = 'L'`,

where `U` is an upper triangular matrix and `L` is a lower triangular.

This is the recursive version of the algorithm. It divides the matrix into four submatrices:

        ┌─────┬─────┐
        │ A11 │ A12 │
    A = ├─────┼─────┤
        │ A21 │ A22 │
        └─────┴─────┘

, where `A11` is `n1 × n1` and `A22` is `n2 × n2` with `n1 = n ÷ 2` and `n2 = n-n1`.

The subroutine calls itself to factorize `A11`. Update and scale `A21` or `A12`, update `A22`, then call itself to factorize `A22`.

- `uplo`
    - `uplo = 'U'`: Upper triangle of `A` is stored;
    - `uplo = 'L'`: Lower triangle of `A` is stored.

- `n`: The order of the matrix `A`. `N ≥ 0`.

- `A`, `size(A) = (lda, n)`
    - On entry, the symmetric matrix `A`. If `uplo = 'U'`, the leading `n × n` upper triangular part of `A` contains the upper triangular part of the matrix `A`, and the strictly lower triangular part of `A` is not referenced.
    - If `uplo = 'L'`, the leading `n × n` lower triangular part of `A` contains the lower triangular part of the matrix `A`, and the strictly upper triangular part of `A` is not referenced.
    - On exit, if `info = 0`, the factor `U` or `L` from the Cholesky factorization `A = Uᵀ⋅U` or `A = L⋅L`.

- `lda`: The leading dimension of the array `A`. `lda ≥ max(1, n)`.

- `info`
    - `info = 0`: successful exit
    - `info < 0`: if `info = -i`, the i-th argument had an illegal value
    - `info > 0`: if `info = +i`, the leading minor of order `i` is not positive definite, and the factorization could not be completed.
"""
function chol!(
        uplo::Char, n::Integer,
        A::AbstractMatrix{Float64},
        lda::Integer, info::Integer 
    )
    info = 0
    upper = uplo ≡ 'U'
    if !upper && !(uplo ≡ 'L')
        info = -1
    elseif( n < 0 )
        info = -2
    elseif lda < max(1, n)
        info = -4
    end
    !iszero(info) && error("chol! ", -info)
    
    # Quick return if possible
    iszero(n) && return 0 # info
    
    # n = 1 case
    if isone(n)
        # Test for non-positive-definiteness
        (@inbounds A[1,1] ≤ 0.0 || isnan(A[1,1])) && return 1 # info

        # Factor
        @inbounds A[1,1] = sqrt(A[1,1])
        
    # Use recursive code
    else
        n1 = n >> 1
        n2 = n - n1
        
        # Factor A11
        iinfo = chol!(uplo, n1, view(A, 1:n1, 1:n1), lda, info)
        !iszero(iinfo) && return iinfo
        
        # Compute the Cholesky factorization A = Uᵀ⋅U
        if upper
            # Update and scale A12
            trsm!(
                'L', 'U', 'T', 'N',
                n1, n2, 1.0,
                view(A, 1:n1, 1:n1), view(A, 1:n1, n1+1:n),
                lda, lda
            )
            
            # Update and factor A22
            syrk!(
                uplo, 'T',
                n2, n1, -1.0,
                view(A, 1:n1, n1+1:n), lda,
                1.0, view(A, n1+1:n, n1+1:n), lda
            )
            iinfo = chol!(uplo, n2, view(A, n1+1:n, n1+1:n), lda, info)
            !iszero(iinfo) && return iinfo + n1
        
        # Compute the Cholesky factorization A = L⋅Lᵀ
        else
            # Update and scale A21
            trsm!(
                'R', 'L', 'T', 'N',
                n2, n1, 1.0,
                view(A, 1:n1, 1:n1), view(A, n1+1:n, 1:n1),
                lda, lda
            )
            
            # Update and factor A22
            syrk!(
                uplo, 'N',
                n2, n1, -1.0,
                view(A, n1+1:n, 1:n1), lda,
                1.0, view(A, n1+1:n, n1+1:n), lda
            )
            iinfo = chol!(uplo, n2, view(A, n1+1:n, n1+1:n), lda, info)
            !iszero(iinfo) && return iinfo + n1
        end
    end
    return info
end
