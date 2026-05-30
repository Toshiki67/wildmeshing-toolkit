#pragma once

// ============================================================
//  Internal shared header for EigenEdgeCollapse implementation files.
//  Contains:
//    * CHOLMOD wrapper helpers (CholFactor, cholmod_factorize_spd, ...).
//    * Apple Accelerate sparse helpers (AppleSparseView, AppleSparseCholFactor,
//      apple_factorize_spd_upper, apple_solve_vec/mat, dvec).
//    * Struct definitions for WoodburyBase / KrrPrecomp / KnewPrecomp.
//
//  Included by: EigenEdgeCollapse.cpp, CostFunctions.cpp.
//  NOT included by the public header (keeps <cholmod.h> / <Accelerate.h> out of
//  any client-facing translation unit).
// ============================================================

#include <cholmod.h>
#include <Accelerate/Accelerate.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <vector>

namespace app::remesh {

// ============================================================
//  Per-section nanosecond accumulators for the Apple Accelerate
//  Woodbury cost path. Shared across the cost / simplify files
//  (defined as inline so all TUs see the same atomic).
//  Reset at the end of each simplify() step after the stats are printed.
// ============================================================
inline std::atomic<long long> g_wba_section_ns[8] = {};
// Section 8 sub-breakdown.
inline std::atomic<long long> g_wba_s8_sub_ns[3] = {};
// apply_knew_inv_accel internal breakdown (9 buckets).
inline std::atomic<long long> g_wba_inv_ns[9] = {};
inline std::atomic<long long> g_wba_inv_count[9] = {};
inline std::atomic<long long> g_wba_inv_calls         = {0};
inline std::atomic<long long> g_wba_inv_refine_solves = {0};

// RAII helper: switches the "current section" — accumulates time spent in the
// previous section into its counter, then starts timing the next one.
struct WbaSectionTimer {
    int current = -1;
    std::chrono::high_resolution_clock::time_point t0;
    void next(int idx) {
        auto now = std::chrono::high_resolution_clock::now();
        if (current >= 0) {
            g_wba_section_ns[current].fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count(),
                std::memory_order_relaxed);
        }
        current = idx;
        t0 = now;
    }
    void stop() { next(-1); }
    ~WbaSectionTimer() {
        if (current >= 0) {
            auto now = std::chrono::high_resolution_clock::now();
            g_wba_section_ns[current].fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count(),
                std::memory_order_relaxed);
        }
    }
};

// ============================================================
//  CHOLMOD helpers
// ============================================================

// Thread-local cholmod_common — CHOLMOD's workspace is NOT thread-safe across
// concurrent calls, so each thread gets its own.
inline cholmod_common* chol_common() {
    thread_local cholmod_common c;
    thread_local bool inited = false;
    if (!inited) {
        cholmod_start(&c);
        c.supernodal = CHOLMOD_SUPERNODAL;
        c.final_ll   = true;
        inited = true;
    }
    return &c;
}

// View an Eigen CSC SparseMatrix as a cholmod_sparse (stype=1: upper triangle).
struct CholSparseView {
    cholmod_sparse A;
    CholSparseView(const Eigen::SparseMatrix<double>& M, int stype = 1) {
        assert(M.isCompressed());
        std::memset(&A, 0, sizeof(A));
        A.nrow = (size_t)M.rows();
        A.ncol = (size_t)M.cols();
        A.nzmax = (size_t)M.nonZeros();
        A.p = const_cast<int*>(M.outerIndexPtr());
        A.i = const_cast<int*>(M.innerIndexPtr());
        A.x = const_cast<double*>(M.valuePtr());
        A.stype = stype;
        A.itype = CHOLMOD_INT;
        A.xtype = CHOLMOD_REAL;
        A.dtype = CHOLMOD_DOUBLE;
        A.sorted = 1;
        A.packed = 1;
    }
};

struct CholFactor {
    cholmod_factor* L = nullptr;
    ~CholFactor() { if (L) cholmod_free_factor(&L, chol_common()); }
    CholFactor() = default;
    CholFactor(const CholFactor&) = delete;
    CholFactor& operator=(const CholFactor&) = delete;
    CholFactor(CholFactor&& o) noexcept : L(o.L) { o.L = nullptr; }
    CholFactor& operator=(CholFactor&& o) noexcept {
        if (this != &o) { if (L) cholmod_free_factor(&L, chol_common()); L = o.L; o.L = nullptr; }
        return *this;
    }
};

// Wrap a column-major Eigen dense as cholmod_dense (no copy).
inline cholmod_dense make_dense_view(const Eigen::Ref<const Eigen::MatrixXd>& B) {
    cholmod_dense d;
    std::memset(&d, 0, sizeof(d));
    d.nrow = (size_t)B.rows();
    d.ncol = (size_t)B.cols();
    d.nzmax = (size_t)(B.rows() * B.cols());
    d.d = (size_t)B.rows();
    d.x = const_cast<double*>(B.data());
    d.xtype = CHOLMOD_REAL;
    d.dtype = CHOLMOD_DOUBLE;
    return d;
}

// Factorize sparse SPD matrix via CHOLMOD.
inline CholFactor cholmod_factorize_spd(const Eigen::SparseMatrix<double>& K) {
    cholmod_common* c = chol_common();
    CholSparseView Av(K, 1);
    CholFactor F;
    F.L = cholmod_analyze(&Av.A, c);
    if (!F.L) return F;
    if (!cholmod_factorize(&Av.A, F.L, c)) { F = CholFactor{}; return F; }
    return F;
}

// CHOLMOD multi-RHS solve: X = L^{-T} L^{-1} B. Result copied into Eigen.
inline Eigen::MatrixXd cholmod_solve_mat(
    cholmod_factor* L, const Eigen::Ref<const Eigen::MatrixXd>& B)
{
    cholmod_common* c = chol_common();
    cholmod_dense Bv = make_dense_view(B);
    cholmod_dense* Xd = cholmod_solve(CHOLMOD_A, L, &Bv, c);
    if (!Xd) return Eigen::MatrixXd();
    Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(
        static_cast<const double*>(Xd->x), Xd->nrow, Xd->ncol);
    cholmod_free_dense(&Xd, c);
    return X;
}

inline Eigen::VectorXd cholmod_solve_vec(
    cholmod_factor* L, const Eigen::Ref<const Eigen::VectorXd>& b)
{
    Eigen::MatrixXd X = cholmod_solve_mat(L, b);
    return X.col(0);
}

// ============================================================
//  Apple Accelerate sparse helpers
// ============================================================

// Wraps an Eigen CSC SparseMatrix as Apple's SparseMatrix_Double.
// Owns a long[] copy of outerIndexPtr (Eigen uses int, Apple uses long).
struct AppleSparseView {
    std::vector<long> colStarts;
    SparseMatrix_Double A{};

    AppleSparseView(const Eigen::SparseMatrix<double>& M, bool sym_upper = false) {
        assert(M.isCompressed());
        const int ncol = (int)M.cols();
        colStarts.resize((size_t)ncol + 1);
        const int* ep = M.outerIndexPtr();
        for (int j = 0; j <= ncol; ++j) colStarts[(size_t)j] = (long)ep[j];

        SparseAttributes_t attrs{};
        attrs.transpose = false;
        attrs.triangle  = SparseUpperTriangle;
        attrs.kind      = sym_upper ? SparseSymmetric : SparseOrdinary;

        A.structure.attributes   = attrs;
        A.structure.rowCount     = (int)M.rows();
        A.structure.columnCount  = ncol;
        A.structure.blockSize    = 1;
        A.structure.columnStarts = colStarts.data();
        A.structure.rowIndices   = const_cast<int*>(M.innerIndexPtr());
        A.data = const_cast<double*>(M.valuePtr());
    }
};

inline DenseVector_Double dvec(double* p, int n) {
    DenseVector_Double d; d.count = n; d.data = p; return d;
}

// RAII wrapper for Apple Accelerate's Cholesky factor.
struct AppleSparseCholFactor {
    SparseOpaqueFactorization_Double F{};
    bool valid = false;

    AppleSparseCholFactor() = default;
    ~AppleSparseCholFactor() { if (valid) SparseCleanup(F); }
    AppleSparseCholFactor(const AppleSparseCholFactor&) = delete;
    AppleSparseCholFactor& operator=(const AppleSparseCholFactor&) = delete;
    AppleSparseCholFactor(AppleSparseCholFactor&& o) noexcept
        : F(o.F), valid(o.valid) { o.valid = false; }
    AppleSparseCholFactor& operator=(AppleSparseCholFactor&& o) noexcept {
        if (this != &o) {
            if (valid) SparseCleanup(F);
            F = o.F; valid = o.valid; o.valid = false;
        }
        return *this;
    }
};

// Factorize SPD matrix via Apple Accelerate. Input is the upper triangle
// in Eigen CSC (compressed). The factor is independent of the input afterwards.
inline AppleSparseCholFactor apple_factorize_spd_upper(
    const Eigen::SparseMatrix<double>& K_upper)
{
    AppleSparseCholFactor out;
    AppleSparseView view(K_upper, /*sym_upper=*/true);
    out.F = SparseFactor(SparseFactorizationCholesky, view.A);
    out.valid = (out.F.status == SparseStatusOK);
    if (!out.valid) SparseCleanup(out.F);
    return out;
}

// A x = b (vector). Allocates and returns x.
inline Eigen::VectorXd apple_solve_vec(
    const AppleSparseCholFactor& chol,
    const Eigen::Ref<const Eigen::VectorXd>& b)
{
    Eigen::VectorXd x = b;
    SparseSolve(chol.F, dvec(x.data(), (int)x.size()));
    return x;
}

// A X = B (multi-RHS, column-major). Allocates and returns X.
inline Eigen::MatrixXd apple_solve_mat(
    const AppleSparseCholFactor& chol,
    const Eigen::Ref<const Eigen::MatrixXd>& B)
{
    Eigen::MatrixXd X = B;
    DenseMatrix_Double xb{};
    xb.rowCount     = (int)X.rows();
    xb.columnCount  = (int)X.cols();
    xb.columnStride = (int)X.rows();
    xb.attributes   = SparseAttributes_t{};
    xb.data         = X.data();
    SparseSolve(chol.F, xb);
    return X;
}

// ============================================================
//  Woodbury precomputed state
// ============================================================

// K_base + its Cholesky factorisation, snapshotted from the current TriMesh.
// One instance is held by EigenEdgeCollapse and rebuilt per simplify step.
struct WoodburyBase {
    Eigen::SparseMatrix<double> K;        // full coarse K (ndof × ndof, with BCs)
    Eigen::SparseMatrix<double> K_bc;     // K restricted to free DOFs (nfree × nfree, SPD)
    CholFactor factor;                     // chol(K_bc) via CHOLMOD
    AppleSparseCholFactor apple_factor;    // chol(K_bc) via Apple Accelerate
    Eigen::MatrixXd V;                     // V_curr at snapshot time (n × 2)
    Eigen::MatrixXi F;                     // F_curr at snapshot time (m × 3)
    Eigen::VectorXd M_diag;                // lumped mass diagonal, length ndof
    std::vector<int> free_dofs;            // size nfree: free idx → full dof idx
    std::vector<int> dof_full_to_free;     // size ndof: full dof → free idx (or -1 if fixed)
    int ndof = 0;
    int nfree = 0;
};

// Per-candidate precompute for the block deletion K_rr (delete `dofs_del`
// rows/cols of K_bc).
//   K_rr^{-1} b_r = z[r] - Q * Q_d_inv * z[d],   where z = K_bc^{-1} b̃,
//   b̃ = embed(b_r, 0 at d)
struct KrrPrecomp {
    Eigen::MatrixXd Q;            // n_r × |dofs_del|
    Eigen::MatrixXd Q_d_inv;      // |dofs_del| × |dofs_del|
    std::vector<int> dofs_del;    // free-DOF indices being deleted
    std::vector<int> r_dofs;      // free-DOF indices NOT in dofs_del (size n_r)
    std::vector<int> free_to_r;   // size nfree: free idx → r idx (or -1)
};

// Per-candidate precompute for the rank-s Woodbury update on K_rr.
//   K_new = K_rr + U Σ U^T,   U = selection of s_dofs (n_r × s, 0/1 entries)
// We use the Σ-inverse-free form:
//   (A + U Σ U^T)^{-1} b = α - W Σ (I_s + U^T W Σ)^{-1} U^T α,    α = A^{-1} b
struct KnewPrecomp {
    Eigen::MatrixXd W;            // n_r × s  (= K_rr^{-1} U)
    Eigen::MatrixXd WSigma;       // n_r × s  (= W * Σ)
    Eigen::MatrixXd H_inv;        // s × s    (= (I_s + U^T W Σ)^{-1})
    Eigen::MatrixXd Sigma;        // s × s    (cached for iterative refinement)
    std::vector<int> s_in_r;      // size s: each s-dof's row index in r-ordering
};

} // namespace app::remesh
