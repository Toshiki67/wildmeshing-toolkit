#include "EigenEdgeCollapse.hpp"
#include "VisUtils.hpp"

#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// SuiteSparse CHOLMOD (direct C API for max performance)
#include <cholmod.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <random>

#include <omp.h>

#include <Accelerate/Accelerate.h>

namespace app::remesh {

// ============================================================
//  CHOLMOD helpers (file-local) — sparse SPD factor/solve only.
//  Dense systems (e.g. Woodbury's small Σ inverse) use Eigen.
// ============================================================
namespace {

// ============================================================
//  Per-section nanosecond accumulators for candidate_cost_woodbury_accelerated.
//  Summed across all edges (and threads) within one simplify() step; reset at
//  the end of each step after printing.
// ============================================================
std::atomic<long long> g_wba_section_ns[8] = {};
// Section 8 sub-breakdown: [0]=apply_knew_inv_accel (solve),
// [1]=SparseMultiply calls, [2]=everything else (vDSP, permutations, ddot).
std::atomic<long long> g_wba_s8_sub_ns[3] = {};
// apply_knew_inv_accel / apply_krr_inv_accel internal breakdown:
//   [0] = apple_solve_vec                           (n×n CHOLMOD/Apple solve)
//   [1] = apply_krr_inv_accel: gemv Q_d_inv (nd×nd)
//   [2] = apply_krr_inv_accel: gemv Q     (n_r×nd, block-deletion correction)
//   [3] = apply_krr_inv_accel: embed b_r + extract z_d/z_r
//   [4] = apply_knew_inv_once_accel: gemv H_inv  (s×s, Woodbury correction)
//   [5] = apply_knew_inv_once_accel: gemv WΣ     (n_r×s, Woodbury correction)
//   [6] = apply_knew_mat_accel: SparseMultiply
//   [7] = apply_knew_mat_accel: gemv Σ           (s×s, rank-s update on r-block)
//   [8] = apply_knew_inv_accel: refinement loop overhead (vsubD/dnrm2/daxpy)
std::atomic<long long> g_wba_inv_ns[9] = {};
// Call counts parallel to g_wba_inv_ns (number of times each sub-op fired).
std::atomic<long long> g_wba_inv_count[9] = {};
std::atomic<long long> g_wba_inv_calls         = {0};
std::atomic<long long> g_wba_inv_refine_solves = {0};

// RAII helper: switches the "current section" — accumulates time spent in the
// previous section into its counter, then starts timing the next one. The
// destructor also flushes the in-progress section, so early returns still
// get accounted for.
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
    // Flush the currently active section and disarm; safe to call multiple times.
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

// Process-wide cholmod_common (single-threaded sequential use).
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

// ----------------------------------------------------------------
// Apple Accelerate Sparse BLAS helpers (used by *_accelerated paths)
// ----------------------------------------------------------------

// Wraps an Eigen CSC SparseMatrix as Apple's SparseMatrix_Double.
// Owns a long[] copy of outerIndexPtr (Eigen uses int, Apple uses long).
// Row indices and values are referenced in-place (no copy).
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

} // anonymous namespace

// ============================================================
//  Woodbury / CHOLMOD precomputed state (definitions)
// ============================================================

// K_base + its CHOLMOD factorisation, snapshotted from the current TriMesh.
// One instance is held by EigenEdgeCollapse and rebuilt per simplify step.
struct WoodburyBase {
    Eigen::SparseMatrix<double> K;        // full coarse K (ndof × ndof, with spring BCs / kinetic shift)
    Eigen::SparseMatrix<double> K_bc;     // K restricted to free DOFs (nfree × nfree, SPD)
    CholFactor factor;                     // chol(K_bc) via CHOLMOD (used by non-accelerated path)
    AppleSparseCholFactor apple_factor;    // chol(K_bc) via Apple Accelerate (used by *_accelerated path)
    Eigen::MatrixXd V;                     // V_curr at snapshot time (n × 2)
    Eigen::MatrixXi F;                     // F_curr at snapshot time (m × 3)
    Eigen::VectorXd M_diag;                // lumped mass diagonal, length ndof
    std::vector<int> free_dofs;            // size nfree: free idx → full dof idx
    std::vector<int> dof_full_to_free;     // size ndof: full dof → free idx (or -1 if fixed)
    int ndof = 0;
    int nfree = 0;
};

// Per-candidate precompute for the block deletion K_rr (delete `dofs_del` rows/cols of K_bc).
//   K_rr^{-1} b_r = z[r] - Q * Q_d_inv * z[d],   where z = K_bc^{-1} b̃, b̃ = embed(b_r, 0 at d)
struct KrrPrecomp {
    Eigen::MatrixXd Q;            // n_r × |dofs_del|, rows of K_bc^{-1}[:, dofs_del] at r-dofs
    Eigen::MatrixXd Q_d_inv;      // |dofs_del| × |dofs_del|, inverse of K_bc^{-1}[d, d]
    std::vector<int> dofs_del;    // free-DOF indices being deleted (typically 2 for one vertex)
    std::vector<int> r_dofs;      // free-DOF indices NOT in dofs_del (size n_r)
    std::vector<int> free_to_r;   // size nfree: free idx → r idx (or -1 if deleted)
};

// Per-candidate precompute for the rank-s Woodbury update on K_rr.
//   K_new = K_rr + U Σ U^T,   U = selection of s_dofs (n_r × s, 0/1 entries)
// We use the Σ-inverse-free form of Woodbury:
//   (A + U Σ U^T)^{-1} b = α - W Σ (I_s + U^T W Σ)^{-1} U^T α,    α = A^{-1} b
// This is essential because Σ = K_new[s,s] - K_old[s,s] is typically
// rank-deficient (each 2D element Ke has 3 rigid-body modes, so Σ inherits
// nullspace contributions). Σ^{-1} is therefore unsafe; H = I + (U^T W) Σ is
// well-conditioned as long as K_new itself is invertible.
//
// The single-Woodbury-step solve loses ~6-10 digits (block-deletion cancellation
// in apply_krr_inv → amplification through W H^{-1}), so we always run 2 steps
// of iterative refinement against the sparse K_new matvec — that recovers
// CHOLMOD-level accuracy (~1e-12).
struct KnewPrecomp {
    Eigen::MatrixXd W;            // n_r × s  (= K_rr^{-1} U, via apply_krr_inv_mat)
    Eigen::MatrixXd WSigma;       // n_r × s  (= W * Σ, cached for the apply step)
    Eigen::MatrixXd H_inv;        // s × s    (= (I_s + U^T W Σ)^{-1}, dense via Eigen)
    Eigen::MatrixXd Sigma;        // s × s    (cached for iterative-refinement matvec)
    std::vector<int> s_in_r;      // size s: each s-dof's row index in the r-dof ordering
};

EigenEdgeCollapse::EigenEdgeCollapse() {
    p_vertex_attrs = &vertex_attrs;
}

EigenEdgeCollapse::~EigenEdgeCollapse() = default;

// ---------------------------------------------------------- helper free functions
// File-scope (not in anonymous namespace) so they can be referenced from
// EigenEdgeCollapse member functions defined later in this TU.


// ============================================================
//  Apple Accelerate versions of apply_krr_inv / apply_knew_*
// ============================================================
// Same math as the Eigen-based versions above, with all matrix operations
// (dense matvec via cblas_dgemv, K_bc sparse matvec via SparseMultiply,
// and the K_bc^{-1} solve via Apple's SparseSolve on wb.apple_factor)
// routed through Apple Accelerate.

static Eigen::VectorXd apply_krr_inv_accel(
    const Eigen::Ref<const Eigen::VectorXd>& b_r,
    const WoodburyBase& wb, const KrrPrecomp& kp)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ns_since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    const int n  = wb.nfree;
    const int nr = (int)kp.r_dofs.size();
    const int nd = (int)kp.dofs_del.size();

    auto t0 = Clock::now();
    Eigen::VectorXd b_full = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < nr; ++i) b_full[kp.r_dofs[i]] = b_r[i];
    auto t1 = Clock::now();

    Eigen::VectorXd z = apple_solve_vec(wb.apple_factor, b_full);
    auto t2 = Clock::now();

    Eigen::VectorXd z_d(nd), z_r(nr);
    for (int i = 0; i < nd; ++i) z_d[i] = z[kp.dofs_del[i]];
    for (int i = 0; i < nr; ++i) z_r[i] = z[kp.r_dofs[i]];
    auto t3 = Clock::now();

    // wd = Q_d_inv * z_d
    Eigen::VectorXd wd(nd);
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nd, nd, 1.0,
                kp.Q_d_inv.data(), nd,
                z_d.data(), 1,
                0.0, wd.data(), 1);
    auto t4 = Clock::now();

    // z_r -= Q * wd
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nr, nd, -1.0,
                kp.Q.data(), nr,
                wd.data(), 1,
                1.0, z_r.data(), 1);
    auto t5 = Clock::now();

    g_wba_inv_ns[0].fetch_add(ns_since(t1, t2), std::memory_order_relaxed); // apple_solve_vec
    g_wba_inv_ns[1].fetch_add(ns_since(t3, t4), std::memory_order_relaxed); // Q_d_inv gemv
    g_wba_inv_ns[2].fetch_add(ns_since(t4, t5), std::memory_order_relaxed); // Q gemv
    g_wba_inv_ns[3].fetch_add(ns_since(t0, t1) + ns_since(t2, t3),
                              std::memory_order_relaxed); // embed + extract
    g_wba_inv_count[0].fetch_add(1, std::memory_order_relaxed);
    g_wba_inv_count[1].fetch_add(1, std::memory_order_relaxed);
    g_wba_inv_count[2].fetch_add(1, std::memory_order_relaxed);
    g_wba_inv_count[3].fetch_add(1, std::memory_order_relaxed);
    return z_r;
}

static Eigen::VectorXd apply_knew_mat_accel(
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const WoodburyBase& wb, const KrrPrecomp& kp, const KnewPrecomp& knew,
    const AppleSparseView& Kbc_view)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ns_since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    const int nr = (int)kp.r_dofs.size();

    Eigen::VectorXd y_full = Eigen::VectorXd::Zero(wb.nfree);
    for (int i = 0; i < nr; ++i) y_full[kp.r_dofs[i]] = y[i];

    auto t_mul0 = Clock::now();
    Eigen::VectorXd Ky_full(wb.nfree);
    SparseMultiply(Kbc_view.A,
                   dvec(y_full.data(), (int)y_full.size()),
                   dvec(Ky_full.data(), (int)Ky_full.size()));
    auto t_mul1 = Clock::now();

    Eigen::VectorXd Ky_r(nr);
    for (int i = 0; i < nr; ++i) Ky_r[i] = Ky_full[kp.r_dofs[i]];

    long long sigma_ns = 0;
    const int s = (int)knew.s_in_r.size();
    if (s > 0) {
        auto t_sig0 = Clock::now();
        Eigen::VectorXd y_s(s);
        for (int k = 0; k < s; ++k) y_s[k] = y[knew.s_in_r[k]];
        Eigen::VectorXd Sigma_ys(s);
        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    s, s, 1.0,
                    knew.Sigma.data(), s,
                    y_s.data(), 1,
                    0.0, Sigma_ys.data(), 1);
        for (int k = 0; k < s; ++k) Ky_r[knew.s_in_r[k]] += Sigma_ys[k];
        auto t_sig1 = Clock::now();
        sigma_ns = ns_since(t_sig0, t_sig1);
    }

    g_wba_inv_ns[6].fetch_add(ns_since(t_mul0, t_mul1), std::memory_order_relaxed); // SparseMultiply
    g_wba_inv_ns[7].fetch_add(sigma_ns, std::memory_order_relaxed);                 // Σ gemv
    g_wba_inv_count[6].fetch_add(1, std::memory_order_relaxed);
    if (s > 0) g_wba_inv_count[7].fetch_add(1, std::memory_order_relaxed);
    return Ky_r;
}

static Eigen::VectorXd apply_knew_inv_once_accel(
    const Eigen::Ref<const Eigen::VectorXd>& b_r,
    const WoodburyBase& wb, const KrrPrecomp& kp, const KnewPrecomp& knew)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ns_since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    Eigen::VectorXd alpha = apply_krr_inv_accel(b_r, wb, kp);  // self-timed

    const int s = (int)knew.s_in_r.size();
    if (s == 0) return alpha;

    Eigen::VectorXd Ut_alpha(s);
    for (int k = 0; k < s; ++k) Ut_alpha[k] = alpha[knew.s_in_r[k]];

    auto t_h0 = Clock::now();
    Eigen::VectorXd H_inv_Ut(s);
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                s, s, 1.0,
                knew.H_inv.data(), s,
                Ut_alpha.data(), 1,
                0.0, H_inv_Ut.data(), 1);
    auto t_h1 = Clock::now();

    const int nr = (int)alpha.size();
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nr, s, -1.0,
                knew.WSigma.data(), nr,
                H_inv_Ut.data(), 1,
                1.0, alpha.data(), 1);
    auto t_w1 = Clock::now();

    g_wba_inv_ns[4].fetch_add(ns_since(t_h0, t_h1), std::memory_order_relaxed); // H_inv gemv
    g_wba_inv_ns[5].fetch_add(ns_since(t_h1, t_w1), std::memory_order_relaxed); // WΣ gemv
    g_wba_inv_count[4].fetch_add(1, std::memory_order_relaxed);
    g_wba_inv_count[5].fetch_add(1, std::memory_order_relaxed);
    return alpha;
}

static Eigen::VectorXd apply_knew_inv_accel(
    const Eigen::Ref<const Eigen::VectorXd>& b_r,
    const WoodburyBase& wb, const KrrPrecomp& kp, const KnewPrecomp& knew,
    const AppleSparseView& Kbc_view)
{
    using Clock = std::chrono::high_resolution_clock;
    auto ns_since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    g_wba_inv_calls.fetch_add(1, std::memory_order_relaxed);

    Eigen::VectorXd y = apply_knew_inv_once_accel(b_r, wb, kp, knew);  // self-timed
    const int n = (int)y.size();
    const double bn = std::max(cblas_dnrm2(n, b_r.data(), 1), 1e-30);

    long long overhead_ns = 0;
    Eigen::VectorXd r_vec(n);
    // Iterative refinement count: when Σ has huge entries (e.g., from
    // spring_k = 10^15 contributions), the Woodbury solver loses ~ε×cond
    // = 10^-1 relative accuracy per call. With 2 iterations the answer
    // is only ~10^-3 accurate. Increase to 10 to get ~10^-11 accuracy.
    for (int it = 0; it < 10; ++it) {
        // r_vec = K_new * y, then r_vec = b_r - r_vec
        Eigen::VectorXd Ky = apply_knew_mat_accel(y, wb, kp, knew, Kbc_view);

        auto t_ov0 = Clock::now();
        vDSP_vsubD(Ky.data(), 1, b_r.data(), 1, r_vec.data(), 1, (vDSP_Length)n);
        const double rn = cblas_dnrm2(n, r_vec.data(), 1);
        auto t_ov1 = Clock::now();
        overhead_ns += ns_since(t_ov0, t_ov1);

        if (rn / bn < 1e-12) break;

        g_wba_inv_refine_solves.fetch_add(1, std::memory_order_relaxed);

        Eigen::VectorXd dy = apply_knew_inv_once_accel(r_vec, wb, kp, knew);

        auto t_ax0 = Clock::now();
        cblas_daxpy(n, 1.0, dy.data(), 1, y.data(), 1);
        auto t_ax1 = Clock::now();
        overhead_ns += ns_since(t_ax0, t_ax1);
    }
    g_wba_inv_ns[8].fetch_add(overhead_ns, std::memory_order_relaxed);
    g_wba_inv_count[8].fetch_add(1, std::memory_order_relaxed);
    return y;
}

// Scalar barycentric coordinates of `q` w.r.t. triangle (p0,p1,p2). Sum to 1.
// Stable for non-degenerate triangles.
static Eigen::Vector3d barycentric_2d(
    const Eigen::Vector2d& q,
    const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
{
    const double det = (p1[0] - p0[0]) * (p2[1] - p0[1])
                     - (p2[0] - p0[0]) * (p1[1] - p0[1]);
    const double inv = 1.0 / det;
    Eigen::Vector3d b;
    b[1] = ((p2[1] - p0[1]) * (q[0] - p0[0]) - (p2[0] - p0[0]) * (q[1] - p0[1])) * inv;
    b[2] = ((p0[1] - p1[1]) * (q[0] - p0[0]) - (p0[0] - p1[0]) * (q[1] - p0[1])) * inv;
    b[0] = 1.0 - b[1] - b[2];
    return b;
}

// ============================================================
//  Polygon clipping helpers (file-local)
// ============================================================

using Poly2d = std::vector<Eigen::Vector2d>;

// Sutherland-Hodgman: clip polygon against the half-plane { a*x + b*y + c >= 0 }
static Poly2d clip_half_plane(const Poly2d& poly, double a, double b, double c)
{
    if (poly.size() < 3) return {};
    Poly2d out;
    out.reserve(poly.size() + 1);
    const int n = (int)poly.size();
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d& cur = poly[i];
        const Eigen::Vector2d& nxt = poly[(i + 1) % n];
        const double dc = a * cur[0] + b * cur[1] + c;
        const double dn = a * nxt[0] + b * nxt[1] + c;
        if (dc >= 0.0) out.push_back(cur);
        if ((dc > 0.0) != (dn > 0.0)) {
            const double t = dc / (dc - dn);
            out.push_back(cur + t * (nxt - cur));
        }
    }
    return out;
}

// Area of a polygon (shoelace, absolute value)
static double polygon_area(const Poly2d& poly)
{
    double s = 0.0;
    const int n = (int)poly.size();
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d& a = poly[i];
        const Eigen::Vector2d& b = poly[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    return 0.5 * std::abs(s);
}

// Clip triangle B against triangle A (A as 3 half-planes), return overlap area.
static double triangle_overlap_area(
    const Eigen::Vector2d& a0, const Eigen::Vector2d& a1, const Eigen::Vector2d& a2,
    const Eigen::Vector2d& b0, const Eigen::Vector2d& b1, const Eigen::Vector2d& b2)
{
    // Half-planes of triangle A (inward normals, CCW winding assumed)
    Poly2d poly = {b0, b1, b2};
    auto edge_halfplane = [&](const Eigen::Vector2d& p, const Eigen::Vector2d& q) {
        // Normal pointing inward: n = (-(q-p).y, (q-p).x)
        Eigen::Vector2d d = q - p;
        double a = -d[1], b = d[0];
        double c = -(a * p[0] + b * p[1]);
        return std::make_tuple(a, b, c);
    };
    for (auto& [ea, eb, ec] : {edge_halfplane(a0, a1),
                                edge_halfplane(a1, a2),
                                edge_halfplane(a2, a0)}) {
        poly = clip_half_plane(poly, ea, eb, ec);
        if (poly.size() < 3) return 0.0;
    }
    return polygon_area(poly);
}

// ============================================================
//  Elasticity material tensor (plane stress or plane strain)
// ============================================================

Eigen::Matrix3d EigenEdgeCollapse::elasticity_D(double E, double nu) const
{
    if (m_p.plane_stress) {
        double c = E / (1.0 - nu * nu);
        Eigen::Matrix3d D;
        D << c,      c * nu, 0,
             c * nu, c,      0,
             0,      0,      c * (1.0 - nu) / 2.0;
        return D;
    } else {
        double c = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Eigen::Matrix3d D;
        D << c * (1.0 - nu), c * nu,          0,
             c * nu,          c * (1.0 - nu), 0,
             0,               0,              c * (1.0 - 2.0 * nu) / 2.0;
        return D;
    }
}

// ============================================================
//  Material region test
// ============================================================

// Returns true when (x, y) is inside the "left" material region.
// If a material OBJ was loaded, the test is point-in-triangle for any triangle
// in that mesh (or ray-casting if the OBJ has no faces).
// Otherwise falls back to the hardcoded analytical boundary.
bool EigenEdgeCollapse::point_in_material_left(double x, double y) const
{
    if (m_material_verts.rows() == 0) {
        // Hardcoded fallback: right half-plane is the "right" material.
        return !(x >= 0.0);
    }

    if (m_material_faces.rows() > 0) {
        // Point-in-triangle for every face of the material mesh.
        for (int f = 0; f < (int)m_material_faces.rows(); ++f) {
            const Eigen::Vector2d a = m_material_verts.row(m_material_faces(f, 0));
            const Eigen::Vector2d b = m_material_verts.row(m_material_faces(f, 1));
            const Eigen::Vector2d c = m_material_verts.row(m_material_faces(f, 2));
            const double d0 = (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0]);
            const double d1 = (c[0]-b[0])*(y-b[1]) - (c[1]-b[1])*(x-b[0]);
            const double d2 = (a[0]-c[0])*(y-c[1]) - (a[1]-c[1])*(x-c[0]);
            const bool has_neg = (d0 < 0) || (d1 < 0) || (d2 < 0);
            const bool has_pos = (d0 > 0) || (d1 > 0) || (d2 > 0);
            if (!(has_neg && has_pos)) return true;
        }
        return false;
    } else {
        // Treat the OBJ vertices as a polygon; ray-casting test.
        const int n = (int)m_material_verts.rows();
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            const double xi = m_material_verts(i, 0), yi = m_material_verts(i, 1);
            const double xj = m_material_verts(j, 0), yj = m_material_verts(j, 1);
            if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }
}

// ============================================================
//  Material E assignment
// ============================================================

// Centroid sampling — used only for fine mesh initialisation.
double EigenEdgeCollapse::material_E_analytical(
    const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const
{
    const Eigen::Vector2d centroid =
        (V.row(tri[0]) + V.row(tri[1]) + V.row(tri[2])) / 3.0;
    if (m_p.is_gradient) {
        const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
        const double t = (xmax > xmin) ? (centroid[0] - xmin) / (xmax - xmin) : 0.0;
        const double tc = std::max(0.0, std::min(1.0, t));
        return m_p.E_left * (1.0 - tc) + m_p.E_right * tc;
    }
    return point_in_material_left(centroid[0], centroid[1]) ? m_p.E_left : m_p.E_right;
}

// ============================================================
//  Material nu assignment
// ============================================================

double EigenEdgeCollapse::material_nu_analytical(
    const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const
{
    const Eigen::Vector2d centroid =
        (V.row(tri[0]) + V.row(tri[1]) + V.row(tri[2])) / 3.0;
    if (m_p.is_gradient) {
        const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
        const double t = (xmax > xmin) ? (centroid[0] - xmin) / (xmax - xmin) : 0.0;
        const double tc = std::max(0.0, std::min(1.0, t));
        return m_p.get_nu_left() * (1.0 - tc) + m_p.get_nu_right() * tc;
    }
    return point_in_material_left(centroid[0], centroid[1])
               ? m_p.get_nu_left() : m_p.get_nu_right();
}

// Area-weighted projection of per-element nu from the fine mesh.
double EigenEdgeCollapse::material_nu_from_fine(
    const Eigen::MatrixXd& V_coarse, const Eigen::Vector3i& tri_c) const
{
    const Eigen::Vector2d ca = V_coarse.row(tri_c[0]);
    const Eigen::Vector2d cb = V_coarse.row(tri_c[1]);
    const Eigen::Vector2d cc = V_coarse.row(tri_c[2]);

    const double cxmin = std::min({ca[0], cb[0], cc[0]});
    const double cxmax = std::max({ca[0], cb[0], cc[0]});
    const double cymin = std::min({ca[1], cb[1], cc[1]});
    const double cymax = std::max({ca[1], cb[1], cc[1]});

    const double area_c =
        0.5 * std::abs((cb[0]-ca[0])*(cc[1]-ca[1]) - (cc[0]-ca[0])*(cb[1]-ca[1]));
    if (area_c < 1e-14) return m_p.get_nu_left();

    double weighted_nu   = 0.0;
    double total_overlap = 0.0;

    for (int f = 0; f < (int)m_F_fine.rows(); ++f) {
        if (m_fine_bb_max(f, 0) < cxmin || m_fine_bb_min(f, 0) > cxmax ||
            m_fine_bb_max(f, 1) < cymin || m_fine_bb_min(f, 1) > cymax)
            continue;

        const Eigen::Vector2d fa = m_V_fine.row(m_F_fine(f, 0));
        const Eigen::Vector2d fb = m_V_fine.row(m_F_fine(f, 1));
        const Eigen::Vector2d fc = m_V_fine.row(m_F_fine(f, 2));

        const double ov = triangle_overlap_area(ca, cb, cc, fa, fb, fc);
        if (ov < 1e-15) continue;

        weighted_nu   += ov * m_fine_nu[f];
        total_overlap += ov;
    }

    if (total_overlap < 1e-14) return m_p.get_nu_left();
    return weighted_nu / total_overlap;
}

double EigenEdgeCollapse::material_nu(
    const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const
{
    if (m_fine_nu.size() > 0)
        return material_nu_from_fine(V, tri);
    return material_nu_analytical(V, tri);
}

// Exact area-weighted projection from fine mesh E values via triangle clipping.
// Each coarse triangle's E = sum_f (overlap_area(T_coarse, T_fine_f) * E_fine_f)
//                          / sum_f  overlap_area(T_coarse, T_fine_f)
double EigenEdgeCollapse::material_E_from_fine(
    const Eigen::MatrixXd& V_coarse, const Eigen::Vector3i& tri_c) const
{
    const Eigen::Vector2d ca = V_coarse.row(tri_c[0]);
    const Eigen::Vector2d cb = V_coarse.row(tri_c[1]);
    const Eigen::Vector2d cc = V_coarse.row(tri_c[2]);

    // Bounding box of coarse triangle
    const double cxmin = std::min({ca[0], cb[0], cc[0]});
    const double cxmax = std::max({ca[0], cb[0], cc[0]});
    const double cymin = std::min({ca[1], cb[1], cc[1]});
    const double cymax = std::max({ca[1], cb[1], cc[1]});

    const double area_c =
        0.5 * std::abs((cb[0]-ca[0])*(cc[1]-ca[1]) - (cc[0]-ca[0])*(cb[1]-ca[1]));
    if (area_c < 1e-14) return m_p.E_left;

    double weighted_E  = 0.0;
    double total_overlap = 0.0;

    for (int f = 0; f < (int)m_F_fine.rows(); ++f) {
        // Bounding-box prefilter
        if (m_fine_bb_max(f, 0) < cxmin || m_fine_bb_min(f, 0) > cxmax ||
            m_fine_bb_max(f, 1) < cymin || m_fine_bb_min(f, 1) > cymax)
            continue;

        const Eigen::Vector2d fa = m_V_fine.row(m_F_fine(f, 0));
        const Eigen::Vector2d fb = m_V_fine.row(m_F_fine(f, 1));
        const Eigen::Vector2d fc = m_V_fine.row(m_F_fine(f, 2));

        const double ov = triangle_overlap_area(ca, cb, cc, fa, fb, fc);
        if (ov < 1e-15) continue;

        weighted_E   += ov * m_fine_E[f];
        total_overlap += ov;
    }

    if (total_overlap < 1e-14) return m_p.E_left;
    return weighted_E / total_overlap;
}

// Dispatch: use fine-mesh projection when m_fine_E is populated (after init),
// otherwise fall back to analytical (during fine mesh FEM assembly itself).
double EigenEdgeCollapse::material_E(
    const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const
{
    if (m_fine_E.size() > 0)
        return material_E_from_fine(V, tri);
    return material_E_analytical(V, tri);
}

// ============================================================
//  FEM assembly: K (stiffness) and M_diag (lumped mass)
// ============================================================

void EigenEdgeCollapse::assemble_fem(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& K,
    Eigen::VectorXd& M_diag) const
{
    const int n    = (int)V.rows();
    const int ndof = 2 * n;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(F.rows() * 36);
    M_diag = Eigen::VectorXd::Zero(ndof);

    for (int e = 0; e < (int)F.rows(); ++e) {
        Eigen::Vector3i tri = F.row(e);
        const double x1 = V(tri[0], 0), y1 = V(tri[0], 1);
        const double x2 = V(tri[1], 0), y2 = V(tri[1], 1);
        const double x3 = V(tri[2], 0), y3 = V(tri[2], 1);

        const double signed_area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        const double area = 0.5 * std::abs(signed_area2);
        if (area <= 1e-14) continue;

        // Shape function gradients
        Eigen::Vector3d dNdx, dNdy;
        dNdx << (y2 - y3), (y3 - y1), (y1 - y2);
        dNdy << (x3 - x2), (x1 - x3), (x2 - x1);
        dNdx /= signed_area2;
        dNdy /= signed_area2;

        // B matrix (3 × 6)
        Eigen::Matrix<double, 3, 6> B = Eigen::Matrix<double, 3, 6>::Zero();
        for (int a = 0; a < 3; ++a) {
            B(0, 2 * a)     = dNdx[a];
            B(1, 2 * a + 1) = dNdy[a];
            B(2, 2 * a)     = dNdy[a];
            B(2, 2 * a + 1) = dNdx[a];
        }

        const double E   = material_E(V, tri);
        const double nu  = material_nu(V, tri);
        const Eigen::Matrix<double, 6, 6> Ke = area * (B.transpose() * elasticity_D(E, nu) * B);

        // Local DOF indices
        std::array<int, 6> dofs = {
            2 * tri[0], 2 * tri[0] + 1,
            2 * tri[1], 2 * tri[1] + 1,
            2 * tri[2], 2 * tri[2] + 1,
        };
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                triplets.emplace_back(dofs[a], dofs[b], Ke(a, b));

        // Lumped mass
        for (int vi : {tri[0], tri[1], tri[2]}) {
            M_diag[2 * vi]     += m_p.rho * area / 3.0;
            M_diag[2 * vi + 1] += m_p.rho * area / 3.0;
        }
    }

    K.resize(ndof, ndof);
    K.setFromTriplets(triplets.begin(), triplets.end());
    // Symmetrize
    K = 0.5 * (K + Eigen::SparseMatrix<double>(K.transpose()));
}

// ============================================================
//  Boundary utilities
// ============================================================

std::vector<int> EigenEdgeCollapse::left_boundary_verts(
    const Eigen::MatrixXd& V) const
{
    const double xmin = V.col(0).minCoeff();
    const double span = std::max(V.col(0).maxCoeff() - xmin, 1.0);
    const double eps  = m_p.boundary_tol * span;

    std::vector<int> verts;
    for (int i = 0; i < (int)V.rows(); ++i)
        if (std::abs(V(i, 0) - xmin) <= eps)
            verts.push_back(i);
    return verts;
}

std::vector<int> EigenEdgeCollapse::right_boundary_verts(
    const Eigen::MatrixXd& V) const
{
    const double xmax = V.col(0).maxCoeff();
    const double span = std::max(xmax - V.col(0).minCoeff(), 1.0);
    const double eps  = m_p.boundary_tol * span;

    std::vector<int> verts;
    for (int i = 0; i < (int)V.rows(); ++i)
        if (std::abs(V(i, 0) - xmax) <= eps)
            verts.push_back(i);
    return verts;
}

void EigenEdgeCollapse::apply_spring_bcs(
    Eigen::SparseMatrix<double>& K,
    const Eigen::MatrixXd& V_coarse,
    const Eigen::MatrixXi& F_coarse) const
{
    if (m_p.spring_k <= 0.0) return;

    // ── Legacy path: no user-specified fine spring verts ──────────────────
    if (m_spring_fine_verts.empty()) {
        for (int v : left_boundary_verts(V_coarse)) {
            K.coeffRef(2 * v,     2 * v)     += m_p.spring_k;
            K.coeffRef(2 * v + 1, 2 * v + 1) += m_p.spring_k;
        }
        for (int v : right_boundary_verts(V_coarse)) {
            K.coeffRef(2 * v,     2 * v)     += m_p.spring_k;
            K.coeffRef(2 * v + 1, 2 * v + 1) += m_p.spring_k;
        }
        return;
    }

    // ── New path: P^T K_spring^fine P contribution ────────────────────────
    // For each user-specified fine vertex v_f, locate its containing coarse
    // triangle in (V_coarse, F_coarse) and add spring_k * bary * bary^T to K
    // on the (x and y) DOF blocks of the three coarse vertices.
    const double eps = 1e-10;
    const int n_t = (int)F_coarse.rows();
    if (n_t == 0) return;

    for (int vf : m_spring_fine_verts) {
        if (vf < 0 || vf >= (int)m_V_fine.rows()) continue;
        const Eigen::Vector2d p = m_V_fine.row(vf);

        std::array<double, 3> bary = {1.0, 0.0, 0.0};
        std::array<int, 3>    verts = {0, 0, 0};
        bool found = false;

        for (int t = 0; t < n_t && !found; ++t) {
            const Eigen::Vector2d a = V_coarse.row(F_coarse(t, 0));
            const Eigen::Vector2d b = V_coarse.row(F_coarse(t, 1));
            const Eigen::Vector2d c = V_coarse.row(F_coarse(t, 2));
            const double den = (b - a)[0] * (c - a)[1] - (c - a)[0] * (b - a)[1];
            if (std::abs(den) < eps) continue;

            const Eigen::Vector2d v2 = p - a;
            const double inv = 1.0 / den;
            const double u = (v2[0] * (c - a)[1] - (c - a)[0] * v2[1]) * inv;
            const double v = ((b - a)[0] * v2[1] - v2[0] * (b - a)[1]) * inv;
            const double w = 1.0 - u - v;
            if (u >= -eps && v >= -eps && w >= -eps &&
                u <= 1.0 + eps && v <= 1.0 + eps && w <= 1.0 + eps)
            {
                bary  = {std::clamp(w, 0.0, 1.0),
                         std::clamp(u, 0.0, 1.0),
                         std::clamp(v, 0.0, 1.0)};
                const double s = bary[0] + bary[1] + bary[2];
                if (s > 0) { bary[0] /= s; bary[1] /= s; bary[2] /= s; }
                verts = {F_coarse(t, 0), F_coarse(t, 1), F_coarse(t, 2)};
                found = true;
            }
        }

        if (!found) {
            // Fallback: nearest coarse vertex (full spring stiffness on it)
            int nearest = 0;
            double best = std::numeric_limits<double>::max();
            for (int vc = 0; vc < (int)V_coarse.rows(); ++vc) {
                const double d = (Eigen::Vector2d(V_coarse.row(vc)) - p).squaredNorm();
                if (d < best) { best = d; nearest = vc; }
            }
            bary  = {1.0, 0.0, 0.0};
            verts = {nearest, nearest, nearest};
        }

        // K += spring_k * bary * bary^T  on each of (x, y) DOF blocks.
        for (int comp = 0; comp < 2; ++comp) {
            for (int i = 0; i < 3; ++i) {
                const double wi = bary[i];
                if (wi == 0.0) continue;
                for (int j = 0; j < 3; ++j) {
                    const double wj = bary[j];
                    if (wj == 0.0) continue;
                    K.coeffRef(2 * verts[i] + comp, 2 * verts[j] + comp)
                        += m_p.spring_k * wi * wj;
                }
            }
        }
    }
}

std::vector<int> EigenEdgeCollapse::free_dof_indices(
    int ndof, const std::vector<int>& fixed_verts) const
{
    std::vector<bool> is_fixed(ndof, false);
    for (int v : fixed_verts) {
        is_fixed[2 * v]     = true;
        is_fixed[2 * v + 1] = true;
    }
    std::vector<int> free;
    free.reserve(ndof);
    for (int i = 0; i < ndof; ++i)
        if (!is_fixed[i]) free.push_back(i);
    return free;
}

EigenEdgeCollapse::BoundaryInfo EigenEdgeCollapse::classify_boundary(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F) const
{
    const int n = (int)V.rows();
    BoundaryInfo info;
    info.on_boundary.resize(n, false);
    info.on_corner.resize(n, false);

    if (!m_p.general_mesh) {
        // Rectangle mode: coordinate-based bounding-box detection (original).
        const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
        const double ymin = V.col(1).minCoeff(), ymax = V.col(1).maxCoeff();
        const double span = std::max({xmax - xmin, ymax - ymin, 1.0});
        const double eps  = m_p.boundary_tol * span;

        for (int i = 0; i < n; ++i) {
            bool onL = std::abs(V(i, 0) - xmin) <= eps;
            bool onR = std::abs(V(i, 0) - xmax) <= eps;
            bool onB = std::abs(V(i, 1) - ymin) <= eps;
            bool onT = std::abs(V(i, 1) - ymax) <= eps;
            info.on_boundary[i] = onL || onR || onB || onT;
            info.on_corner[i]   = (onL || onR) && (onB || onT);
        }
        // Populate boundary_edges topologically so the boundary–boundary
        // non-boundary-edge rejection works in rectangle mode too.
        {
            std::map<std::pair<int,int>, int> efc;
            for (int f = 0; f < (int)F.rows(); ++f)
                for (int e = 0; e < 3; ++e) {
                    int a = F(f, e), b = F(f, (e + 1) % 3);
                    efc[{std::min(a, b), std::max(a, b)}]++;
                }
            for (auto& [edge, cnt] : efc)
                if (cnt == 1) info.boundary_edges.insert(edge);
        }
        return info;
    }

    // General mesh mode: topological boundary detection.
    // An edge is a boundary edge if it is shared by exactly one triangle.
    std::map<std::pair<int,int>, int> edge_face_count;
    for (int f = 0; f < (int)F.rows(); ++f)
        for (int e = 0; e < 3; ++e) {
            int a = F(f, e), b = F(f, (e + 1) % 3);
            edge_face_count[{std::min(a, b), std::max(a, b)}]++;
        }

    for (auto& [edge, cnt] : edge_face_count) {
        if (cnt == 1) {
            info.boundary_edges.insert(edge);
            info.on_boundary[edge.first]  = true;
            info.on_boundary[edge.second] = true;
        }
    }

    // Count boundary edges incident to each vertex.
    std::vector<int> bdry_deg(n, 0);
    for (auto& edge : info.boundary_edges) {
        bdry_deg[edge.first]++;
        bdry_deg[edge.second]++;
    }

    // Corner = boundary vertex with ≠ 2 incident boundary edges
    // (endpoint of open boundary, or junction where ≥3 boundary components meet).
    for (int i = 0; i < n; ++i)
        if (info.on_boundary[i] && bdry_deg[i] != 2)
            info.on_corner[i] = true;

    return info;
}

Eigen::Vector2d EigenEdgeCollapse::constrained_pos(
    int vi, int vj,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const BoundaryInfo& binfo) const
{
    if (binfo.on_corner[vi]) return V.row(vi);
    if (binfo.on_corner[vj]) return V.row(vj);

    const bool bi = binfo.on_boundary[vi];
    const bool bj = binfo.on_boundary[vj];
    if (bi || bj) {
        if (bi && bj)
            return enclosure_pos(vi, vj, V, F);
        return bi ? Eigen::Vector2d(V.row(vi)) : Eigen::Vector2d(V.row(vj));
    }
    return 0.5 * (V.row(vi) + V.row(vj));
}

// Build m_coarse_face_fine_bdry_verts: for each coarse face, list the fine
// boundary vertex indices (from m_fine_boundary_verts) that lie inside it.
// Called once per simplify() step before candidate evaluation.
void EigenEdgeCollapse::rebuild_coarse_fine_overlap(
    const Eigen::MatrixXd& V_coarse,
    const Eigen::MatrixXi& F_coarse)
{
    const int nc  = (int)F_coarse.rows();
    const double eps = 1e-10;

    // Coarse face bounding boxes
    Eigen::MatrixXd bb_min(nc, 2), bb_max(nc, 2);
    for (int c = 0; c < nc; ++c)
        for (int d = 0; d < 2; ++d) {
            bb_min(c, d) = std::min({V_coarse(F_coarse(c,0), d),
                                     V_coarse(F_coarse(c,1), d),
                                     V_coarse(F_coarse(c,2), d)}) - eps;
            bb_max(c, d) = std::max({V_coarse(F_coarse(c,0), d),
                                     V_coarse(F_coarse(c,1), d),
                                     V_coarse(F_coarse(c,2), d)}) + eps;
        }

    m_coarse_face_fine_bdry_verts.assign(nc, {});

    for (int fbv : m_fine_boundary_verts) {
        const Eigen::Vector2d p = m_V_fine.row(fbv);

        for (int c = 0; c < nc; ++c) {
            if (p[0] < bb_min(c,0) || p[0] > bb_max(c,0) ||
                p[1] < bb_min(c,1) || p[1] > bb_max(c,1)) continue;

            const Eigen::Vector2d a  = V_coarse.row(F_coarse(c,0));
            const Eigen::Vector2d b  = V_coarse.row(F_coarse(c,1));
            const Eigen::Vector2d cc = V_coarse.row(F_coarse(c,2));

            const double den = (b-a)[0]*(cc-a)[1] - (cc-a)[0]*(b-a)[1];
            if (std::abs(den) < 1e-14) continue;

            const double inv = 1.0 / den;
            const Eigen::Vector2d v2 = p - a;
            const double u  = (v2[0]*(cc-a)[1] - (cc-a)[0]*v2[1]) * inv;
            const double v_ = ((b-a)[0]*v2[1]  - v2[0]*(b-a)[1]) * inv;
            const double w  = 1.0 - u - v_;

            if (u >= -eps && v_ >= -eps && w >= -eps)
                m_coarse_face_fine_bdry_verts[c].push_back(fbv);
        }
    }
}

// Fine-mesh-enclosing position for collapsing boundary edge (vi, vj).
// Searches ALL coarse faces in the 1-ring of vi AND vj for the fine boundary
// vertex that is farthest "outward" (away from the mesh interior) relative to
// the chord vi–vj.  Searching the full 1-ring is necessary because after the
// collapse every face incident to vi or vj changes shape; a fine boundary
// vertex in those faces can become uncovered if v_new is too far inward.
// Falls back to the midpoint when the boundary is straight (rectangle sides,
// where all projections are ≈ 0 by construction).
Eigen::Vector2d EigenEdgeCollapse::enclosure_pos(
    int vi, int vj,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F) const
{
    const Eigen::Vector2d vi_pos = V.row(vi), vj_pos = V.row(vj);
    const Eigen::Vector2d mid = 0.5 * (vi_pos + vj_pos);

    if (m_coarse_face_fine_bdry_verts.empty()) return mid;

    int adj_face = -1;
    std::vector<int> ring_faces;
    ring_faces.reserve(16);
    for (int f = 0; f < (int)F.rows(); ++f) {
        bool has_vi = false, has_vj = false;
        for (int k = 0; k < 3; ++k) {
            if (F(f, k) == vi) has_vi = true;
            if (F(f, k) == vj) has_vj = true;
        }
        if (has_vi || has_vj) {
            ring_faces.push_back(f);
            if (has_vi && has_vj) adj_face = f;
        }
    }
    if (adj_face < 0) return mid;

    // Collect all fine boundary vertices seen in any ring face (AND or adj).
    std::vector<int> seen_fbvs;
    {
        std::set<int> seen;
        for (int f : ring_faces) {
            if (f >= (int)m_coarse_face_fine_bdry_verts.size()) continue;
            for (int fbv : m_coarse_face_fine_bdry_verts[f])
                seen.insert(fbv);
        }
        seen_fbvs.assign(seen.begin(), seen.end());
    }
    if (seen_fbvs.empty()) return mid;

    // Build post-collapse face descriptors {a, b, sgn} for each non-adj ring face.
    // After collapse the face becomes [p_new, a, b] (p_new replaces rep_v).
    struct PostFace { Eigen::Vector2d a, b; double sgn; };
    std::vector<PostFace> post_faces;
    for (int f : ring_faces) {
        if (f == adj_face) continue;
        int krep = -1;
        for (int k = 0; k < 3; ++k)
            if (F(f,k) == vi || F(f,k) == vj) { krep = k; break; }
        if (krep < 0) continue;
        const Eigen::Vector2d o = V.row(F(f, krep));
        const Eigen::Vector2d a = V.row(F(f, (krep+1)%3));
        const Eigen::Vector2d b = V.row(F(f, (krep+2)%3));
        const double ori = (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
        post_faces.push_back({a, b, (ori >= 0.0) ? 1.0 : -1.0});
    }
    if (post_faces.empty()) return mid;

    auto cross2 = [](const Eigen::Vector2d& u, const Eigen::Vector2d& v) -> double {
        return u[0]*v[1] - u[1]*v[0];
    };

    // How well does p cover fbv q in post-collapse face [p, a, b]?
    // Returns min(s1, s2, s3); s2 is p-independent (filters geometrically impossible faces).
    auto face_slack = [&](const Eigen::Vector2d& p, const Eigen::Vector2d& q,
                          const PostFace& pf) -> double {
        const double s1 = pf.sgn * cross2(pf.a - p, q - p);
        const double s2 = pf.sgn * cross2(pf.b - pf.a, q - pf.a);
        const double s3 = pf.sgn * cross2(p - pf.b, q - pf.b);
        return std::min({s1, s2, s3});
    };

    auto is_covered = [&](const Eigen::Vector2d& p, int fbv) -> bool {
        const Eigen::Vector2d q = m_V_fine.row(fbv);
        for (const auto& pf : post_faces)
            if (face_slack(p, q, pf) >= -1e-9) return true;
        return false;
    };

    auto coverage_count = [&](const Eigen::Vector2d& p) -> int {
        int cnt = 0;
        for (int fbv : seen_fbvs)
            if (is_covered(p, fbv)) ++cnt;
        return cnt;
    };

    // Start from the candidate with the most initial coverage.
    Eigen::Vector2d p = mid;
    {
        int best_cov = coverage_count(mid);
        for (const Eigen::Vector2d& cand : {vi_pos, vj_pos}) {
            int c = coverage_count(cand);
            if (c > best_cov) { best_cov = c; p = cand; }
        }
    }

    Eigen::Vector2d best_p = p;
    int best_ever = coverage_count(p);
    const int n_fbvs = (int)seen_fbvs.size();

    // OR-POCS: for each uncovered fbv, project onto the halfplanes of the
    // post-collapse face that currently gives the highest face_slack.
    // Face assignment is recomputed dynamically every iteration.
    for (int iter = 0; iter < 2000 && best_ever < n_fbvs; ++iter) {
        for (int fbv : seen_fbvs) {
            if (is_covered(p, fbv)) continue;
            const Eigen::Vector2d q = m_V_fine.row(fbv);

            // Pick the post-face with the best (least negative) slack for q.
            int    best_fi    = -1;
            double best_sl    = -1e18;
            for (int fi = 0; fi < (int)post_faces.size(); ++fi) {
                double sl = face_slack(p, q, post_faces[fi]);
                if (sl > best_sl) { best_sl = sl; best_fi = fi; }
            }
            if (best_fi < 0) continue;

            const PostFace& pf = post_faces[best_fi];

            // Project onto n1: sgn * cross(a-p, q-p) >= 0
            {
                const Eigen::Vector2d n1 = pf.sgn * Eigen::Vector2d(pf.a[1]-q[1], q[0]-pf.a[0]);
                const double c1 = pf.sgn * (pf.a[1]*q[0] - pf.a[0]*q[1]);
                const double sl = n1.dot(p) - c1;
                if (sl < -1e-12) {
                    const double nn = n1.squaredNorm();
                    if (nn > 1e-24) p -= (sl / nn) * n1;
                }
            }
            // Project onto n3: sgn * cross(p-b, q-b) >= 0
            {
                const Eigen::Vector2d n3 = pf.sgn * Eigen::Vector2d(q[1]-pf.b[1], pf.b[0]-q[0]);
                const double c3 = pf.sgn * (pf.b[0]*q[1] - pf.b[1]*q[0]);
                const double sl = n3.dot(p) - c3;
                if (sl < -1e-12) {
                    const double nn = n3.squaredNorm();
                    if (nn > 1e-24) p -= (sl / nn) * n3;
                }
            }
        }

        const int cov = coverage_count(p);
        if (cov > best_ever) { best_ever = cov; best_p = p; }
    }

    return best_p;
}

// ============================================================
//  Barycentric prolongation (n_fine × n_coarse sparse)
// ============================================================

Eigen::SparseMatrix<double> EigenEdgeCollapse::build_barycentric_P(
    const Eigen::MatrixXd& V_coarse,
    const Eigen::MatrixXi& F_coarse) const
{
    const int n_f = (int)m_V_fine.rows();
    const int n_c = (int)V_coarse.rows();
    const double eps = 1e-10;

    // Precompute bounding boxes
    const int nt = (int)F_coarse.rows();
    Eigen::MatrixXd bb_min(nt, 2), bb_max(nt, 2);
    for (int t = 0; t < nt; ++t) {
        for (int d = 0; d < 2; ++d) {
            double lo = std::min({V_coarse(F_coarse(t, 0), d),
                                  V_coarse(F_coarse(t, 1), d),
                                  V_coarse(F_coarse(t, 2), d)});
            double hi = std::max({V_coarse(F_coarse(t, 0), d),
                                  V_coarse(F_coarse(t, 1), d),
                                  V_coarse(F_coarse(t, 2), d)});
            bb_min(t, d) = lo - eps;
            bb_max(t, d) = hi + eps;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n_f * 3);

    for (int pf = 0; pf < n_f; ++pf) {
        const Eigen::Vector2d p = m_V_fine.row(pf);
        bool found = false;

        for (int t = 0; t < nt && !found; ++t) {
            if (p[0] < bb_min(t, 0) || p[0] > bb_max(t, 0) ||
                p[1] < bb_min(t, 1) || p[1] > bb_max(t, 1))
                continue;

            const Eigen::Vector2d a = V_coarse.row(F_coarse(t, 0));
            const Eigen::Vector2d b = V_coarse.row(F_coarse(t, 1));
            const Eigen::Vector2d c = V_coarse.row(F_coarse(t, 2));

            const double den = (b - a)[0] * (c - a)[1] - (c - a)[0] * (b - a)[1];
            if (std::abs(den) < eps) continue;

            const Eigen::Vector2d v2 = p - a;
            const double inv = 1.0 / den;
            const double u = (v2[0] * (c - a)[1] - (c - a)[0] * v2[1]) * inv;
            const double v = ((b - a)[0] * v2[1] - v2[0] * (b - a)[1]) * inv;
            const double w = 1.0 - u - v;

            if (u >= -eps && v >= -eps && w >= -eps &&
                u <= 1.0 + eps && v <= 1.0 + eps && w <= 1.0 + eps)
            {
                std::array<double, 3> bary = {
                    std::max(0.0, std::min(1.0, w)),
                    std::max(0.0, std::min(1.0, u)),
                    std::max(0.0, std::min(1.0, v)),
                };
                double s = bary[0] + bary[1] + bary[2];
                if (s > 0) { bary[0] /= s; bary[1] /= s; bary[2] /= s; }

                for (int k = 0; k < 3; ++k)
                    if (std::abs(bary[k]) > 1e-15)
                        triplets.emplace_back(pf, F_coarse(t, k), bary[k]);
                found = true;
            }
        }

        if (!found) {
            // Fallback: nearest coarse vertex
            int nearest = 0;
            double best = std::numeric_limits<double>::max();
            for (int c = 0; c < n_c; ++c) {
                double d = (V_coarse.row(c) - m_V_fine.row(pf)).squaredNorm();
                if (d < best) { best = d; nearest = c; }
            }
            triplets.emplace_back(pf, nearest, 1.0);
        }
    }

    Eigen::SparseMatrix<double> Ps(n_f, n_c);
    Ps.setFromTriplets(triplets.begin(), triplets.end());
    return Ps;
}

// ============================================================
//  Static edge collapse on (V, F) arrays (no TriMesh involved)
// ============================================================

bool EigenEdgeCollapse::collapse_edge_static(
    const Eigen::MatrixXd& V_in,
    const Eigen::MatrixXi& F_in,
    int vi, int vj,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& F_out) const
{
    // Keep vi, remove vj (vi < vj by convention)
    if (vi > vj) std::swap(vi, vj);

    const auto binfo = classify_boundary(V_in, F_in);

    // Reject boundary–boundary collapses that cross the interior
    // (both endpoints on boundary but edge is not a boundary edge).
    if (binfo.on_boundary[vi] && binfo.on_boundary[vj]) {
        auto key = std::make_pair(std::min(vi, vj), std::max(vi, vj));
        if (!binfo.boundary_edges.count(key)) return false;
    }

    Eigen::Vector2d cpos = constrained_pos(vi, vj, V_in, F_in, binfo);

    // Build old->new vertex map (remove vj)
    const int n_old = (int)V_in.rows();
    std::vector<int> old_to_new(n_old, -1);
    int cnt = 0;
    for (int i = 0; i < n_old; ++i) {
        if (i == vj) continue;
        old_to_new[i] = cnt++;
    }
    old_to_new[vj] = old_to_new[vi]; // vj maps to vi's new index

    V_out.resize(cnt, 2);
    for (int i = 0; i < n_old; ++i) {
        if (i == vj) continue;
        if (i == vi)
            V_out.row(old_to_new[i]) = cpos.transpose();
        else
            V_out.row(old_to_new[i]) = V_in.row(i);
    }

    // Remap faces, drop degenerate
    std::vector<Eigen::Vector3i> faces;
    faces.reserve(F_in.rows());
    for (int f = 0; f < (int)F_in.rows(); ++f) {
        int a = old_to_new[F_in(f, 0)];
        int b = old_to_new[F_in(f, 1)];
        int c = old_to_new[F_in(f, 2)];
        if (a == b || b == c || a == c) continue;

        // Check area — reject if degenerate or flipped
        Eigen::Vector2d p0 = V_out.row(a);
        Eigen::Vector2d p1 = V_out.row(b);
        Eigen::Vector2d p2 = V_out.row(c);
        double area2 = (p1 - p0)[0] * (p2 - p0)[1] - (p2 - p0)[0] * (p1 - p0)[1];
        if (area2 <= 1e-14) return false; // face flip or degenerate → infeasible
        faces.push_back({a, b, c});
    }

    if (faces.empty()) return false;

    F_out.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); ++i)
        F_out.row(i) = faces[i];

    return true;
}

// ============================================================
//  Cost for a candidate coarse mesh
// ============================================================

double EigenEdgeCollapse::candidate_cost(
    const Eigen::MatrixXd& V_cand,
    const Eigen::MatrixXi& F_cand) const
{
    if (F_cand.rows() == 0) return std::numeric_limits<double>::infinity();

    // Assemble coarse K
    Eigen::SparseMatrix<double> Kc;
    Eigen::VectorXd Mc_diag;
    assemble_fem(V_cand, F_cand, Kc, Mc_diag);
    apply_spring_bcs(Kc, V_cand, F_cand);

    // const int ndof_c = (int)Kc.rows();
    const int ndof_c = 2 * (int)V_cand.rows();

    // Boundary conditions on coarse
    std::vector<int> fixed_c_verts =
        (m_p.spring_k > 0.0) ? std::vector<int>{}
        : (m_p.fixed_left    ? left_boundary_verts(V_cand) : std::vector<int>{});
    std::vector<int> free_c = free_dof_indices(ndof_c, fixed_c_verts);
    if (free_c.empty()) return std::numeric_limits<double>::infinity();

    // Extract K_c_ff
    const int nf_c = (int)free_c.size();
    Eigen::SparseMatrix<double> Kc_ff(nf_c, nf_c);
    {
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(Kc.nonZeros());
        // Build index map
        std::vector<int> idx(ndof_c, -1);
        for (int k = 0; k < nf_c; ++k) idx[free_c[k]] = k;

        for (int j = 0; j < nf_c; ++j) {
            int col = free_c[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(Kc, col); it; ++it) {
                int row_global = (int)it.row();
                if (idx[row_global] >= 0)
                    tr.emplace_back(idx[row_global], j, it.value());
            }
        }
        Kc_ff.setFromTriplets(tr.begin(), tr.end());
    }

    // // Kinetic shift on coarse K_eff = K_c + alpha*M_c (matches fine-mesh shift).
    // // Skipped when spring_k > 0 (springs already regularise the system).
    if (m_p.spring_k <= 0.0 && !m_p.fixed_left && m_p.alpha > 0.0)
        for (int k = 0; k < nf_c; ++k)
            Kc_ff.coeffRef(k, k) += m_p.alpha * Mc_diag[free_c[k]];

    auto time_start = std::chrono::high_resolution_clock::now();
    // Factorize
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(Kc_ff);
    // if (ldlt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();


    Kc_ff.makeCompressed();
    CholFactor chol = cholmod_factorize_spd(Kc_ff);
    if (!chol.L) return std::numeric_limits<double>::infinity();



    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> factor_time = time_end - time_start;
    // std::cout << "Factorization time: " << factor_time.count() << " seconds\n";

    // Barycentric prolongation P_s (n_fine × n_coarse scalar)
    Eigen::SparseMatrix<double> Ps = build_barycentric_P(V_cand, F_cand);

    // Expand to vector DOF: P (2*n_fine × 2*n_coarse)
    // P_free restricts coarse to free DOFs: (2*n_fine × nf_c)
    // P_free[2r, k]   = Ps[r, free_c[k]/2] if free_c[k] is even
    // P_free[2r+1, k] = Ps[r, free_c[k]/2] if free_c[k] is odd
    const int n_f = (int)m_V_fine.rows();
    const int ndof_f = 2 * n_f;

    // Build P_free as a sparse matrix (preserves sparsity of Ps).
    Eigen::SparseMatrix<double> P_free(ndof_f, nf_c);
    {
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(Ps.nonZeros());
        for (int k = 0; k < nf_c; ++k) {
            int gc = free_c[k];
            int vc = gc / 2;
            int comp = gc % 2; // 0=x, 1=y
            // Column k of P_free: set rows 2*pf+comp = Ps(pf, vc)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ps, vc); it; ++it) {
                int pf = (int)it.row();
                tr.emplace_back(2 * pf + comp, k, it.value());
            }
        }
        P_free.setFromTriplets(tr.begin(), tr.end());
        P_free.makeCompressed();
    }

    // M_f as diagonal vector (already stored)
    const Eigen::VectorXd& Mfd = m_M_diag;

    double cost = 0.0;
    int used = 0;

    // Eigen::MatrixXd P_Mfd_inv = P_free.transpose() * Mfd.cwiseInverse().asDiagonal();
    // Eigen::MatrixXd Mc_diag_inv_Kcff_P = Mc_diag.cwiseInverse().asDiagonal() * Kc_ff * P_free.transpose();

    // Avoid materializing PPT = P_free * P_free^T (potentially dense).
    // Apply P_free^T and P_free directly to each mode vector instead.

    // Restrict M_c to free DOFs so dimensions match Kc_ff (nf_c × nf_c).


    auto time_start_eval = std::chrono::high_resolution_clock::now();
    for (int mode_id : m_modes) {
        const double lam = m_evals[mode_id];
        if (!std::isfinite(lam) || std::abs(lam) <= m_p.eig_tol) continue;

        Eigen::VectorXd phi = m_evecs.col(mode_id); // 2*n_fine

        // // b_f = M_f phi
        Eigen::VectorXd bf = Mfd.cwiseProduct(phi);

        // b_c = P_free^T b_f (nf_c vector)
        Eigen::VectorXd bc = P_free.transpose() * bf;

        // Solve K_c_ff y = b_c
        // Eigen::VectorXd yc = ldlt.solve(bc);
        // if (ldlt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
        Eigen::VectorXd yc = cholmod_solve_vec(chol.L, bc);
        if (yc.size() == 0) return std::numeric_limits<double>::infinity();

        Eigen::VectorXd residual = Kc_ff * yc - bc;
        double res_norm = residual.norm();
        double bf_norm = bf.norm();
        if (res_norm > 1e-1) {
            std::cerr << "Warning: high residual norm for mode " << mode_id << ": ||Kc_ff*yc - bc|| = " << res_norm
                      << " (relative: " << res_norm / (bf_norm + 1e-15) << ")\n";
        }

        // Coarse response projected back: P_free * yc
        Eigen::VectorXd response = P_free * yc;

        // Target: (1/lambda) phi
        Eigen::VectorXd diff = (1.0 / lam) * phi - response;

        double norm2 = (Mfd.cwiseProduct(diff)).dot(diff);

        // Weight
        double w = 1.0;
        if (m_p.weight_mode == 1)      w = 1.0 / std::abs(lam);
        else if (m_p.weight_mode == 2) w = 1.0 / (lam * lam);
        w = 1;
        cost += w * norm2;

        // diff = phi - P_free * (P_free^T * phi)  (equivalent to phi - PPT * phi)
        // Eigen::VectorXd diff = phi - P_free * (P_free.transpose() * phi);


        


    

        


        
        // Additional term: || H^-1 phi - H^-1 P P^T phi ||^2_{M_f}
        // (K_f is the fine-mesh stiffness restricted to free DOFs, factor cached in init).
        // Eigen::VectorXd diff_full = phi - P_free * (P_free.transpose() * phi);
        // const int nf_f = (int)m_free_fine.size();
        // Eigen::VectorXd diff_ff(nf_f);
        // for (int k = 0; k < nf_f; ++k) diff_ff[k] = diff_full[m_free_fine[k]];
        // Eigen::VectorXd Kf_inv_diff = m_K_fine_ff_solver.solve(diff_ff);
        // // free to full mapping: Kf_inv_diff_full[i] = Kf_inv_diff[j] if m_free_fine[j] = i, else 0
        // Eigen::VectorXd Kf_inv_diff_full(Mfd.size());
        // Kf_inv_diff_full.setZero();
        // for (int k = 0; k < nf_f; ++k) {
        //     Kf_inv_diff_full[m_free_fine[k]] = Kf_inv_diff[k];
        // }

        // double norm2_Kf = (Mfd.cwiseProduct(Kf_inv_diff_full)).dot(Kf_inv_diff_full);

        // const double norm2_Kf = diff_ff.dot(Kf_inv_diff);

        // cost += w * norm2_Kf;


        // Eigen::VectorXd weighted_phi = lam * P_Mfd_inv * phi;

        // Eigen::VectorXd weighted_bc = Mc_diag_inv_Kcff_P * phi;
        // Eigen::VectorXd diff = weighted_phi - weighted_bc;
        // double norm2 = (Mc_diag.cwiseProduct(diff)).dot(diff);
        // std::cout << "Mode " << mode_id << ": lambda=" << lam << ", cost contribution=" << norm2 << "\n";
        
        ++used;
    }
    auto time_end_eval = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eval_time = time_end_eval - time_start_eval;
    // std::cout << "Evaluation time: " << eval_time.count() << " seconds\n";

    if (used == 0) return std::numeric_limits<double>::infinity();
    return cost;
}

// ============================================================
//  Accelerate-based Woodbury candidate cost
// ============================================================
// Same algorithm as candidate_cost_woodbury, with all dense and sparse
// matrix-vector products routed through Apple Accelerate (cblas_dgemv,
// SparseMultiply, vDSP). The inner CHOLMOD solve on wb.factor.L is reused.

double EigenEdgeCollapse::candidate_cost_woodbury_accelerated(
    const Eigen::MatrixXd& V_cand,
    const Eigen::MatrixXi& F_cand,
    int vi_curr, int vj_curr) const
{
    if (!m_wb_base || !m_wb_base->factor.L)
        return std::numeric_limits<double>::infinity();
    if (F_cand.rows() == 0) return std::numeric_limits<double>::infinity();
    const auto& wb = *m_wb_base;

    if (vi_curr > vj_curr) std::swap(vi_curr, vj_curr);

    WbaSectionTimer _wba_timer;

    // ── 1) dofs_del ──────────────────────────────────────────────────────────
    _wba_timer.next(0);
    std::vector<int> dofs_del;
    dofs_del.reserve(2);
    for (int c = 0; c < 2; ++c) {
        const int full = 2 * vj_curr + c;
        const int fi   = wb.dof_full_to_free[full];
        if (fi >= 0) dofs_del.push_back(fi);
    }
    if (dofs_del.empty()) return std::numeric_limits<double>::infinity();

    const int vi_in_cand = vi_curr;
    auto cand_to_curr = [vj_curr](int kc) { return (kc < vj_curr) ? kc : kc + 1; };

    // ── 2) old / new triangles touching the collapsed vertices ───────────────
    _wba_timer.next(1);
    std::vector<int> old_tris; old_tris.reserve(16);
    std::set<int> affected;
    for (int f = 0; f < (int)wb.F.rows(); ++f) {
        const int v0 = wb.F(f, 0), v1 = wb.F(f, 1), v2 = wb.F(f, 2);
        if (v0 == vi_curr || v1 == vi_curr || v2 == vi_curr ||
            v0 == vj_curr || v1 == vj_curr || v2 == vj_curr) {
            old_tris.push_back(f);
            affected.insert(v0); affected.insert(v1); affected.insert(v2);
        }
    }
    std::vector<int> new_tris; new_tris.reserve(old_tris.size());
    for (int f = 0; f < (int)F_cand.rows(); ++f) {
        const int v0 = F_cand(f, 0), v1 = F_cand(f, 1), v2 = F_cand(f, 2);
        if (v0 == vi_in_cand || v1 == vi_in_cand || v2 == vi_in_cand) {
            new_tris.push_back(f);
            affected.insert(cand_to_curr(v0));
            affected.insert(cand_to_curr(v1));
            affected.insert(cand_to_curr(v2));
        }
    }
    affected.erase(vj_curr);

    // ── 2b) Spring delta tracking ────────────────────────────────────────────
    // For each user-specified spring fine vertex vf, compute its barycentric
    // expansion in BOTH (wb.V, wb.F) [old] and (V_cand, F_cand) [new].  The
    // contribution of vf to K_c is spring_k · bary · bary^T at the affected
    // coarse vertices.  When the collapse changes vf's containing triangle or
    // weights, K_new − K_old picks up a non-zero delta which must be folded
    // into Σ for the Woodbury update to be correct.
    struct SpringChange {
        std::array<double, 3> bary_old{};   // weights in (wb.V, wb.F)
        std::array<int,    3> verts_old{};  // V_curr indices
        std::array<double, 3> bary_new{};   // weights in (V_cand, F_cand)
        std::array<int,    3> verts_new{};  // V_curr indices (via cand_to_curr)
        bool changed = false;
    };
    std::vector<SpringChange> spring_changes;

    auto find_bary_in = [&](
        const Eigen::Vector2d& p,
        const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
        std::array<double, 3>& bary, std::array<int, 3>& verts) -> bool
    {
        const double eps = 1e-10;
        const int n_t = (int)F.rows();
        for (int t = 0; t < n_t; ++t) {
            const Eigen::Vector2d a = V.row(F(t, 0));
            const Eigen::Vector2d b = V.row(F(t, 1));
            const Eigen::Vector2d c = V.row(F(t, 2));
            const double den = (b - a)[0] * (c - a)[1] - (c - a)[0] * (b - a)[1];
            if (std::abs(den) < eps) continue;
            const Eigen::Vector2d v2 = p - a;
            const double inv = 1.0 / den;
            const double u = (v2[0] * (c - a)[1] - (c - a)[0] * v2[1]) * inv;
            const double v = ((b - a)[0] * v2[1] - v2[0] * (b - a)[1]) * inv;
            const double w = 1.0 - u - v;
            if (u >= -eps && v >= -eps && w >= -eps &&
                u <= 1.0 + eps && v <= 1.0 + eps && w <= 1.0 + eps) {
                bary  = {std::clamp(w, 0.0, 1.0),
                         std::clamp(u, 0.0, 1.0),
                         std::clamp(v, 0.0, 1.0)};
                const double s_b = bary[0] + bary[1] + bary[2];
                if (s_b > 0) { bary[0] /= s_b; bary[1] /= s_b; bary[2] /= s_b; }
                verts = {F(t, 0), F(t, 1), F(t, 2)};
                return true;
            }
        }
        // nearest-vertex fallback
        int nearest = 0;
        double best = std::numeric_limits<double>::max();
        for (int v = 0; v < (int)V.rows(); ++v) {
            const double d = (Eigen::Vector2d(V.row(v)) - p).squaredNorm();
            if (d < best) { best = d; nearest = v; }
        }
        bary  = {1.0, 0.0, 0.0};
        verts = {nearest, nearest, nearest};
        return false;
    };

    if (m_p.spring_k > 0.0 && !m_spring_fine_verts.empty()) {
        spring_changes.reserve(m_spring_fine_verts.size());
        for (int vf : m_spring_fine_verts) {
            if (vf < 0 || vf >= (int)m_V_fine.rows()) continue;
            const Eigen::Vector2d p = m_V_fine.row(vf);

            SpringChange sc;
            find_bary_in(p, wb.V,  wb.F,  sc.bary_old, sc.verts_old);
            std::array<int, 3> verts_new_cand{};
            find_bary_in(p, V_cand, F_cand, sc.bary_new, verts_new_cand);
            for (int k = 0; k < 3; ++k)
                sc.verts_new[k] = cand_to_curr(verts_new_cand[k]);

            // Detect whether bary changed (same triangle + weights → no delta).
            sc.changed = false;
            for (int k = 0; k < 3; ++k) {
                if (sc.verts_old[k] != sc.verts_new[k] ||
                    std::abs(sc.bary_old[k] - sc.bary_new[k]) > 1e-12) {
                    sc.changed = true; break;
                }
            }
            spring_changes.push_back(sc);
        }
    }

    // ── 3) s_dofs ────────────────────────────────────────────────────────────
    _wba_timer.next(2);
    std::vector<int> s_dofs;
    s_dofs.reserve(affected.size() * 2 + spring_changes.size() * 12);
    for (int v : affected) {
        for (int c = 0; c < 2; ++c) {
            const int full = 2 * v + c;
            const int fi   = wb.dof_full_to_free[full];
            if (fi >= 0) s_dofs.push_back(fi);
        }
    }
    // Include coarse DOFs touched by spring bary changes (exclude vj's DOFs,
    // which are deleted in the r-space; their contributions are handled by the
    // block-deletion Woodbury, not by Σ).
    for (const auto& sc : spring_changes) {
        if (!sc.changed) continue;
        for (int k = 0; k < 3; ++k) {
            if (sc.verts_old[k] != vj_curr) {
                for (int c = 0; c < 2; ++c) {
                    const int fi_o = wb.dof_full_to_free[2 * sc.verts_old[k] + c];
                    if (fi_o >= 0) s_dofs.push_back(fi_o);
                }
            }
            if (sc.verts_new[k] != vj_curr) {
                for (int c = 0; c < 2; ++c) {
                    const int fi_n = wb.dof_full_to_free[2 * sc.verts_new[k] + c];
                    if (fi_n >= 0) s_dofs.push_back(fi_n);
                }
            }
        }
    }
    std::sort(s_dofs.begin(), s_dofs.end());
    s_dofs.erase(std::unique(s_dofs.begin(), s_dofs.end()), s_dofs.end());
    const int s = (int)s_dofs.size();
    if (s == 0) return std::numeric_limits<double>::infinity();

    std::vector<int> free_to_s(wb.nfree, -1);
    for (int k = 0; k < s; ++k) free_to_s[s_dofs[k]] = k;

    // ── 4) Build Σ = K_new[s,s] − K_old[s,s] ─────────────────────────────────
    _wba_timer.next(3);
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(s, s);
    const bool has_kinetic = (m_p.spring_k <= 0.0 && !m_p.fixed_left && m_p.alpha > 0.0);
    std::vector<double> dM_at_v;
    if (has_kinetic) dM_at_v.assign(wb.V.rows(), 0.0);

    auto accumulate_element = [&](
        const std::array<int,3>&             tri_curr,
        const std::array<Eigen::Vector2d,3>& pos,
        double                               sign)
    {
        const double x1 = pos[0][0], y1 = pos[0][1];
        const double x2 = pos[1][0], y2 = pos[1][1];
        const double x3 = pos[2][0], y3 = pos[2][1];
        const double signed_area2 = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
        const double area = 0.5 * std::abs(signed_area2);
        if (area <= 1e-14) return;

        Eigen::Vector3d dNdx, dNdy;
        dNdx << (y2 - y3), (y3 - y1), (y1 - y2);
        dNdy << (x3 - x2), (x1 - x3), (x2 - x1);
        dNdx /= signed_area2;
        dNdy /= signed_area2;

        Eigen::Matrix<double, 3, 6> B = Eigen::Matrix<double, 3, 6>::Zero();
        for (int a = 0; a < 3; ++a) {
            B(0, 2*a)     = dNdx[a];
            B(1, 2*a + 1) = dNdy[a];
            B(2, 2*a)     = dNdy[a];
            B(2, 2*a + 1) = dNdx[a];
        }

        Eigen::MatrixXd Vlocal(3, 2);
        Vlocal << x1, y1, x2, y2, x3, y3;
        const Eigen::Vector3i tri_local(0, 1, 2);
        const double E  = material_E (Vlocal, tri_local);
        const double nu = material_nu(Vlocal, tri_local);

        const Eigen::Matrix<double, 6, 6> Ke =
            area * (B.transpose() * elasticity_D(E, nu) * B);

        std::array<int, 6> sidx{};
        for (int a = 0; a < 3; ++a) {
            const int v_curr = tri_curr[a];
            for (int c = 0; c < 2; ++c) {
                const int full = 2 * v_curr + c;
                const int fi   = wb.dof_full_to_free[full];
                sidx[2*a + c] = (fi >= 0) ? free_to_s[fi] : -1;
            }
        }
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                if (sidx[a] >= 0 && sidx[b] >= 0)
                    Sigma(sidx[a], sidx[b]) += sign * Ke(a, b);

        if (has_kinetic) {
            const double m_lump = m_p.rho * area / 3.0;
            for (int a = 0; a < 3; ++a) dM_at_v[tri_curr[a]] += sign * m_lump;
        }
    };

    for (int f : old_tris) {
        const std::array<int, 3> tri = {wb.F(f,0), wb.F(f,1), wb.F(f,2)};
        const std::array<Eigen::Vector2d, 3> pos = {
            Eigen::Vector2d(wb.V.row(tri[0])),
            Eigen::Vector2d(wb.V.row(tri[1])),
            Eigen::Vector2d(wb.V.row(tri[2])),
        };
        accumulate_element(tri, pos, -1.0);
    }
    for (int f : new_tris) {
        const std::array<int, 3> tri_cand = {F_cand(f,0), F_cand(f,1), F_cand(f,2)};
        const std::array<int, 3> tri_curr = {
            cand_to_curr(tri_cand[0]),
            cand_to_curr(tri_cand[1]),
            cand_to_curr(tri_cand[2])
        };
        const std::array<Eigen::Vector2d, 3> pos = {
            Eigen::Vector2d(V_cand.row(tri_cand[0])),
            Eigen::Vector2d(V_cand.row(tri_cand[1])),
            Eigen::Vector2d(V_cand.row(tri_cand[2])),
        };
        accumulate_element(tri_curr, pos, +1.0);
    }

    if (has_kinetic) {
        for (int v = 0; v < (int)dM_at_v.size(); ++v) {
            if (dM_at_v[v] == 0.0) continue;
            for (int c = 0; c < 2; ++c) {
                const int full = 2 * v + c;
                const int fi   = wb.dof_full_to_free[full];
                if (fi < 0) continue;
                const int si = free_to_s[fi];
                if (si < 0) continue;
                Sigma(si, si) += m_p.alpha * dM_at_v[v];
            }
        }
    }

    // ── 4b) Σ ← Σ + (P_cand^T K_s P_cand − P_curr^T K_s P_curr) ──────────────
    // wb.K_bc already contains P_curr^T K_s P_curr (added in rebuild_woodbury_base).
    // The candidate's K_c needs P_cand^T K_s P_cand, so the Woodbury delta must
    // include (new − old) of the spring contribution for every spring fine vert
    // whose containing coarse triangle changed.
    if (m_p.spring_k > 0.0) {
        for (const auto& sc : spring_changes) {
            if (!sc.changed) continue;
            for (int comp = 0; comp < 2; ++comp) {
                // Subtract old: P_curr^T K_s P_curr at bary_old / verts_old
                for (int i = 0; i < 3; ++i) {
                    const double wi = sc.bary_old[i];
                    if (wi == 0.0) continue;
                    const int fi_i = wb.dof_full_to_free[2 * sc.verts_old[i] + comp];
                    if (fi_i < 0) continue;
                    const int si_i = free_to_s[fi_i];
                    if (si_i < 0) continue;
                    for (int j = 0; j < 3; ++j) {
                        const double wj = sc.bary_old[j];
                        if (wj == 0.0) continue;
                        const int fi_j = wb.dof_full_to_free[2 * sc.verts_old[j] + comp];
                        if (fi_j < 0) continue;
                        const int si_j = free_to_s[fi_j];
                        if (si_j < 0) continue;
                        Sigma(si_i, si_j) -= m_p.spring_k * wi * wj;
                    }
                }
                // Add new: P_cand^T K_s P_cand at bary_new / verts_new
                for (int i = 0; i < 3; ++i) {
                    const double wi = sc.bary_new[i];
                    if (wi == 0.0) continue;
                    const int fi_i = wb.dof_full_to_free[2 * sc.verts_new[i] + comp];
                    if (fi_i < 0) continue;
                    const int si_i = free_to_s[fi_i];
                    if (si_i < 0) continue;
                    for (int j = 0; j < 3; ++j) {
                        const double wj = sc.bary_new[j];
                        if (wj == 0.0) continue;
                        const int fi_j = wb.dof_full_to_free[2 * sc.verts_new[j] + comp];
                        if (fi_j < 0) continue;
                        const int si_j = free_to_s[fi_j];
                        if (si_j < 0) continue;
                        Sigma(si_i, si_j) += m_p.spring_k * wi * wj;
                    }
                }
            }
        }
    }

    // ── 5) Woodbury precompute (inlined Apple-Accelerate version) ────────────
    // Mirrors precompute_krr + precompute_knew, but every K_bc^{-1} solve
    // goes through Apple's SparseSolve on wb.apple_factor instead of CHOLMOD.
    _wba_timer.next(4);
    if (!wb.apple_factor.valid)
        return std::numeric_limits<double>::infinity();

    KrrPrecomp kp;
    {
        kp.dofs_del = dofs_del;
        const int n  = wb.nfree;
        const int nd = (int)dofs_del.size();

        Eigen::MatrixXd Bs = Eigen::MatrixXd::Zero(n, nd);
        for (int k = 0; k < nd; ++k) Bs(dofs_del[k], k) = 1.0;

        Eigen::MatrixXd Q_full = apple_solve_mat(wb.apple_factor, Bs);

        std::vector<bool> is_del(n, false);
        for (int d : dofs_del) is_del[d] = true;
        kp.r_dofs.reserve(n - nd);
        kp.free_to_r.assign(n, -1);
        for (int i = 0; i < n; ++i) {
            if (!is_del[i]) {
                kp.free_to_r[i] = (int)kp.r_dofs.size();
                kp.r_dofs.push_back(i);
            }
        }

        const int n_r = (int)kp.r_dofs.size();
        kp.Q.resize(n_r, nd);
        for (int i = 0; i < n_r; ++i) kp.Q.row(i) = Q_full.row(kp.r_dofs[i]);

        Eigen::MatrixXd Q_d(nd, nd);
        for (int i = 0; i < nd; ++i) Q_d.row(i) = Q_full.row(dofs_del[i]);
        kp.Q_d_inv = Q_d.inverse();
    }

    KnewPrecomp knew;
    {
        const int n_r = (int)kp.r_dofs.size();
        const int sk  = (int)s_dofs.size();

        knew.s_in_r.resize(sk);
        for (int k = 0; k < sk; ++k) {
            const int rk = kp.free_to_r[s_dofs[k]];
            if (rk < 0) return std::numeric_limits<double>::infinity();
            knew.s_in_r[k] = rk;
        }

        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n_r, sk);
        for (int k = 0; k < sk; ++k) U(knew.s_in_r[k], k) = 1.0;

        // W = K_rr^{-1} U via Apple solve + block-deletion correction.
        // Inlined apply_krr_inv_mat: embed cols of U at r_dofs, solve K_bc^{-1},
        // gather z_d / z_r, then subtract kp.Q * (kp.Q_d_inv * Z_d).
        const int n    = wb.nfree;
        const int nrhs = sk;
        Eigen::MatrixXd B_full = Eigen::MatrixXd::Zero(n, nrhs);
        for (int i = 0; i < n_r; ++i) B_full.row(kp.r_dofs[i]) = U.row(i);

        Eigen::MatrixXd Z = apple_solve_mat(wb.apple_factor, B_full);

        Eigen::MatrixXd Z_d((int)kp.dofs_del.size(), nrhs);
        for (size_t i = 0; i < kp.dofs_del.size(); ++i) Z_d.row((int)i) = Z.row(kp.dofs_del[i]);
        Eigen::MatrixXd Z_r(n_r, nrhs);
        for (int i = 0; i < n_r; ++i) Z_r.row(i) = Z.row(kp.r_dofs[i]);

        knew.W = Z_r - kp.Q * (kp.Q_d_inv * Z_d);

        Eigen::MatrixXd UtW(sk, sk);
        for (int i = 0; i < sk; ++i)
            for (int j = 0; j < sk; ++j)
                UtW(i, j) = knew.W(knew.s_in_r[i], j);
        knew.WSigma = knew.W * Sigma;
        Eigen::MatrixXd H = Eigen::MatrixXd::Identity(sk, sk) + UtW * Sigma;
        knew.H_inv = H.inverse();
        knew.Sigma = Sigma;
    }

    // ── 6) Map V_cand free DOFs to r-space ───────────────────────────────────
    _wba_timer.next(5);
    std::vector<int> fixed_cand_verts =
        (m_p.spring_k > 0.0) ? std::vector<int>{}
        : (m_p.fixed_left    ? left_boundary_verts(V_cand) : std::vector<int>{});
    const int ndof_cand = 2 * (int)V_cand.rows();
    std::vector<int> free_cand = free_dof_indices(ndof_cand, fixed_cand_verts);
    const int nf_cand = (int)free_cand.size();

    if (nf_cand != (int)kp.r_dofs.size())
        return std::numeric_limits<double>::infinity();

    std::vector<int> cand_free_to_r(nf_cand, -1);
    for (int k = 0; k < nf_cand; ++k) {
        const int gc     = free_cand[k];
        const int vc     = gc / 2;
        const int comp   = gc % 2;
        const int v_curr = cand_to_curr(vc);
        const int full   = 2 * v_curr + comp;
        const int fi     = wb.dof_full_to_free[full];
        if (fi < 0) return std::numeric_limits<double>::infinity();
        const int rk = kp.free_to_r[fi];
        if (rk < 0) return std::numeric_limits<double>::infinity();
        cand_free_to_r[k] = rk;
    }

    // ── 7) Barycentric prolongation P_free on V_cand ─────────────────────────
    _wba_timer.next(6);
    Eigen::SparseMatrix<double> Ps = build_barycentric_P(V_cand, F_cand);
    const int n_f    = (int)m_V_fine.rows();
    const int ndof_f = 2 * n_f;
    Eigen::SparseMatrix<double> P_free(ndof_f, nf_cand);
    {
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(Ps.nonZeros());
        for (int k = 0; k < nf_cand; ++k) {
            const int gc   = free_cand[k];
            const int vc   = gc / 2;
            const int comp = gc % 2;
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ps, vc); it; ++it) {
                const int pf = (int)it.row();
                tr.emplace_back(2*pf + comp, k, it.value());
            }
        }
        P_free.setFromTriplets(tr.begin(), tr.end());
        P_free.makeCompressed();
    }

    // Apple Sparse views: P_free (and its transposed alias), wb.K_bc.
    AppleSparseView P_view(P_free);
    SparseMatrix_Double P_T = P_view.A;
    P_T.structure.attributes.transpose = true;

    // wb.K_bc has both triangles stored; treat as ordinary for matvec.
    AppleSparseView Kbc_view(wb.K_bc);

    // ── 8) Per-mode cost ─────────────────────────────────────────────────────
    _wba_timer.next(7);
    const Eigen::VectorXd& Mfd = m_M_diag;
    double cost = 0.0;
    int    used = 0;

    Eigen::VectorXd bf(ndof_f);
    Eigen::VectorXd bc_cand(nf_cand);
    Eigen::VectorXd bc_r((int)kp.r_dofs.size());
    Eigen::VectorXd yc_cand(nf_cand);
    Eigen::VectorXd response(ndof_f), diff(ndof_f), tmp(ndof_f);

    // Section-8 sub-timing accumulators (local to keep per-mode overhead low;
    // flushed to the file-scope atomics once after the loop).
    long long s8_ns_solve = 0, s8_ns_mul = 0, s8_ns_other = 0;
    using Clock = std::chrono::high_resolution_clock;
    auto ns_since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    for (int mode_id : m_modes) {
        const double lam = m_evals[mode_id];
        if (!std::isfinite(lam) || std::abs(lam) <= m_p.eig_tol) continue;

        const double* phi = m_evecs.col(mode_id).data();

        auto t0_iter = Clock::now();

        // bf = Mfd .* phi
        vDSP_vmulD(Mfd.data(), 1, phi, 1, bf.data(), 1, (vDSP_Length)ndof_f);

        auto t_pre_mul1 = Clock::now();
        s8_ns_other += ns_since(t0_iter, t_pre_mul1);

        // bc_cand = P_free^T * bf
        SparseMultiply(P_T,
                       dvec(bf.data(), ndof_f),
                       dvec(bc_cand.data(), nf_cand));

        auto t_post_mul1 = Clock::now();
        s8_ns_mul += ns_since(t_pre_mul1, t_post_mul1);

        // Permute bc_cand → bc_r (r-space ordering)
        for (int k = 0; k < nf_cand; ++k) bc_r[cand_free_to_r[k]] = bc_cand[k];

        auto t_pre_solve = Clock::now();
        s8_ns_other += ns_since(t_post_mul1, t_pre_solve);

        // K_new^{-1} via Woodbury (Accelerate-based; CHOLMOD solve inside)
        Eigen::VectorXd yc_r =
            apply_knew_inv_accel(bc_r, wb, kp, knew, Kbc_view);

        auto t_post_solve = Clock::now();
        s8_ns_solve += ns_since(t_pre_solve, t_post_solve);

        // Permute back to V_cand ordering
        for (int k = 0; k < nf_cand; ++k) yc_cand[k] = yc_r[cand_free_to_r[k]];

        auto t_pre_mul2 = Clock::now();
        s8_ns_other += ns_since(t_post_solve, t_pre_mul2);

        // response = P_free * yc_cand
        SparseMultiply(P_view.A,
                       dvec(yc_cand.data(), nf_cand),
                       dvec(response.data(), ndof_f));

        auto t_post_mul2 = Clock::now();
        s8_ns_mul += ns_since(t_pre_mul2, t_post_mul2);

        // diff = (1/lam) * phi - response       via vDSP_vsmsbD: D = A*B - C
        const double inv_lam = 1.0 / lam;
        vDSP_vsmsbD(phi, 1, &inv_lam,
                    response.data(), 1,
                    diff.data(), 1,
                    (vDSP_Length)ndof_f);

        // tmp = Mfd .* diff
        vDSP_vmulD(Mfd.data(), 1, diff.data(), 1, tmp.data(), 1, (vDSP_Length)ndof_f);

        // norm2 = tmp · diff   (= diff^T M_f diff)
        const double norm2 = cblas_ddot(ndof_f, tmp.data(), 1, diff.data(), 1);

        double w = 1.0;
        if (m_p.weight_mode == 1)      w = 1.0 / std::abs(lam);
        else if (m_p.weight_mode == 2) w = 1.0 / (lam * lam);

        cost += w * norm2;
        ++used;

        auto t_end_iter = Clock::now();
        s8_ns_other += ns_since(t_post_mul2, t_end_iter);
    }
    g_wba_s8_sub_ns[0].fetch_add(s8_ns_solve, std::memory_order_relaxed);
    g_wba_s8_sub_ns[1].fetch_add(s8_ns_mul,   std::memory_order_relaxed);
    g_wba_s8_sub_ns[2].fetch_add(s8_ns_other, std::memory_order_relaxed);
    _wba_timer.stop();

    if (used == 0) return std::numeric_limits<double>::infinity();
    return cost;
}

// ============================================================
//  cost_approx: O(1-ring) approximate cost using local post-collapse bary
// ============================================================
// Two-stage:
//  (1) Build the local post-collapse 1-ring mesh in V_curr indexing (vi moves
//      to cpos, vj merged into vi, degenerate triangles dropped). Compute the
//      barycentric expansion of V_curr[vi] and V_curr[vj] inside this local
//      mesh, decomposing the merged-vertex weight into V_curr[vi]/V_curr[vj]
//      via ratio r along the (vi, vj) segment.
//      The decomposer reports `found`: true if the query point is strictly
//      inside some triangle, false if extrapolation was used (= nearest
//      triangle in the bary-violation sense).
//  (2) Use the bary expansion (bary_vi / bary_vj) to construct a constraint
//      vector C and compute x = K_bc^{-1} C, returning x^T M x.
//      Stage (2) is currently a placeholder; refine the C construction to
//      use bary_vi / bary_vj as desired.

double EigenEdgeCollapse::cost_approx(
    int vi, int vj,
    int vi_wmtk, int vj_wmtk,
    const Eigen::MatrixXd& V_curr,
    const Eigen::MatrixXi& F_curr,
    const std::vector<std::vector<int>>& v_faces,
    const BoundaryInfo& binfo_curr) const
{
    if (!m_wb_base) return std::numeric_limits<double>::infinity();
    const auto& wb = *m_wb_base;

    // Keep vi < vj for consistent ratio direction (cpos = (1-r) pi + r pj).
    if (vi > vj) {
        std::swap(vi, vj);
        std::swap(vi_wmtk, vj_wmtk);
    }

    // ── Stage (1): local 1-ring bary expansion ───────────────────────────

    // Collapse position
    const Eigen::Vector2d cpos = constrained_pos(vi, vj, V_curr, F_curr, binfo_curr);

    // Ratio r such that cpos = (1-r) V_curr[vi] + r V_curr[vj]
    const Eigen::Vector2d pi = V_curr.row(vi);
    const Eigen::Vector2d pj = V_curr.row(vj);
    const Eigen::Vector2d d_ij = pj - pi;
    const double d2 = std::max(d_ij.squaredNorm(), 1e-30);
    const double r = (cpos - pi).dot(d_ij) / d2;

    // Local post-collapse triangles (V_curr indices, vi → cpos, vj renamed to vi).
    std::set<int> ring_face_set;
    for (int f : v_faces[vi]) ring_face_set.insert(f);
    for (int f : v_faces[vj]) ring_face_set.insert(f);

    struct LocalTri { int v[3]; };
    std::vector<LocalTri> new_local_tris;
    new_local_tris.reserve(ring_face_set.size());
    for (int f : ring_face_set) {
        int a = F_curr(f, 0), b = F_curr(f, 1), c = F_curr(f, 2);
        const bool has_vi = (a == vi || b == vi || c == vi);
        const bool has_vj = (a == vj || b == vj || c == vj);
        if (has_vi && has_vj) continue;       // shared triangle collapses → drop
        if (a == vj) a = vi;
        if (b == vj) b = vi;
        if (c == vj) c = vi;
        new_local_tris.push_back({a, b, c});
    }

    // Vertex position in the post-collapse 1-ring (vi → cpos, others unchanged)
    auto pos = [&](int v) -> Eigen::Vector2d {
        return (v == vi) ? cpos : Eigen::Vector2d(V_curr.row(v));
    };

    // Find bary of P in new_local_tris and decompose into V_curr-indexed
    // weights, splitting the merged vertex's weight via ratio r.
    //
    //   * `found_out` = true  : P is strictly inside some triangle.
    //   * `found_out` = false : P is outside every triangle; the nearest
    //                            triangle (minimum bary violation) is used and
    //                            the resulting weights may be < 0 or > 1
    //                            (extrapolation, still sums to 1).
    //   * return value = false: all triangles were degenerate (fatal).
    auto bary_decompose = [&](
        const Eigen::Vector2d& P,
        std::vector<std::pair<int, double>>& out,
        bool& found_out) -> bool
    {
        const double eps = 1e-10;
        int best_tri = -1;
        double best_violation = std::numeric_limits<double>::infinity();
        std::array<double, 3> best_bary{};
        std::array<int, 3>    best_verts{};
        bool inside_found = false;

        for (int t = 0; t < (int)new_local_tris.size(); ++t) {
            const auto& tri = new_local_tris[t];
            const int A = tri.v[0], B = tri.v[1], C = tri.v[2];
            const Eigen::Vector2d a = pos(A);
            const Eigen::Vector2d b = pos(B);
            const Eigen::Vector2d c = pos(C);
            const double den = (b - a)[0] * (c - a)[1] - (c - a)[0] * (b - a)[1];
            if (std::abs(den) < eps) continue;
            const Eigen::Vector2d v2 = P - a;
            const double inv = 1.0 / den;
            const double u = (v2[0] * (c - a)[1] - (c - a)[0] * v2[1]) * inv;
            const double v = ((b - a)[0] * v2[1] - v2[0] * (b - a)[1]) * inv;
            const double w = 1.0 - u - v;

            // Violation: sum of "how negative" each weight is. 0 ⇒ inside.
            const double viol =
                std::max(0.0, -w) + std::max(0.0, -u) + std::max(0.0, -v);

            if (viol < best_violation) {
                best_violation = viol;
                best_tri   = t;
                best_bary  = {w, u, v};         // raw weights (no clamping)
                best_verts = {A, B, C};
            }
            if (viol <= eps) {
                inside_found = true;
                break;                          // strict containment → done
            }
        }

        if (best_tri < 0) { found_out = false; return false; }
        found_out = inside_found;

        // Decompose into V_curr-indexed weights (keep raw values for the
        // extrapolated case so sum stays = 1).
        out.clear();
        out.reserve(4);
        auto add = [&](int v_curr, double w_add) {
            for (auto& kv : out) {
                if (kv.first == v_curr) { kv.second += w_add; return; }
            }
            out.emplace_back(v_curr, w_add);
        };
        for (int k = 0; k < 3; ++k) {
            if (best_verts[k] == vi) {
                // cpos = (1-r) V_curr[vi] + r V_curr[vj]
                add(vi, best_bary[k] * (1.0 - r));
                add(vj, best_bary[k] * r);
            } else {
                add(best_verts[k], best_bary[k]);
            }
        }
        return true;
    };

    // V_curr[vi] (OLD vi position) expressed in post-collapse bary
    std::vector<std::pair<int, double>> bary_vi;
    bool found_vi = false;
    if (!bary_decompose(pi, bary_vi, found_vi))
        return std::numeric_limits<double>::infinity();

    // V_curr[vj] (OLD vj position) expressed in post-collapse bary
    std::vector<std::pair<int, double>> bary_vj;
    bool found_vj = false;
    if (!bary_decompose(pj, bary_vj, found_vj))
        return std::numeric_limits<double>::infinity();

    // ── Stage (2): C construction and cost (placeholder) ─────────────────
    // TODO: build C using bary_vi / bary_vj to get a more accurate measure.
    // For now we keep the original placeholder constraint "+1 at vi, -1 at vj".

    

    Eigen::VectorXd C_vi_x = Eigen::VectorXd::Zero(wb.nfree);
    Eigen::VectorXd C_vi_y = Eigen::VectorXd::Zero(wb.nfree);
    Eigen::VectorXd C_vj_x = Eigen::VectorXd::Zero(wb.nfree);
    Eigen::VectorXd C_vj_y = Eigen::VectorXd::Zero(wb.nfree);
    for (const auto& [v_curr, w] : bary_vi) {
        const int fx = wb.dof_full_to_free[2 * v_curr];
        const int fy = wb.dof_full_to_free[2 * v_curr + 1];
        if (fx >= 0) C_vi_x[fx] += w;
        if (fy >= 0) C_vi_y[fy] += w;
    }
    for (const auto& [v_curr, w] : bary_vj) {
        const int fx = wb.dof_full_to_free[2 * v_curr];
        const int fy = wb.dof_full_to_free[2 * v_curr + 1];
        if (fx >= 0) C_vj_x[fx] += w;
        if (fy >= 0) C_vj_y[fy] += w;
    }
    const int fx_vi = wb.dof_full_to_free[2 * vi];
    const int fy_vi = wb.dof_full_to_free[2 * vi + 1];
    const int fx_vj = wb.dof_full_to_free[2 * vj];
    const int fy_vj = wb.dof_full_to_free[2 * vj + 1];
    if (fx_vi >= 0) C_vi_x[fx_vi] -= 1.0;
    if (fy_vi >= 0) C_vi_y[fy_vi] -= 1.0;
    if (fx_vj >= 0) C_vj_x[fx_vj] -= 1.0;
    if (fy_vj >= 0) C_vj_y[fy_vj] -= 1.0;

    if (C_vi_x.isZero() && C_vj_x.isZero())
        return std::numeric_limits<double>::infinity();

    Eigen::VectorXd M_diag(wb.nfree);
    for (int k = 0; k < wb.nfree; ++k)
        M_diag[k] = wb.M_diag[wb.free_dofs[k]];

    Eigen::VectorXd x_i = cholmod_solve_vec(wb.factor.L, C_vi_x);
    Eigen::VectorXd y_i = cholmod_solve_vec(wb.factor.L, C_vi_y);
    Eigen::VectorXd x_j = cholmod_solve_vec(wb.factor.L, C_vj_x);
    Eigen::VectorXd y_j = cholmod_solve_vec(wb.factor.L, C_vj_y);
    Eigen::VectorXd Mx_i = M_diag.cwiseProduct(x_i);
    Eigen::VectorXd My_i = M_diag.cwiseProduct(y_i);
    Eigen::VectorXd Mx_j = M_diag.cwiseProduct(x_j);
    Eigen::VectorXd My_j = M_diag.cwiseProduct(y_j);
    return x_i.dot(Mx_i) + y_i.dot(My_i) + x_j.dot(Mx_j) + y_j.dot(My_j);
}

// ============================================================
//  Extract current TriMesh as V, F arrays
// ============================================================

void EigenEdgeCollapse::extract_current_mesh(
    Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
{
    const int vcap = (int)vert_capacity();
    const int fcap = (int)tri_capacity();

    // Build compact index (some vertices may be removed)
    std::vector<int> old_to_new(vcap, -1);
    int cnt = 0;
    for (const auto& t : get_vertices())
        old_to_new[t.vid(*this)] = cnt++;

    V.resize(cnt, 2);
    for (const auto& t : get_vertices()) {
        int vid = (int)t.vid(*this);
        V.row(old_to_new[vid]) = vertex_attrs[vid].pos;
    }

    std::vector<Eigen::Vector3i> faces;
    faces.reserve(get_faces().size());
    for (const auto& t : get_faces()) {
        auto vs = oriented_tri_vertices(t);
        Eigen::Vector3i tri;
        for (int j = 0; j < 3; ++j)
            tri[j] = old_to_new[(int)vs[j].vid(*this)];
        faces.push_back(tri);
    }
    F.resize((int)faces.size(), 3);
    for (int i = 0; i < (int)faces.size(); ++i)
        F.row(i) = faces[i];
}

// ============================================================
//  Woodbury base: snapshot current mesh K, factorize with CHOLMOD
// ============================================================

void EigenEdgeCollapse::rebuild_woodbury_base()
{
    if (!m_wb_base) m_wb_base = std::make_unique<WoodburyBase>();
    auto& wb = *m_wb_base;

    // Snapshot current coarse mesh
    extract_current_mesh(wb.V, wb.F);

    // Assemble K and lumped mass on V/F (full, ndof × ndof)
    assemble_fem(wb.V, wb.F, wb.K, wb.M_diag);
    apply_spring_bcs(wb.K, wb.V, wb.F);
    wb.ndof = (int)wb.K.rows();

    // Free-DOF map (after fixed_left BC; empty fixed_verts when spring_k > 0)
    std::vector<int> fixed_verts =
        (m_p.spring_k > 0.0) ? std::vector<int>{}
        : (m_p.fixed_left    ? left_boundary_verts(wb.V) : std::vector<int>{});
    wb.free_dofs = free_dof_indices(wb.ndof, fixed_verts);
    wb.nfree     = (int)wb.free_dofs.size();
    wb.dof_full_to_free.assign(wb.ndof, -1);
    for (int k = 0; k < wb.nfree; ++k) wb.dof_full_to_free[wb.free_dofs[k]] = k;

    if (wb.nfree == 0) {
        wb.factor = CholFactor{};
        wb.apple_factor = AppleSparseCholFactor{};
        return;
    }

    // Build K_bc as submatrix of K at free DOFs (sparse, symmetric)
    std::vector<Eigen::Triplet<double>> tr;
    tr.reserve(wb.K.nonZeros());
    for (int j = 0; j < wb.nfree; ++j) {
        const int col = wb.free_dofs[j];
        for (Eigen::SparseMatrix<double>::InnerIterator it(wb.K, col); it; ++it) {
            const int r  = (int)it.row();
            const int rf = wb.dof_full_to_free[r];
            if (rf >= 0) tr.emplace_back(rf, j, it.value());
        }
    }
    wb.K_bc.resize(wb.nfree, wb.nfree);
    wb.K_bc.setFromTriplets(tr.begin(), tr.end());

    // Kinetic shift K_eff = K + alpha M  (matches candidate_cost when no fixed BC)
    if (m_p.spring_k <= 0.0 && !m_p.fixed_left && m_p.alpha > 0.0)
        for (int k = 0; k < wb.nfree; ++k)
            wb.K_bc.coeffRef(k, k) += m_p.alpha * wb.M_diag[wb.free_dofs[k]];

    wb.K_bc.makeCompressed();
    auto time_cholmod_start = std::chrono::high_resolution_clock::now();
    wb.factor = cholmod_factorize_spd(wb.K_bc);
    auto time_cholmod_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_cholmod = time_cholmod_end - time_cholmod_start;
    std::cout << "CHOLMOD factorization time: " << time_cholmod.count() << " seconds\n";
    if (!wb.factor.L)
        std::cerr << "rebuild_woodbury_base: CHOLMOD factorisation failed\n";

    // Parallel Apple Accelerate Cholesky factor for the *_accelerated cost path.
    // SparseFactor reads the upper-triangle of K_bc; the factor is independent
    // of the input matrix afterwards, so the temporary upper view can die.
    {
        Eigen::SparseMatrix<double> K_bc_upper = wb.K_bc.triangularView<Eigen::Upper>();
        K_bc_upper.makeCompressed();
        auto time_apple_start = std::chrono::high_resolution_clock::now();
        wb.apple_factor = apple_factorize_spd_upper(K_bc_upper);
        auto time_apple_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_apple = time_apple_end - time_apple_start;
        std::cout << "Apple Accelerate factorization time: " << time_apple.count() << " seconds\n";
        if (!wb.apple_factor.valid)
            std::cerr << "rebuild_woodbury_base: Apple Cholesky factorisation failed\n";
    }
}

// ============================================================
//  Initialise from OBJ: load, assemble fine FEM, eigenmodes
// ============================================================

void EigenEdgeCollapse::init_from_obj(
    const std::string& path, const CollapseParams& p)
{
    m_p = p;

    // ── Optional material-region OBJ ──────────────────────────────────────
    if (!p.material_obj.empty()) {
        Eigen::MatrixXd Vm3;
        Eigen::MatrixXi Fm;
        if (!igl::read_triangle_mesh(p.material_obj, Vm3, Fm))
            throw std::runtime_error("Could not read material OBJ: " + p.material_obj);
        m_material_verts = Vm3.leftCols(2);
        m_material_faces = Fm;
        std::cout << "Loaded material region: " << m_material_verts.rows()
                  << " vertices, " << m_material_faces.rows() << " triangles\n";
    }

    // Load mesh via igl (stores z=0 for 2D OBJ)
    Eigen::MatrixXd V3;
    Eigen::MatrixXi F;
    if (!igl::read_triangle_mesh(path, V3, F))
        throw std::runtime_error("Could not read mesh: " + path);

    m_V_fine = V3.leftCols(2);
    m_F_fine = F;

    std::cout << "Loaded fine mesh: " << m_V_fine.rows()
              << " vertices, " << m_F_fine.rows() << " triangles\n";

    // ── Optional: load user-specified fine vertex indices for springs ─────
    // If m_spring_fine_verts is already populated (e.g. via set_spring_fine_verts
    // from an interactive picker), keep it; the file path is ignored in that
    // case. Otherwise load from the file when given.
    if (m_spring_fine_verts.empty() && !p.spring_fine_verts_file.empty()) {
        std::ifstream ifs(p.spring_fine_verts_file);
        if (!ifs)
            throw std::runtime_error("Could not open spring_fine_verts file: "
                                     + p.spring_fine_verts_file);
        int idx;
        while (ifs >> idx) {
            if (idx < 0 || idx >= (int)m_V_fine.rows()) {
                std::cerr << "warning: spring_fine_verts index out of range: "
                          << idx << " (n_fine=" << m_V_fine.rows() << "), skipping\n";
                continue;
            }
            m_spring_fine_verts.push_back(idx);
        }
        std::cout << "Loaded " << m_spring_fine_verts.size()
                  << " spring fine-vertex indices from "
                  << p.spring_fine_verts_file << "\n";
    } else if (!m_spring_fine_verts.empty()) {
        // Validate pre-set indices against the just-loaded fine mesh.
        std::vector<int> kept;
        kept.reserve(m_spring_fine_verts.size());
        for (int idx : m_spring_fine_verts) {
            if (idx >= 0 && idx < (int)m_V_fine.rows()) kept.push_back(idx);
            else std::cerr << "warning: preset spring vertex " << idx
                           << " out of range, skipping\n";
        }
        m_spring_fine_verts = std::move(kept);
        std::cout << "Using " << m_spring_fine_verts.size()
                  << " preset spring fine-vertex indices\n";
    }

    // ── Per-element E and nu on fine mesh (ground-truth material distribution) ──
    // Computed from centroid sampling; used by material_{E,nu}_from_fine to
    // project material onto any coarser mesh via area-weighted averaging.
    const int nf_tris = (int)m_F_fine.rows();
    m_fine_E.resize(nf_tris);
    m_fine_nu.resize(nf_tris);
    m_fine_bb_min.resize(nf_tris, 2);
    m_fine_bb_max.resize(nf_tris, 2);
    for (int f = 0; f < nf_tris; ++f) {
        m_fine_E[f]  = material_E_analytical(m_V_fine, m_F_fine.row(f));
        m_fine_nu[f] = material_nu_analytical(m_V_fine, m_F_fine.row(f));
        for (int d = 0; d < 2; ++d) {
            double lo = std::min({m_V_fine(m_F_fine(f,0), d),
                                  m_V_fine(m_F_fine(f,1), d),
                                  m_V_fine(m_F_fine(f,2), d)});
            double hi = std::max({m_V_fine(m_F_fine(f,0), d),
                                  m_V_fine(m_F_fine(f,1), d),
                                  m_V_fine(m_F_fine(f,2), d)});
            m_fine_bb_min(f, d) = lo;
            m_fine_bb_max(f, d) = hi;
        }
    }

    // ── Fine FEM (material_E dispatch now uses m_fine_E) ──────────────────
    Eigen::SparseMatrix<double> K_fine;
    assemble_fem(m_V_fine, m_F_fine, K_fine, m_M_diag);

    // Mass matrix (diagonal sparse)
    const int ndof_f = (int)K_fine.rows();
    m_M_fine.resize(ndof_f, ndof_f);
    {
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(ndof_f);
        for (int i = 0; i < ndof_f; ++i)
            tr.emplace_back(i, i, m_M_diag[i]);
        m_M_fine.setFromTriplets(tr.begin(), tr.end());
    }

    // ── Apply BCs ──────────────────────────────────────────────────────────
    // spring_k > 0: modify K in place; keep all DOFs free.
    // Otherwise:    remove fixed DOFs from the system (fixed_left) and/or
    //               apply kinetic shift (alpha) for regularisation.
    apply_spring_bcs(K_fine, m_V_fine, m_F_fine);

    std::vector<int> fixed_verts =
        (p.spring_k > 0.0) ? std::vector<int>{}
        : (p.fixed_left    ? left_boundary_verts(m_V_fine) : std::vector<int>{});
    m_free_fine = free_dof_indices(ndof_f, fixed_verts);

    const int nf = (int)m_free_fine.size();

    // Extract K_ff and M_ff as dense for eigensolver
    Eigen::MatrixXd Kff_d(nf, nf), Mff_d(nf, nf);
    {
        // index map
        std::vector<int> idx(ndof_f, -1);
        for (int k = 0; k < nf; ++k) idx[m_free_fine[k]] = k;

        Kff_d.setZero();
        for (int j = 0; j < nf; ++j) {
            int col = m_free_fine[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(K_fine, col); it; ++it) {
                int ri = (int)it.row();
                if (idx[ri] >= 0) Kff_d(idx[ri], j) = it.value();
            }
        }

        Mff_d.setZero();
        for (int k = 0; k < nf; ++k)
            Mff_d(k, k) = m_M_diag[m_free_fine[k]];
    }

    // ── Kinetic shift K_eff = K + alpha*M (only when no fixed BCs) ────────
    // Regularises the singular rigid-body modes so the eigenproblem is
    // well-posed.  Eigenvalues become lambda_s = lambda + alpha.
    // Skipped when spring_k > 0 (springs already regularise the system).
    if (p.spring_k <= 0.0 && !p.fixed_left && p.alpha > 0.0)
        for (int k = 0; k < nf; ++k)
            Kff_d(k, k) += p.alpha * Mff_d(k, k);

    // ── Sparse K_fine_ff (free DOFs, with kinetic shift) + LDLT ───────────
    // Reused inside candidate_cost for the K_f^{-1} norm term.
    {
        std::vector<int> idx(ndof_f, -1);
        for (int k = 0; k < nf; ++k) idx[m_free_fine[k]] = k;
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(K_fine.nonZeros());
        for (int j = 0; j < nf; ++j) {
            int col = m_free_fine[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(K_fine, col); it; ++it) {
                int ri = (int)it.row();
                if (idx[ri] >= 0) tr.emplace_back(idx[ri], j, it.value());
            }
        }
        m_K_fine_ff.resize(nf, nf);
        m_K_fine_ff.setFromTriplets(tr.begin(), tr.end());
        if (p.spring_k <= 0.0 && !p.fixed_left && p.alpha > 0.0)
            for (int k = 0; k < nf; ++k)
                m_K_fine_ff.coeffRef(k, k) += p.alpha * m_M_diag[m_free_fine[k]];
        m_K_fine_ff.makeCompressed();

        // ── Diagnostic: identify why LDLT might fail ───────────────────────
        {
            const int nrows = (int)m_K_fine_ff.rows();
            int n_zero_diag = 0, n_neg_diag = 0, n_small_diag = 0;
            double diag_min = std::numeric_limits<double>::infinity();
            double diag_max = -std::numeric_limits<double>::infinity();
            double diag_abs_min = std::numeric_limits<double>::infinity();
            for (int k = 0; k < nrows; ++k) {
                const double d = m_K_fine_ff.coeff(k, k);
                if (d == 0.0)            ++n_zero_diag;
                else if (d < 0.0)        ++n_neg_diag;
                else if (d < 1e-12)      ++n_small_diag;
                diag_min = std::min(diag_min, d);
                diag_max = std::max(diag_max, d);
                diag_abs_min = std::min(diag_abs_min, std::abs(d));
            }
            // Orphan vertices: those with M_diag == 0 (no incident triangle area)
            int n_orphan = 0;
            std::vector<int> orphan_examples;
            for (int v = 0; v < (int)m_V_fine.rows(); ++v) {
                if (m_M_diag[2*v] == 0.0 && m_M_diag[2*v+1] == 0.0) {
                    if ((int)orphan_examples.size() < 10) orphan_examples.push_back(v);
                    ++n_orphan;
                }
            }
            // Degenerate triangles
            int n_degen_tri = 0;
            for (int f = 0; f < (int)m_F_fine.rows(); ++f) {
                const Eigen::Vector2d a = m_V_fine.row(m_F_fine(f,0));
                const Eigen::Vector2d b = m_V_fine.row(m_F_fine(f,1));
                const Eigen::Vector2d c = m_V_fine.row(m_F_fine(f,2));
                const double area2 = std::abs((b.x()-a.x())*(c.y()-a.y())
                                            - (c.x()-a.x())*(b.y()-a.y()));
                if (area2 < 1e-20) ++n_degen_tri;
            }
            std::cout << "[K_fine_ff diag] n=" << nrows
                      << "  diag_min=" << diag_min
                      << "  diag_max=" << diag_max
                      << "  |diag|_min=" << diag_abs_min << "\n";
            std::cout << "[K_fine_ff diag] zero_diag=" << n_zero_diag
                      << "  neg_diag=" << n_neg_diag
                      << "  small_diag(<1e-12)=" << n_small_diag << "\n";
            std::cout << "[K_fine_ff diag] orphan_verts(M_diag==0)=" << n_orphan
                      << "  degen_tris=" << n_degen_tri
                      << "  total_verts=" << m_V_fine.rows()
                      << "  total_tris=" << m_F_fine.rows() << "\n";
            if (!orphan_examples.empty()) {
                std::cout << "[K_fine_ff diag] first orphan vertex IDs:";
                for (int v : orphan_examples) std::cout << " " << v;
                std::cout << "\n";
            }
        }

        m_K_fine_ff_solver.compute(m_K_fine_ff);
        if (m_K_fine_ff_solver.info() != Eigen::Success) {
            std::cerr << "[K_fine_ff LDLT] info()=" << (int)m_K_fine_ff_solver.info()
                      << " (1=NumericalIssue, 2=NoConvergence, 3=InvalidInput)\n";
            throw std::runtime_error("K_fine_ff LDLT factorization failed");
        }
    }

    // ── Generalized eigenproblem: K_eff u = λ_s M u ───────────────────────
    const int k_eig = std::min(p.num_modes, nf);
    std::cout << "Computing " << k_eig << " eigenmodes on "
              << nf << " free DOFs...\n";

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigs(Kff_d, Mff_d);
    if (eigs.info() != Eigen::Success)
        throw std::runtime_error("Eigenvalue solve failed");

    // Take first k_eig modes (smallest eigenvalues)
    m_evals = eigs.eigenvalues().head(k_eig);
    Eigen::MatrixXd evecs_free = eigs.eigenvectors().leftCols(k_eig); // nf × k_eig

    // Expand to all DOFs (zero on fixed DOFs)
    m_evecs = Eigen::MatrixXd::Zero(ndof_f, k_eig);
    for (int k = 0; k < nf; ++k)
        m_evecs.row(m_free_fine[k]) = evecs_free.row(k);

    // ── Mode selection ─────────────────────────────────────────────────────
    m_modes.clear();
    for (int i = 0; i < k_eig && (int)m_modes.size() < p.cost_modes; ++i)
        if (std::abs(m_evals[i]) > p.eig_tol)
            m_modes.push_back(i);

    std::cout << "Using " << m_modes.size() << " modes for cost\n";

    // ── Eigenvalue diagnostics ─────────────────────────────────────────────
    {
        // Smallest eigenvalue (regardless of tolerance), and the cost_modes
        // eigenvalues actually used.
        double abs_min = std::numeric_limits<double>::infinity();
        int    abs_min_idx = -1;
        for (int i = 0; i < k_eig; ++i) {
            const double a = std::abs(m_evals[i]);
            if (a < abs_min) { abs_min = a; abs_min_idx = i; }
        }
        std::cout << "[eigvals] k_eig=" << k_eig
                  << "  abs_min=" << std::scientific << std::setprecision(4) << abs_min
                  << "  at idx " << abs_min_idx
                  << "  (signed=" << (abs_min_idx >= 0 ? m_evals[abs_min_idx] : 0.0) << ")"
                  << "  eig_tol=" << m_p.eig_tol << "\n";

        std::cout << "[eigvals] cost_modes used: index lambda (1/lambda)\n";
        for (int mid : m_modes) {
            const double lam = m_evals[mid];
            std::cout << "  mode[" << std::setw(3) << mid << "] lambda="
                      << std::scientific << std::setprecision(6) << lam
                      << "  1/lambda=" << (lam != 0.0 ? 1.0 / lam : 0.0)
                      << "\n";
        }
        std::cout << std::defaultfloat;
    }

    // ── Init TriMesh topology ──────────────────────────────────────────────
    const int n = (int)m_V_fine.rows();
    vertex_attrs.resize(n);

    std::vector<std::array<size_t, 3>> tris((size_t)m_F_fine.rows());
    for (int f = 0; f < (int)m_F_fine.rows(); ++f)
        tris[f] = {(size_t)m_F_fine(f, 0),
                   (size_t)m_F_fine(f, 1),
                   (size_t)m_F_fine(f, 2)};

    wmtk::TriMesh::init((size_t)n, tris);

    const auto binfo = classify_boundary(m_V_fine, m_F_fine);
    m_fine_boundary_verts.clear();
    for (int i = 0; i < n; ++i) {
        vertex_attrs[i].pos    = m_V_fine.row(i);
        vertex_attrs[i].freeze = binfo.on_corner[i];
        if (binfo.on_boundary[i])
            m_fine_boundary_verts.push_back(i);
    }
    std::cout << "Fine mesh boundary vertices: " << m_fine_boundary_verts.size() << "\n";
}

// ============================================================
//  TriMesh callbacks
// ============================================================

bool EigenEdgeCollapse::collapse_edge_before(const Tuple& t)
{
    if (!wmtk::TriMesh::collapse_edge_before(t)) return false;

    const int vi = (int)t.vid(*this);
    const int vj = (int)t.switch_vertex(*this).vid(*this);

    // Reject only if BOTH endpoints are corners; one-corner collapses are allowed
    // (the surviving vertex will be promoted to corner in collapse_edge_after).
    if (vertex_attrs[vi].freeze && vertex_attrs[vj].freeze) return false;

    m_ccache.v1pos     = vertex_attrs[vi].pos;
    m_ccache.v2pos     = vertex_attrs[vj].pos;
    m_ccache.v1_frozen = vertex_attrs[vi].freeze;
    m_ccache.v2_frozen = vertex_attrs[vj].freeze;
    return true;
}

bool EigenEdgeCollapse::collapse_edge_after(const Tuple& t)
{
    const int vid = (int)t.vid(*this);
    vertex_attrs[vid].pos = m_pending_pos;

    // Mirror interior_tet_opt's is_inverted check: inspect the entire one-ring
    // of the surviving vertex.  A negative (or zero) signed area means the
    // collapse created a face flip → return false so wmtk rolls back both
    // the topology (collapse_edge_rollback) and attributes (rollback_protected_attributes).
    for (const auto& tri_t : get_one_ring_tris_for_vertex(t)) {
        const auto vs = oriented_tri_vertices(tri_t);
        const Eigen::Vector2d& p0 = vertex_attrs[vs[0].vid(*this)].pos;
        const Eigen::Vector2d& p1 = vertex_attrs[vs[1].vid(*this)].pos;
        const Eigen::Vector2d& p2 = vertex_attrs[vs[2].vid(*this)].pos;
        const double area2 =
            (p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if (area2 <= 1e-14) return false; // face flip → rollback
    }
    // If either original endpoint was a corner, the survivor inherits corner status.
    if (m_ccache.v1_frozen || m_ccache.v2_frozen)
        vertex_attrs[vid].freeze = true;
    return true;
}

// ============================================================
//  Main simplification loop (brute-force, sequential)
// ============================================================

int EigenEdgeCollapse::simplify(int target_vertices)
{
    int collapses = 0;

    while ((int)get_vertices().size() > target_vertices) {
        // measure time per collapse step
        auto start_time = std::chrono::high_resolution_clock::now();
        // Extract current mesh arrays
        Eigen::MatrixXd V_curr;
        Eigen::MatrixXi F_curr;
        extract_current_mesh(V_curr, F_curr);

        auto after_extract_time = std::chrono::high_resolution_clock::now();
        double extract_time = std::chrono::duration<double>(after_extract_time - start_time).count();
        std::cout << "Extracted current mesh in " << extract_time << " seconds.\n";

        // Precompute which fine boundary vertices lie in each coarse face.
        // Used by enclosure_pos() for boundary edge collapse positioning.
        rebuild_coarse_fine_overlap(V_curr, F_curr);

        auto after_overlap_time = std::chrono::high_resolution_clock::now();
        double overlap_time = std::chrono::duration<double>(after_overlap_time - after_extract_time).count();
        std::cout << "Rebuilt coarse-fine overlap in " << overlap_time << " seconds.\n";

        // K_base + CHOLMOD factorisation for Woodbury-based candidate_cost.
        // Re-snapshot of V_curr/F_curr at the start of every step (after a collapse,
        // the previous factor is stale).
        rebuild_woodbury_base();

        auto after_wb_time = std::chrono::high_resolution_clock::now();
        double wb_time = std::chrono::duration<double>(after_wb_time - after_overlap_time).count();
        std::cout << "Rebuilt Woodbury base (CHOLMOD factor) in " << wb_time << " seconds.\n";

        std::cout << "Step " << collapses
                  << ": vertices=" << V_curr.rows()
                  << " faces=" << F_curr.rows()
                  << " target=" << target_vertices << "\n";

        // Map from compact index back to wmtk vertex IDs
        std::vector<int> compact_to_wmtk;
        {
            compact_to_wmtk.reserve(V_curr.rows());
            std::vector<int> old_to_new((int)vert_capacity(), -1);
            int cnt = 0;
            for (const auto& t : get_vertices()) {
                int vid = (int)t.vid(*this);
                old_to_new[vid] = cnt;
                compact_to_wmtk.push_back(vid);
                ++cnt;
            }
        }

        // Evaluate all edges
        auto edges = get_edges();
        double best_cost = std::numeric_limits<double>::infinity();
        int    best_vi   = -1, best_vj = -1;
        Tuple  best_tuple;
        Eigen::Vector2d best_pos;

        // Build compact index for wmtk vid → compact id
        std::vector<int> wmtk_to_compact((int)vert_capacity(), -1);
        for (int k = 0; k < (int)compact_to_wmtk.size(); ++k)
            wmtk_to_compact[compact_to_wmtk[k]] = k;

        // Precompute boundary info once per step
        const auto binfo_curr = classify_boundary(V_curr, F_curr);

        // Build a sorted candidate list: (cost, tuple, compact_vi, compact_vj, pos)
        struct Candidate {
            double          cost;
            Tuple           tuple;
            int             vi, vj;
            Eigen::Vector2d pos;
            double ratio;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(edges.size());


        std::vector<double> face_E(F_curr.rows());
        for (int f = 0; f < (int)F_curr.rows(); ++f)
            face_E[f] = material_E(V_curr, F_curr.row(f));

        std::vector<std::vector<int>> v_faces(V_curr.rows());
        for (int f = 0; f < (int)F_curr.rows(); ++f)
            for (int k = 0; k < 3; ++k)
                v_faces[F_curr(f,k)].push_back(f);

        auto before_cost_time = std::chrono::high_resolution_clock::now();
        double before_cost_time_sec = std::chrono::duration<double>(before_cost_time - after_overlap_time).count();
        std::cout << "Prepared for candidate evaluation in " << before_cost_time_sec << " seconds.\n";
        int count_evaluated = 0;

        // sample 20 random edges if too many
        if (edges.size() > 100) {
            std::shuffle(edges.begin(), edges.end(), std::mt19937{std::random_device{}()});
            edges.resize(100);
        }
        // set thread count for OpenMP

        omp_set_num_threads(std::max(1, omp_get_max_threads())); // leave one thread free for main thread and other tasks
        // omp_set_num_threads(1); // for testing, set to 4 threads


        #pragma omp parallel for
        for (const auto& et : edges) {
            const int vi_wmtk = (int)et.vid(*this);
            const int vj_wmtk = (int)et.switch_vertex(*this).vid(*this);

            const int vi = wmtk_to_compact[vi_wmtk];
            const int vj = wmtk_to_compact[vj_wmtk];

            // bool all_large = true;
            // for (int f : v_faces[vi]) if (face_E[f] <= 50.0) { all_large = false; break; }
            // if (all_large)
            //     for (int f : v_faces[vj]) if (face_E[f] <= 50.0) { all_large = false; break; }
            
            #pragma omp atomic
            count_evaluated++;
            // Reject only corner–corner edges; one-corner edges are allowed
            if (vertex_attrs[vi_wmtk].freeze && vertex_attrs[vj_wmtk].freeze) continue;

            // Skip edges that fail wmtk's link condition (topological constraint)
            if (!check_link_condition(et)) continue;

            // Reject boundary–boundary collapses that cross the interior
            // (both endpoints on boundary but edge is not a boundary edge).
            if (binfo_curr.on_boundary[vi] && binfo_curr.on_boundary[vj]) {
                auto key = std::make_pair(std::min(vi, vj), std::max(vi, vj));
                if (!binfo_curr.boundary_edges.count(key)) continue;
            }

            Eigen::MatrixXd V_cand;
            Eigen::MatrixXi F_cand;
            if (!collapse_edge_static(V_curr, F_curr, vi, vj, V_cand, F_cand))
                continue;

            // double cost = cost_approx(vi, vj, vi_wmtk, vj_wmtk,
            //                           V_curr, F_curr, v_faces, binfo_curr);
            // if (!std::isfinite(cost)) continue;

            // measure time for candidate_cost
            auto start_candidate_cost = std::chrono::high_resolution_clock::now();
            // double cost = candidate_cost(V_cand, F_cand);

            // Woodbury cost: reuses the precomputed CHOLMOD factor of K_base.
            // Falls back to full assembly+factorisation when Woodbury invariants
            // don't hold (e.g. vi's boundary status changes after the collapse).
            // double cost = candidate_cost_woodbury(V_cand, F_cand, vi, vj);
            double cost = candidate_cost_woodbury_accelerated(V_cand, F_cand, vi, vj);
            // double cost = candidate_cost_perturbation(V_cand, F_cand, vi, vj);
            if (!std::isfinite(cost)){
                std::cout << "Woodbury cost failed for edge (" << vi << "," << vj << "), falling back to full cost\n";
                cost = candidate_cost(V_cand, F_cand);
            }

            // ── COST_CMP=N: print accelerated vs full cost for first N candidates
            {
                static std::atomic<int> cmp_remaining{[]{
                    const char* e = std::getenv("COST_CMP");
                    return (e && std::atoi(e) > 0) ? std::atoi(e) : 0;
                }()};
                int prev = cmp_remaining.fetch_sub(1, std::memory_order_relaxed);
                if (prev > 0 && std::isfinite(cost)) {
                    const double cf = candidate_cost(V_cand, F_cand);
                    const double abs_d = std::abs(cost - cf);
                    const double rel_d = (std::abs(cf) > 1e-300) ? abs_d / std::abs(cf) : 0.0;
                    #pragma omp critical
                    std::cout << "[COST_CMP] edge=(" << vi << "," << vj << ")"
                              << "  wood_accel=" << std::scientific << std::setprecision(8) << cost
                              << "  full=" << cf
                              << "  abs_diff=" << abs_d
                              << "  rel_diff=" << rel_d << "\n";
                }
            }
            // double cost_candidate = candidate_cost(V_cand, F_cand);
            // std::cout << "relative difference between Woodbury and full cost: " << (std::isfinite(cost) && std::isfinite(cost_candidate) ? std::abs(cost - cost_candidate)  : 0.0) << "\n";
            // std::cout << "  Edge (" << vi << "," << vj << ") cost=" << cost
            //           << " (full cost=" << cost_candidate << ")\n";
            // if (!std::isfinite(cost)) continue;
            auto end_candidate_cost = std::chrono::high_resolution_clock::now();
            double candidate_cost_time_sec = std::chrono::duration<double>(end_candidate_cost - start_candidate_cost).count();
            // std::cout << "  Evaluated edge (" << vi << "," << vj << ") cost=" << cost << " in " << candidate_cost_time_sec << " seconds.\n";

            Eigen::Vector2d cpos = constrained_pos(
                std::min(vi, vj), std::max(vi, vj), V_curr, F_curr, binfo_curr);

            // cpos == vi then ratio = 0; cpos == vj then ratio = 1 or midpoint then ratio = 0.5
            Eigen::Vector2d vi_pos = V_curr.row(vi);
            Eigen::Vector2d vj_pos = V_curr.row(vj);
            double ratio = (cpos - vi_pos).norm() / (vj_pos - vi_pos).norm();

            #pragma omp critical
            candidates.push_back({cost, et, vi, vj, cpos, ratio});
        }
        auto after_cost_time = std::chrono::high_resolution_clock::now();
        double cost_time_sec = std::chrono::duration<double>(after_cost_time - before_cost_time).count();
        std::cout << "Evaluated " << candidates.size() << " candidates in " << cost_time_sec << " seconds.\n";

        // Per-section breakdown for candidate_cost_woodbury_accelerated
        // (totals across all edges and threads in this step).
        {
            static const char* labels[8] = {
                "1) dofs_del               ",
                "2) old/new triangles      ",
                "3) s_dofs                 ",
                "4) build Sigma            ",
                "5) Woodbury precompute    ",
                "6) V_cand -> r-space map  ",
                "7) barycentric P_free     ",
                "8) per-mode cost          ",
            };
            double total = 0.0;
            for (int i = 0; i < 8; ++i)
                total += g_wba_section_ns[i].load(std::memory_order_relaxed) * 1e-9;
            std::cout << "  candidate_cost_woodbury_accelerated section totals (sec):\n";
            for (int i = 0; i < 8; ++i) {
                double sec = g_wba_section_ns[i].load(std::memory_order_relaxed) * 1e-9;
                double pct = (total > 0.0) ? 100.0 * sec / total : 0.0;
                std::cout << "    " << labels[i] << " " << sec
                          << "  (" << pct << "%)\n";
                g_wba_section_ns[i].store(0, std::memory_order_relaxed);
            }
            std::cout << "    sum                       " << total << "\n";

            // Section 8 sub-breakdown
            static const char* s8_labels[3] = {
                "8a) solve (apply_knew_inv) ",
                "8b) SparseMultiply         ",
                "8c) other (vDSP/permute)   ",
            };
            double s8_total = 0.0;
            for (int i = 0; i < 3; ++i)
                s8_total += g_wba_s8_sub_ns[i].load(std::memory_order_relaxed) * 1e-9;
            std::cout << "  section 8 sub-breakdown (sec):\n";
            for (int i = 0; i < 3; ++i) {
                double sec = g_wba_s8_sub_ns[i].load(std::memory_order_relaxed) * 1e-9;
                double pct = (s8_total > 0.0) ? 100.0 * sec / s8_total : 0.0;
                std::cout << "    " << s8_labels[i] << " " << sec
                          << "  (" << pct << "%)\n";
                g_wba_s8_sub_ns[i].store(0, std::memory_order_relaxed);
            }
            std::cout << "    sum                       " << s8_total << "\n";

            // apply_knew_inv_accel internal breakdown
            static const char* inv_labels[9] = {
                "a) apple_solve_vec (n×n)   ",
                "b) krr: gemv Q_d_inv       ",
                "c) krr: gemv Q (n_r×nd)    ",
                "d) krr: embed/extract      ",
                "e) wb: gemv H_inv (s×s)    ",
                "f) wb: gemv WΣ (n_r×s)     ",
                "g) knew_mat: SparseMultiply",
                "h) knew_mat: gemv Σ (s×s)  ",
                "i) refine: vsub/dnrm2/daxpy",
            };
            double inv_total = 0.0;
            for (int i = 0; i < 9; ++i)
                inv_total += g_wba_inv_ns[i].load(std::memory_order_relaxed) * 1e-9;
            const long long calls   = g_wba_inv_calls.load(std::memory_order_relaxed);
            const long long refines = g_wba_inv_refine_solves.load(std::memory_order_relaxed);
            std::cout << "  apply_knew_inv_accel internal breakdown (sec):\n"
                      << "    calls=" << calls
                      << "  refine_solves=" << refines
                      << "  (avg refine/call=" << (calls ? double(refines) / double(calls) : 0.0)
                      << ")\n";
            for (int i = 0; i < 9; ++i) {
                const long long n_calls = g_wba_inv_count[i].load(std::memory_order_relaxed);
                double sec = g_wba_inv_ns[i].load(std::memory_order_relaxed) * 1e-9;
                double pct = (inv_total > 0.0) ? 100.0 * sec / inv_total : 0.0;
                std::cout << "    " << inv_labels[i] << " " << sec
                          << "  (" << pct << "%, " << n_calls << " calls)\n";
                g_wba_inv_ns[i].store(0, std::memory_order_relaxed);
                g_wba_inv_count[i].store(0, std::memory_order_relaxed);
            }
            std::cout << "    sum                       " << inv_total << "\n";
            g_wba_inv_calls.store(0, std::memory_order_relaxed);
            g_wba_inv_refine_solves.store(0, std::memory_order_relaxed);
        }

        if (candidates.empty()) {
            std::cout << "No collapsible edges remain.\n";
            break;
        }

        // Sort by cost ascending
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

        // Save edge-cost PNG for this step (if output dir is set)
        if (!m_output_dir.empty()) {
            std::vector<std::pair<int,int>> edge_pairs;
            std::vector<double> edge_costs;
            edge_pairs.reserve(candidates.size());
            edge_costs.reserve(candidates.size());
            for (const auto& cand : candidates) {
                edge_pairs.push_back({cand.vi, cand.vj});
                edge_costs.push_back(cand.cost);
            }
            
            save_edge_cost_png(collapses, V_curr, F_curr, edge_pairs, edge_costs, 0);
        }

        // Try candidates in order until one succeeds
        bool collapsed = false;
        for (auto& cand : candidates) {
            std::cout << "  try edge (" << cand.vi << "," << cand.vj
                      << ") cost=" << cand.cost << "\n";

            m_pending_pos = cand.pos;
            std::vector<Tuple> new_tris;
            if (collapse_edge(cand.tuple, new_tris)) {
                ++collapses;
                collapsed = true;
                m_energy_log.push_back(cand.cost);
                std::cout << "  COLLAPSED (" << cand.vi << "," << cand.vj
                          << ") cost=" << cand.cost << " pos=(" << cand.pos[0] << "," << cand.pos[1] << ")\n";
                break;
            }
            std::cout << "    rejected by TriMesh, trying next\n";
        }

        if (!collapsed) {
            std::cout << "All candidates rejected; stopping.\n";
            break;
        }


        // Collapse cheapest 20% of candidates in one batch
        // const int n_verts_now = (int)get_vertices().size();
        // const int batch_target = std::max(1, n_verts_now / 5);  // 20%
        // int batch_done = 0;

        // for (auto& cand : candidates) {
        //     if (batch_done >= batch_target) break;
        //     if ((int)get_vertices().size() <= target_vertices) break;

        //     // Skip stale Tuples invalidated by earlier collapses in this batch.
        //     // Must check before any vid() / switch_vertex() access to avoid UB,
        //     // and before collapse_edge() which asserts is_valid inside check_link_condition.
        //     if (!cand.tuple.is_valid(*this)) continue;

        //     const auto& edge = cand.tuple;
        //     const int vi_wmtk = (int)edge.vid(*this);
        //     const int vj_wmtk = (int)edge.switch_vertex(*this).vid(*this);

        //     Eigen::Vector2d vi_pos = vertex_attrs[vi_wmtk].pos;
        //     Eigen::Vector2d vj_pos = vertex_attrs[vj_wmtk].pos;
        //     m_pending_pos = vi_pos + cand.ratio * (vj_pos - vi_pos);
        //     std::vector<Tuple> new_tris;
        //     if (collapse_edge(cand.tuple, new_tris)) {
        //         ++collapses;
        //         ++batch_done;
        //         m_energy_log.push_back(cand.cost);
        //         std::cout << "  COLLAPSED (" << cand.vi << "," << cand.vj
        //                 << ") cost=" << cand.cost << "\n";
        //     }
        //     // 失敗した辺はスキップして次へ（wmtkがstale Tupleを自動拒否）
        // }

        // if (batch_done == 0) {
        //     std::cout << "All candidates rejected; stopping.\n";
        //     break;
        // }

        // Update V_curr / F_curr to reflect the just-performed collapse.
        extract_current_mesh(V_curr, F_curr);

        // ── Coverage check (debug): count fine boundary verts outside coarse mesh ──
        if (!m_fine_boundary_verts.empty()) {
            const double eps_bary = 1e-8; // loose tolerance for on-boundary verts
            int uncovered = 0;
            for (int fbv : m_fine_boundary_verts) {
                const Eigen::Vector2d p = m_V_fine.row(fbv);
                bool found = false;
                for (int f = 0; f < (int)F_curr.rows() && !found; ++f) {
                    const Eigen::Vector2d a  = V_curr.row(F_curr(f,0));
                    const Eigen::Vector2d b  = V_curr.row(F_curr(f,1));
                    const Eigen::Vector2d cc = V_curr.row(F_curr(f,2));
                    const double den = (b-a)[0]*(cc-a)[1] - (cc-a)[0]*(b-a)[1];
                    if (std::abs(den) < 1e-14) continue;
                    const double inv = 1.0 / den;
                    const Eigen::Vector2d v2 = p - a;
                    const double u  = (v2[0]*(cc-a)[1] - (cc-a)[0]*v2[1]) * inv;
                    const double v_ = ((b-a)[0]*v2[1]  - v2[0]*(b-a)[1]) * inv;
                    const double w  = 1.0 - u - v_;
                    if (u >= -eps_bary && v_ >= -eps_bary && w >= -eps_bary)
                        found = true;
                }
                if (!found) {
                    ++uncovered;
                    std::cout << "  UNCOVERED fbv=" << fbv
                              << " pos=(" << p[0] << "," << p[1] << ")\n";
                }
            }
            if (uncovered > 0)
                std::cout << "  WARNING: " << uncovered << " fine boundary verts uncovered!\n";
        }

        // ── Per-step OBJ save + energy plot ─────────────────────────────────
        if (!m_output_dir.empty()) {
            std::filesystem::create_directories(m_output_dir + "/steps");
            std::ostringstream ss;
            ss << m_output_dir << "/steps/step_"
               << std::setw(4) << std::setfill('0') << collapses << ".obj";
            Eigen::MatrixXd V3 = Eigen::MatrixXd::Zero(V_curr.rows(), 3);
            V3.leftCols(2) = V_curr;
            igl::write_triangle_mesh(ss.str(), V3, F_curr);

            save_energy_svg(collapses);
        }
        auto after_step_time = std::chrono::high_resolution_clock::now();
        double step_time_sec = std::chrono::duration<double>(after_step_time - after_cost_time).count();
        std::cout << "Completed step " << collapses << " in " << step_time_sec << " seconds.\n";
    }

    return collapses;
}

// ============================================================
//  Write output OBJ
// ============================================================

void EigenEdgeCollapse::write_obj(const std::string& path) const
{
    Eigen::MatrixXd V3 = Eigen::MatrixXd::Zero((int)vert_capacity(), 3);
    for (const auto& t : get_vertices()) {
        int i = (int)t.vid(*this);
        V3(i, 0) = vertex_attrs[i].pos[0];
        V3(i, 1) = vertex_attrs[i].pos[1];
    }

    Eigen::MatrixXi F((int)get_faces().size(), 3);
    int row = 0;
    for (const auto& t : get_faces()) {
        auto vs = oriented_tri_vertices(t);
        for (int j = 0; j < 3; ++j)
            F(row, j) = (int)vs[j].vid(*this);
        ++row;
    }

    igl::write_triangle_mesh(path, V3, F);
    std::cout << "Wrote " << path << "\n";
}

// ============================================================
//  SVG material visualisation
// ============================================================

// Map a value in [0,1] to an RGB colour using a blue→white→red colormap
// (matches matplotlib's RdBu_r qualitatively: blue=low, red=high)
static void colormap(double t, int& r, int& g, int& b)
{
    t = std::max(0.0, std::min(1.0, t));
    if (t < 0.5) {
        // blue (0,0,255) → white (255,255,255)
        double s = t * 2.0;
        r = (int)(255 * s);
        g = (int)(255 * s);
        b = 255;
    } else {
        // white (255,255,255) → red (255,0,0)
        double s = (t - 0.5) * 2.0;
        r = 255;
        g = (int)(255 * (1.0 - s));
        b = (int)(255 * (1.0 - s));
    }
}

// ============================================================
//  Output directory
// ============================================================

void EigenEdgeCollapse::set_output_dir(const std::string& dir)
{
    m_output_dir = dir;
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir + "/eigenmodes");
    std::filesystem::create_directories(dir + "/edge_costs");
}

// ============================================================
//  build_P_free  (shared helper for cost + eigen viz)
// ============================================================

Eigen::MatrixXd EigenEdgeCollapse::build_P_free(
    const Eigen::MatrixXd& V_coarse,
    const Eigen::MatrixXi& F_coarse,
    const std::vector<int>& free_c) const
{
    const int n_f   = (int)m_V_fine.rows();
    const int ndof_f = 2 * n_f;
    const int nf_c  = (int)free_c.size();

    Eigen::SparseMatrix<double> Ps = build_barycentric_P(V_coarse, F_coarse);
    Eigen::MatrixXd P_free = Eigen::MatrixXd::Zero(ndof_f, nf_c);
    for (int k = 0; k < nf_c; ++k) {
        int gc = free_c[k], vc = gc / 2, comp = gc % 2;
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ps, vc); it; ++it)
            P_free(2 * (int)it.row() + comp, k) = it.value();
    }
    return P_free;
}

// ============================================================
//  Tiny bitmap font + scientific-notation formatter
//  (used to label the colorbar in save_edge_cost_png)
// ============================================================

namespace {

// 4-column × 6-row bitmap font.  bit3 = leftmost pixel per row.
// Covers digits 0-9, '.', '-', '+', 'e', 'E', ' '.
struct G46 { char ch; uint8_t r[6]; };
static const G46 kG46[] = {
    {' ', {0x0,0x0,0x0,0x0,0x0,0x0}},
    {'0', {0x6,0x9,0x9,0x9,0x9,0x6}},
    {'1', {0x2,0x6,0x2,0x2,0x2,0x7}},
    {'2', {0x6,0x9,0x1,0x2,0x4,0xF}},
    {'3', {0xE,0x1,0x6,0x1,0x1,0xE}},
    {'4', {0x9,0x9,0xF,0x1,0x1,0x1}},
    {'5', {0xF,0x8,0xE,0x1,0x9,0x6}},
    {'6', {0x6,0x8,0xE,0x9,0x9,0x6}},
    {'7', {0xF,0x1,0x2,0x2,0x4,0x4}},
    {'8', {0x6,0x9,0x6,0x9,0x9,0x6}},
    {'9', {0x6,0x9,0x9,0x7,0x1,0x6}},
    {'.', {0x0,0x0,0x0,0x0,0x6,0x6}},
    {'-', {0x0,0x0,0xF,0x0,0x0,0x0}},
    {'+', {0x0,0x4,0xE,0x4,0x0,0x0}},
    {'e', {0x0,0x6,0x9,0xF,0x8,0x6}},
    {'E', {0xF,0x8,0xE,0x8,0x8,0xF}},
    {0,   {0,0,0,0,0,0}},
};

static void draw_small_text(vis::Framebuffer& fb, int x, int y,
                             const std::string& s, uint8_t r, uint8_t g, uint8_t b)
{
    int cx = x;
    for (char ch : s) {
        for (int i = 0; kG46[i].ch != 0; ++i) {
            if (kG46[i].ch != ch) continue;
            for (int row = 0; row < 6; ++row) {
                const uint8_t bits = kG46[i].r[row];
                for (int col = 0; col < 4; ++col)
                    if (bits & (0x8u >> col))
                        fb.set_pixel(cx + col, y + row, r, g, b);
            }
            break;
        }
        cx += 5; // 4px glyph + 1px gap
    }
}

static std::string fmt_sci(double v)
{
    if (!std::isfinite(v) || v <= 0) return "---";
    int exp = (int)std::floor(std::log10(v));
    double mant = v / std::pow(10.0, (double)exp);
    if (mant >= 9.95) { mant = 1.0; ++exp; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << mant << "e";
    if (exp >= 0) ss << "+";
    ss << exp;
    return ss.str();
}

} // namespace

// ============================================================
//  Edge-cost PNG  (called per step from simplify)
// ============================================================

void EigenEdgeCollapse::save_edge_cost_png(
    int step,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int,int>>& edges,
    const std::vector<double>& costs,
    int best_idx) const
{
    using namespace vis;
    const int W = 1200, H = 480, pad = 35;
    Framebuffer fb(W, H, 245);
    CoordMap cm(V, W - 90, H, pad); // leave 90px for colorbar

    // Draw mesh faces in very light gray
    for (int f = 0; f < (int)F.rows(); ++f) {
        fill_tri(fb,
                 cm.screen_x(V(F(f,0),0)), cm.screen_y(V(F(f,0),1)),
                 cm.screen_x(V(F(f,1),0)), cm.screen_y(V(F(f,1),1)),
                 cm.screen_x(V(F(f,2),0)), cm.screen_y(V(F(f,2),1)),
                 220, 220, 220);
    }
    draw_mesh_edges(fb, cm, V, F, 180, 180, 180, 1);

    // Collect finite positive costs for log-scale range
    std::vector<double> finite_costs;
    for (double c : costs)
        if (std::isfinite(c) && c > 0) finite_costs.push_back(std::log10(c));
    if (finite_costs.empty()) {
        fb.save_png(m_output_dir + "/edge_costs/step_"
                    + std::string(4 - std::to_string(step).size(), '0')
                    + std::to_string(step) + ".png");
        return;
    }
    std::sort(finite_costs.begin(), finite_costs.end());
    const float log_min = (float)finite_costs.front();
    const float log_max = (float)finite_costs.back();
    const float log_range = std::max(log_max - log_min, 1e-6f);

    // Draw edges coloured by log10(cost) directly
    for (int k = 0; k < (int)edges.size(); ++k) {
        if (!std::isfinite(costs[k]) || costs[k] <= 0) continue;
        float t = (std::log10((float)costs[k]) - log_min) / log_range;
        uint8_t r, g, b;
        cm_viridis(t, r, g, b);
        const auto& [vi, vj] = edges[k];
        draw_line(fb,
                  cm.screen_x(V(vi,0)), cm.screen_y(V(vi,1)),
                  cm.screen_x(V(vj,0)), cm.screen_y(V(vj,1)),
                  r, g, b, 2);
    }

    // Highlight best edge in red
    if (best_idx >= 0 && best_idx < (int)edges.size()) {
        const auto& [vi, vj] = edges[best_idx];
        draw_line(fb,
                  cm.screen_x(V(vi,0)), cm.screen_y(V(vi,1)),
                  cm.screen_x(V(vj,0)), cm.screen_y(V(vj,1)),
                  220, 30, 30, 3);
        draw_circle(fb, cm.screen_x(V(vi,0)), cm.screen_y(V(vi,1)), 4, 220,30,30);
        draw_circle(fb, cm.screen_x(V(vj,0)), cm.screen_y(V(vj,1)), 4, 220,30,30);
    }

    // Draw vertex dots
    for (int i = 0; i < (int)V.rows(); ++i)
        draw_circle(fb, cm.screen_x(V(i,0)), cm.screen_y(V(i,1)), 2, 50, 50, 50);

    auto binfo = classify_boundary(V, F);
    // Highlight boundary vertices in blue
    for (int i = 0; i < (int)V.rows(); ++i)
        if (binfo.on_boundary[i])
            draw_circle(fb, cm.screen_x(V(i,0)), cm.screen_y(V(i,1)), 3, 30, 30, 220);


    // Colorbar
    draw_colorbar(fb, W - 80, pad, 16, H - 2 * pad, log_min, log_max, false, true);

    // Colorbar labels: max (top), mid (centre), min (bottom)
    {
        const int lx = W - 80 + 16 + 3; // just right of the colorbar
        const int cb_h = H - 2 * pad;
        const double c_max = std::pow(10.0, (double)log_max);
        const double c_mid = std::pow(10.0, 0.5 * ((double)log_min + (double)log_max));
        const double c_min = std::pow(10.0, (double)log_min);
        draw_small_text(fb, lx, pad,                     fmt_sci(c_max), 30, 30, 30);
        draw_small_text(fb, lx, pad + cb_h / 2 - 3,     fmt_sci(c_mid), 30, 30, 30);
        draw_small_text(fb, lx, pad + cb_h - 6,         fmt_sci(c_min), 30, 30, 30);
    }

    // Save
    std::string idx4 = std::string(4 - std::to_string(step).size(), '0') + std::to_string(step);
    fb.save_png(m_output_dir + "/edge_costs/step_" + idx4 + ".png");

    // ── Also save SVG with the same content ──────────────────────────────
    {
        const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
        const double ymin = V.col(1).minCoeff(), ymax = V.col(1).maxCoeff();
        const int draw_w = W - 90;
        const double scale = std::min(
            (draw_w - 2.0 * pad) / std::max(xmax - xmin, 1e-9),
            (H      - 2.0 * pad) / std::max(ymax - ymin, 1e-9));
        auto px = [&](double x) { return pad + (x - xmin) * scale; };
        auto py = [&](double y) { return H - pad - (y - ymin) * scale; };

        std::ofstream ofs(m_output_dir + "/edge_costs/step_" + idx4 + ".svg");
        ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            << "width=\"" << W << "\" height=\"" << H << "\">\n"
            << "  <rect width=\"100%\" height=\"100%\" fill=\"rgb(245,245,245)\"/>\n";

        // Mesh faces (light gray)
        for (int f = 0; f < (int)F.rows(); ++f) {
            ofs << "  <polygon points=\""
                << std::fixed << std::setprecision(2)
                << px(V(F(f,0),0)) << "," << py(V(F(f,0),1)) << " "
                << px(V(F(f,1),0)) << "," << py(V(F(f,1),1)) << " "
                << px(V(F(f,2),0)) << "," << py(V(F(f,2),1)) << "\""
                << " fill=\"rgb(220,220,220)\""
                << " stroke=\"rgb(180,180,180)\" stroke-width=\"1\"/>\n";
        }

        // Candidate edges colored by log10(cost) directly
        for (int k = 0; k < (int)edges.size(); ++k) {
            if (!std::isfinite(costs[k]) || costs[k] <= 0) continue;
            float t = (std::log10((float)costs[k]) - log_min) / log_range;
            uint8_t r, g, b;
            cm_viridis(t, r, g, b);
            const auto& [vi, vj] = edges[k];
            ofs << "  <line x1=\"" << px(V(vi,0)) << "\" y1=\"" << py(V(vi,1))
                << "\" x2=\"" << px(V(vj,0)) << "\" y2=\"" << py(V(vj,1))
                << "\" stroke=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\""
                << " stroke-width=\"2\"/>\n";
        }

        // Best edge in red
        if (best_idx >= 0 && best_idx < (int)edges.size()) {
            const auto& [vi, vj] = edges[best_idx];
            ofs << "  <line x1=\"" << px(V(vi,0)) << "\" y1=\"" << py(V(vi,1))
                << "\" x2=\"" << px(V(vj,0)) << "\" y2=\"" << py(V(vj,1))
                << "\" stroke=\"rgb(220,30,30)\" stroke-width=\"3\"/>\n";
            ofs << "  <circle cx=\"" << px(V(vi,0)) << "\" cy=\"" << py(V(vi,1))
                << "\" r=\"4\" fill=\"rgb(220,30,30)\"/>\n";
            ofs << "  <circle cx=\"" << px(V(vj,0)) << "\" cy=\"" << py(V(vj,1))
                << "\" r=\"4\" fill=\"rgb(220,30,30)\"/>\n";
        }

        // Vertex dots
        for (int i = 0; i < (int)V.rows(); ++i)
            ofs << "  <circle cx=\"" << px(V(i,0)) << "\" cy=\"" << py(V(i,1))
                << "\" r=\"2\" fill=\"rgb(50,50,50)\"/>\n";

        // Boundary vertices in blue
        for (int i = 0; i < (int)V.rows(); ++i)
            if (binfo.on_boundary[i])
                ofs << "  <circle cx=\"" << px(V(i,0)) << "\" cy=\"" << py(V(i,1))
                    << "\" r=\"3\" fill=\"rgb(30,30,220)\"/>\n";

        // Colorbar (viridis gradient)
        const int cb_x = W - 80;
        const int cb_y = pad;
        const int cb_w = 16;
        const int cb_h = H - 2 * pad;
        const int gsteps = 64;
        for (int s = 0; s < gsteps; ++s) {
            float t = 1.0f - (float)s / (float)gsteps;
            uint8_t r, g, b;
            cm_viridis(t, r, g, b);
            double y0 = cb_y + (double)s / gsteps * cb_h;
            double dy = (double)cb_h / gsteps + 1.0;
            ofs << "  <rect x=\"" << cb_x << "\" y=\""
                << std::fixed << std::setprecision(1)
                << y0 << "\" width=\"" << cb_w << "\" height=\"" << dy << "\""
                << " fill=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\"/>\n";
        }
        ofs << "  <rect x=\"" << cb_x << "\" y=\"" << cb_y
            << "\" width=\"" << cb_w << "\" height=\"" << cb_h
            << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

        // Colorbar labels
        const double c_max = std::pow(10.0, (double)log_max);
        const double c_mid = std::pow(10.0, 0.5 * ((double)log_min + (double)log_max));
        const double c_min = std::pow(10.0, (double)log_min);
        auto svg_label = [&](double cval, double frac) {
            std::ostringstream ss;
            ss << std::scientific << std::setprecision(1) << cval;
            ofs << "  <text x=\"" << cb_x + cb_w + 3
                << "\" y=\"" << std::fixed << std::setprecision(1)
                << (cb_y + (1.0 - frac) * cb_h + 4)
                << "\" font-size=\"10\" font-family=\"monospace\">"
                << ss.str() << "</text>\n";
        };
        svg_label(c_max, 1.0);
        svg_label(c_mid, 0.5);
        svg_label(c_min, 0.0);

        ofs << "</svg>\n";
    }
}

// ============================================================
//  Eigenmode PNG (left = fine deformation, right = coarse)
// ============================================================

void EigenEdgeCollapse::save_eigenmode_pngs() const
{
    using namespace vis;

    // Extract current coarse mesh
    Eigen::MatrixXd V_c;
    Eigen::MatrixXi F_c;
    extract_current_mesh(V_c, F_c);
    if (F_c.rows() == 0) return;

    // Assemble and factorize coarse system
    Eigen::SparseMatrix<double> Kc;
    Eigen::VectorXd Mc_diag;
    assemble_fem(V_c, F_c, Kc, Mc_diag);
    apply_spring_bcs(Kc, V_c, F_c);

    const int ndof_c = (int)Kc.rows();
    std::vector<int> fixed_c =
        (m_p.spring_k > 0.0) ? std::vector<int>{}
        : (m_p.fixed_left    ? left_boundary_verts(V_c) : std::vector<int>{});
    std::vector<int> free_c  = free_dof_indices(ndof_c, fixed_c);
    if (free_c.empty()) return;

    const int nf_c = (int)free_c.size();
    std::vector<int> idx_c(ndof_c, -1);
    for (int k = 0; k < nf_c; ++k) idx_c[free_c[k]] = k;

    Eigen::SparseMatrix<double> Kc_ff(nf_c, nf_c);
    {
        std::vector<Eigen::Triplet<double>> tr;
        for (int j = 0; j < nf_c; ++j) {
            int col = free_c[j];
            for (Eigen::SparseMatrix<double>::InnerIterator it(Kc, col); it; ++it) {
                int ri = (int)it.row();
                if (idx_c[ri] >= 0) tr.emplace_back(idx_c[ri], j, it.value());
            }
        }
        Kc_ff.setFromTriplets(tr.begin(), tr.end());
    }

    // Kinetic shift on coarse K_eff = K_c + alpha*M_c (matches fine-mesh shift).
    // Skipped when spring_k > 0 (springs already regularise the system).
    if (m_p.spring_k <= 0.0 && !m_p.fixed_left && m_p.alpha > 0.0)
        for (int k = 0; k < nf_c; ++k)
            Kc_ff.coeffRef(k, k) += m_p.alpha * Mc_diag[free_c[k]];

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(Kc_ff);
    if (ldlt.info() != Eigen::Success) return;

    const Eigen::MatrixXd P_free = build_P_free(V_c, F_c, free_c);

    // Canvas layout: two panels side by side
    const int PW = 860, PH = 380, gap = 10, pad = 30;
    const int W = 2 * PW + gap, H = PH;

    // Coordinate maps — both panels use the same world scale for comparability
    CoordMap cm_fine (m_V_fine, PW, PH, pad);
    CoordMap cm_coarse(V_c,    PW, PH, pad);

    // Auto displacement scale: 8% of fine mesh diagonal / max |phi|
    const double diag_f = (m_V_fine.colwise().maxCoeff() - m_V_fine.colwise().minCoeff()).norm();

    for (int mi = 0; mi < (int)m_modes.size(); ++mi) {
        const int mode_id = m_modes[mi];
        const double lam = m_evals[mode_id];
        if (!std::isfinite(lam) || std::abs(lam) <= m_p.eig_tol) continue;

        // ── Fine displacement ─────────────────────────────────────────
        Eigen::VectorXd u_fine = (1.0 / lam) * m_evecs.col(mode_id);

        // ── Coarse displacement ───────────────────────────────────────
        Eigen::VectorXd bf   = m_M_diag.cwiseProduct(m_evecs.col(mode_id));
        Eigen::VectorXd bc   = P_free.transpose() * bf;
        Eigen::VectorXd yc_f = ldlt.solve(bc);
        // Expand free-DOF solution to all coarse DOFs (zero on fixed)
        Eigen::VectorXd u_coarse = Eigen::VectorXd::Zero(ndof_c);
        for (int k = 0; k < nf_c; ++k) u_coarse[free_c[k]] = yc_f[k];

        // ── Displacement scale ────────────────────────────────────────
        double max_uf = u_fine.cwiseAbs().maxCoeff();
        double max_uc = u_coarse.cwiseAbs().maxCoeff();
        double max_u  = std::max({max_uf, max_uc, 1e-12});
        double dscale = 0.08 * diag_f / max_u;

        u_fine   *= dscale;
        u_coarse *= dscale;

        // ── Per-face colour = average displacement magnitude ──────────
        auto face_mag = [&](const Eigen::MatrixXd& V,
                            const Eigen::MatrixXi& F,
                            const Eigen::VectorXd& u,
                            double ref_max) -> std::vector<float> {
            std::vector<float> mag(F.rows());
            for (int f = 0; f < (int)F.rows(); ++f) {
                double s = 0;
                for (int j = 0; j < 3; ++j) {
                    int v = F(f, j);
                    double ux = u[2*v], uy = u[2*v+1];
                    s += std::sqrt(ux*ux + uy*uy);
                }
                mag[f] = (float)std::min(1.0, (s / 3.0) / ref_max);
            }
            return mag;
        };

        double ref = std::max(u_fine.cwiseAbs().maxCoeff(),
                              u_coarse.cwiseAbs().maxCoeff());
        ref = std::max(ref, 1e-12);

        auto mag_f = face_mag(m_V_fine, m_F_fine, u_fine,   ref);
        auto mag_c = face_mag(V_c,      F_c,       u_coarse, ref);

        // ── Render ────────────────────────────────────────────────────
        Framebuffer fb(W, H, 245);

        // Left panel: fine mesh deformed by u_fine
        draw_displaced_mesh(fb, cm_fine, m_V_fine, m_F_fine, u_fine, mag_f);

        // Right panel: coarse mesh deformed by u_coarse
        // Shift the CoordMap origin so right panel starts at x = PW + gap
        CoordMap cm_r = cm_coarse;
        cm_r.ox += (float)(PW + gap);
        draw_displaced_mesh(fb, cm_r, V_c, F_c, u_coarse, mag_c);

        // Panel separator
        for (int y = 0; y < H; ++y)
            for (int x = PW; x < PW + gap; ++x)
                fb.set_pixel(x, y, 200, 200, 200);

        // Colorbars
        draw_colorbar(fb, PW - 28, pad, 14, H - 2*pad, 0.0f, (float)ref);
        draw_colorbar(fb, W   - 28, pad, 14, H - 2*pad, 0.0f, (float)ref);

        // Save
        std::string idx4 = std::string(4 - std::to_string(mi).size(), '0') + std::to_string(mi);
        fb.save_png(m_output_dir + "/eigenmodes/mode_" + idx4 + ".png");

        // ── Also save SVG with the same content ──────────────────────────
        {
            std::ofstream ofs(m_output_dir + "/eigenmodes/mode_" + idx4 + ".svg");
            ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                << "width=\"" << W << "\" height=\"" << H << "\">\n"
                << "  <rect width=\"100%\" height=\"100%\" fill=\"rgb(245,245,245)\"/>\n";

            // Helper: emit one mesh panel
            auto draw_panel_svg = [&](
                const Eigen::MatrixXd& V,
                const Eigen::MatrixXi& F,
                const Eigen::VectorXd& u,
                const std::vector<float>& mag,
                const CoordMap& cm)
            {
                for (int f = 0; f < (int)F.rows(); ++f) {
                    uint8_t r, g, b;
                    cm_viridis(mag[f], r, g, b);
                    const int v0 = F(f, 0), v1 = F(f, 1), v2 = F(f, 2);
                    const double x0 = V(v0,0) + u[2*v0],   y0 = V(v0,1) + u[2*v0+1];
                    const double x1 = V(v1,0) + u[2*v1],   y1 = V(v1,1) + u[2*v1+1];
                    const double x2 = V(v2,0) + u[2*v2],   y2 = V(v2,1) + u[2*v2+1];
                    ofs << "  <polygon points=\""
                        << std::fixed << std::setprecision(2)
                        << cm.screen_x(x0) << "," << cm.screen_y(y0) << " "
                        << cm.screen_x(x1) << "," << cm.screen_y(y1) << " "
                        << cm.screen_x(x2) << "," << cm.screen_y(y2) << "\""
                        << " fill=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\""
                        << " stroke=\"rgb(80,80,80)\" stroke-width=\"0.3\"/>\n";
                }
            };

            draw_panel_svg(m_V_fine, m_F_fine, u_fine,   mag_f, cm_fine);
            draw_panel_svg(V_c,      F_c,      u_coarse, mag_c, cm_r);

            // Panel separator
            ofs << "  <rect x=\"" << PW << "\" y=\"0\" width=\"" << gap
                << "\" height=\"" << H << "\" fill=\"rgb(200,200,200)\"/>\n";

            // Colorbars (one per panel) — viridis from 0 (bottom) to ref (top)
            auto draw_colorbar_svg = [&](int cb_x) {
                const int cb_y = pad;
                const int cb_w = 14;
                const int cb_h = H - 2 * pad;
                const int steps = 64;
                for (int s = 0; s < steps; ++s) {
                    float t = 1.0f - (float)s / (float)steps;
                    uint8_t r, g, b;
                    cm_viridis(t, r, g, b);
                    double y0 = cb_y + (double)s / steps * cb_h;
                    double dy = (double)cb_h / steps + 1.0;
                    ofs << "  <rect x=\"" << cb_x << "\" y=\""
                        << std::fixed << std::setprecision(1)
                        << y0 << "\" width=\"" << cb_w << "\" height=\"" << dy << "\""
                        << " fill=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\"/>\n";
                }
                ofs << "  <rect x=\"" << cb_x << "\" y=\"" << cb_y
                    << "\" width=\"" << cb_w << "\" height=\"" << cb_h
                    << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

                std::ostringstream ss_max, ss_zero;
                ss_max  << std::scientific << std::setprecision(1) << ref;
                ss_zero << "0";
                ofs << "  <text x=\"" << cb_x + cb_w + 3 << "\" y=\""
                    << cb_y + 4 << "\" font-size=\"10\" font-family=\"monospace\">"
                    << ss_max.str() << "</text>\n";
                ofs << "  <text x=\"" << cb_x + cb_w + 3 << "\" y=\""
                    << cb_y + cb_h + 4 << "\" font-size=\"10\" font-family=\"monospace\">"
                    << ss_zero.str() << "</text>\n";
            };
            draw_colorbar_svg(PW - 28);
            draw_colorbar_svg(W  - 28);

            // Mode label
            std::ostringstream lam_str;
            lam_str << std::scientific << std::setprecision(3) << lam;
            ofs << "  <text x=\"" << pad << "\" y=\"" << pad - 8
                << "\" font-size=\"12\" font-family=\"sans-serif\" font-weight=\"bold\">"
                << "Mode " << mi << "  lambda=" << lam_str.str() << "</text>\n";

            ofs << "</svg>\n";
        }
    }
    std::cout << "Saved " << m_modes.size() << " eigenmode PNG/SVGs to "
              << m_output_dir << "/eigenmodes/\n";
}

// ============================================================
//  Energy plot SVG
// ============================================================

void EigenEdgeCollapse::save_energy_svg(int step) const
{
    if (m_output_dir.empty() || m_energy_log.empty()) return;

    std::filesystem::create_directories(m_output_dir + "/energy");

    std::ostringstream fname;
    fname << m_output_dir << "/energy/energy_"
          << std::setw(4) << std::setfill('0') << step << ".svg";

    // Canvas layout
    const int W = 720, H = 400;
    const int ml = 90, mr = 30, mt = 30, mb = 60; // margins
    const int pw = W - ml - mr; // plot width
    const int ph = H - mt - mb; // plot height

    const int n = (int)m_energy_log.size();

    // y-axis range
    double ymin = *std::min_element(m_energy_log.begin(), m_energy_log.end());
    double ymax = *std::max_element(m_energy_log.begin(), m_energy_log.end());
    if (ymax <= ymin) ymax = ymin + 1.0;
    // Add 5% padding
    const double ypad = 0.05 * (ymax - ymin);
    ymin -= ypad;
    ymax += ypad;

    // Coordinate helpers
    auto sx = [&](int i) -> double {
        return ml + (n == 1 ? pw * 0.5 : (double)i / (n - 1) * pw);
    };
    auto sy = [&](double v) -> double {
        return mt + ph * (1.0 - (v - ymin) / (ymax - ymin));
    };

    std::ofstream ofs(fname.str());
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << W << "\" height=\"" << H << "\">\n"
        << "  <rect width=\"100%\" height=\"100%\" fill=\"#fafafa\"/>\n";

    // Plot background
    ofs << "  <rect x=\"" << ml << "\" y=\"" << mt
        << "\" width=\"" << pw << "\" height=\"" << ph
        << "\" fill=\"white\" stroke=\"#ccc\" stroke-width=\"1\"/>\n";

    // Horizontal grid lines (5 lines)
    for (int k = 0; k <= 4; ++k) {
        double v = ymin + k * (ymax - ymin) / 4.0;
        double gy = sy(v);
        ofs << "  <line x1=\"" << ml << "\" y1=\"" << std::fixed << std::setprecision(1) << gy
            << "\" x2=\"" << ml + pw << "\" y2=\"" << gy
            << "\" stroke=\"#ddd\" stroke-width=\"1\"/>\n";
        // y-axis tick label
        std::ostringstream ss;
        if (std::abs(v) >= 1000 || (std::abs(v) < 0.001 && v != 0.0))
            ss << std::scientific << std::setprecision(2) << v;
        else
            ss << std::fixed << std::setprecision(4) << v;
        ofs << "  <text x=\"" << ml - 6 << "\" y=\""
            << std::fixed << std::setprecision(1) << (gy + 4)
            << "\" font-size=\"11\" font-family=\"monospace\""
            << " text-anchor=\"end\">" << ss.str() << "</text>\n";
    }

    // Vertical grid lines (up to 10)
    const int x_ticks = std::min(n, 10);
    for (int k = 0; k <= x_ticks; ++k) {
        int idx = (x_ticks == 0) ? 0 : (int)std::round((double)k / x_ticks * (n - 1));
        idx = std::min(idx, n - 1);
        double gx = sx(idx);
        ofs << "  <line x1=\"" << std::fixed << std::setprecision(1) << gx
            << "\" y1=\"" << mt << "\" x2=\"" << gx << "\" y2=\"" << mt + ph
            << "\" stroke=\"#ddd\" stroke-width=\"1\"/>\n";
        ofs << "  <text x=\"" << gx << "\" y=\"" << mt + ph + 18
            << "\" font-size=\"11\" font-family=\"monospace\""
            << " text-anchor=\"middle\">" << idx + 1 << "</text>\n";
    }

    // Energy polyline
    ofs << "  <polyline fill=\"none\" stroke=\"#2266cc\" stroke-width=\"2\""
        << " points=\"";
    for (int i = 0; i < n; ++i)
        ofs << std::fixed << std::setprecision(2)
            << sx(i) << "," << sy(m_energy_log[i])
            << (i + 1 < n ? " " : "");
    ofs << "\"/>\n";

    // Data points
    for (int i = 0; i < n; ++i) {
        ofs << "  <circle cx=\"" << std::fixed << std::setprecision(2)
            << sx(i) << "\" cy=\"" << sy(m_energy_log[i])
            << "\" r=\"3\" fill=\"#2266cc\"/>\n";
    }
    // Highlight last point in red
    ofs << "  <circle cx=\"" << std::fixed << std::setprecision(2)
        << sx(n - 1) << "\" cy=\"" << sy(m_energy_log.back())
        << "\" r=\"4\" fill=\"#cc2222\"/>\n";

    // Axes
    ofs << "  <line x1=\"" << ml << "\" y1=\"" << mt
        << "\" x2=\"" << ml << "\" y2=\"" << mt + ph
        << "\" stroke=\"#333\" stroke-width=\"1.5\"/>\n";
    ofs << "  <line x1=\"" << ml << "\" y1=\"" << mt + ph
        << "\" x2=\"" << ml + pw << "\" y2=\"" << mt + ph
        << "\" stroke=\"#333\" stroke-width=\"1.5\"/>\n";

    // Axis labels
    ofs << "  <text x=\"" << W / 2 << "\" y=\"" << H - 8
        << "\" font-size=\"13\" font-family=\"sans-serif\" text-anchor=\"middle\""
        << ">Collapse iteration</text>\n";
    ofs << "  <text x=\"" << 14 << "\" y=\"" << mt + ph / 2
        << "\" font-size=\"13\" font-family=\"sans-serif\" text-anchor=\"middle\""
        << " transform=\"rotate(-90," << 14 << "," << mt + ph / 2 << ")\""
        << ">Energy (cost)</text>\n";

    // Title
    ofs << "  <text x=\"" << W / 2 << "\" y=\"" << mt - 8
        << "\" font-size=\"13\" font-family=\"sans-serif\" text-anchor=\"middle\""
        << ">Energy vs collapse step  (step " << step << ")</text>\n";

    ofs << "</svg>\n";
}

// ============================================================
//  SVG material visualisation
// ============================================================

void EigenEdgeCollapse::save_material_svg(
    const std::string& path,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F) const
{
    if (F.rows() == 0) return;

    // Each panel: PW wide, H tall.  Two panels side by side with a colorbar each.
    const int PW = 500, H = 300, pad = 20, cb_w = 18, cb_gap = 10, label_w = 60;
    const int panel_stride = PW + cb_gap + cb_w + label_w + 20; // x-offset for second panel
    const int total_W = 2 * panel_stride;

    const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
    const double ymin = V.col(1).minCoeff(), ymax = V.col(1).maxCoeff();
    const double scale = std::min(
        (PW - 2.0 * pad) / std::max(xmax - xmin, 1e-9),
        (H  - 2.0 * pad) / std::max(ymax - ymin, 1e-9));

    auto px = [&](double x, int ox) { return ox + pad + (x - xmin) * scale; };
    auto py = [&](double y)         { return H - pad - (y - ymin) * scale; };

    // Per-triangle E and nu
    const int nf = (int)F.rows();
    std::vector<double> E_tri(nf), nu_tri(nf);
    for (int f = 0; f < nf; ++f) {
        E_tri[f]  = material_E(V, F.row(f));
        nu_tri[f] = material_nu(V, F.row(f));
    }

    auto vrange = [](const std::vector<double>& v) -> std::pair<double,double> {
        double lo = *std::min_element(v.begin(), v.end());
        double hi = *std::max_element(v.begin(), v.end());
        if (hi <= lo) hi = lo + 1.0;
        return {lo, hi};
    };
    auto [emin, emax] = vrange(E_tri);
    auto [numin, numax] = vrange(nu_tri);

    std::ofstream ofs(path);
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << total_W << "\" height=\"" << H + 20 << "\">\n"
        << "  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n";

    // Helper: draw one panel (triangles + vertices + colorbar) at x offset ox.
    auto draw_panel = [&](const std::vector<double>& vals,
                          double vmin, double vmax,
                          int ox, const std::string& title) {
        // Panel title
        ofs << "  <text x=\"" << ox + pad << "\" y=\"14\""
            << " font-size=\"12\" font-family=\"sans-serif\" font-weight=\"bold\">"
            << title << "</text>\n";

        // Triangles
        for (int f = 0; f < nf; ++f) {
            const double t = (vals[f] - vmin) / (vmax - vmin);
            int r, g, b;
            colormap(t, r, g, b);
            ofs << "  <polygon points=\""
                << std::fixed << std::setprecision(2)
                << px(V(F(f,0),0),ox) << "," << py(V(F(f,0),1)) << " "
                << px(V(F(f,1),0),ox) << "," << py(V(F(f,1),1)) << " "
                << px(V(F(f,2),0),ox) << "," << py(V(F(f,2),1)) << "\""
                << " fill=\"rgb(" << r << "," << g << "," << b << ")\""
                << " stroke=\"#444\" stroke-width=\"0.4\"/>\n";
        }

        // Vertices
        for (int i = 0; i < (int)V.rows(); ++i)
            ofs << "  <circle cx=\"" << std::fixed << std::setprecision(2)
                << px(V(i,0),ox) << "\" cy=\"" << py(V(i,1))
                << "\" r=\"2\" fill=\"#222\" opacity=\"0.7\"/>\n";

        // Colorbar
        const int cb_x = ox + PW + cb_gap;
        const int cb_y = pad, cb_h = H - 2 * pad;
        const int steps = 64;
        for (int s = 0; s < steps; ++s) {
            double t = 1.0 - (double)s / steps;
            int r, g, b;
            colormap(t, r, g, b);
            double y0 = cb_y + (double)s / steps * cb_h;
            double dy = (double)cb_h / steps + 1.0;
            ofs << "  <rect x=\"" << cb_x << "\" y=\""
                << std::fixed << std::setprecision(1)
                << y0 << "\" width=\"" << cb_w << "\" height=\"" << dy << "\""
                << " fill=\"rgb(" << r << "," << g << "," << b << ")\"/>\n";
        }
        ofs << "  <rect x=\"" << cb_x << "\" y=\"" << cb_y
            << "\" width=\"" << cb_w << "\" height=\"" << cb_h
            << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

        auto label = [&](double val, double frac) {
            std::ostringstream ss;
            if (std::abs(val) >= 1000 || (std::abs(val) < 0.01 && val != 0.0))
                ss << std::scientific << std::setprecision(2) << val;
            else
                ss << std::fixed << std::setprecision(3) << val;
            ofs << "  <text x=\"" << cb_x + cb_w + 3
                << "\" y=\"" << std::fixed << std::setprecision(1)
                << (cb_y + (1.0 - frac) * cb_h + 4)
                << "\" font-size=\"10\" font-family=\"monospace\">"
                << ss.str() << "</text>\n";
        };
        label(vmax, 1.0);
        label(0.5 * (vmin + vmax), 0.5);
        label(vmin, 0.0);
    };

    draw_panel(E_tri,  emin,  emax,  0,            "E  (vertices=" + std::to_string(V.rows()) + ", faces=" + std::to_string(nf) + ")");
    draw_panel(nu_tri, numin, numax, panel_stride, "nu");

    ofs << "</svg>\n";
    std::cout << "Saved material SVG: " << path << "\n";
}

// Overload: visualise the current TriMesh state
void EigenEdgeCollapse::save_material_svg(const std::string& path) const
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    extract_current_mesh(V, F);
    save_material_svg(path, V, F);
}

// ============================================================
//  Spring overlay SVG: fine mesh + E + spring fine vertices
// ============================================================

void EigenEdgeCollapse::save_springs_svg(const std::string& path) const
{
    using vis::cm_viridis;

    if (m_F_fine.rows() == 0) return;
    if (m_spring_fine_verts.empty()) {
        std::cout << "save_springs_svg: no spring fine vertices configured\n";
        return;
    }

    const Eigen::MatrixXd& V = m_V_fine;
    const Eigen::MatrixXi& F = m_F_fine;

    const int PW = 800, H = 600, pad = 30, cb_w = 18, cb_gap = 10, label_w = 70;
    const int total_W = PW + cb_gap + cb_w + label_w + 20;

    const double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
    const double ymin = V.col(1).minCoeff(), ymax = V.col(1).maxCoeff();
    const double scale = std::min(
        (PW - 2.0 * pad) / std::max(xmax - xmin, 1e-9),
        (H  - 2.0 * pad) / std::max(ymax - ymin, 1e-9));
    auto px = [&](double x) { return pad + (x - xmin) * scale; };
    auto py = [&](double y) { return H - pad - (y - ymin) * scale; };

    // Per-triangle E (use precomputed m_fine_E if available)
    const int nf = (int)F.rows();
    std::vector<double> E_tri(nf);
    for (int f = 0; f < nf; ++f) {
        E_tri[f] = (m_fine_E.size() == nf) ? m_fine_E[f]
                                           : material_E(V, F.row(f));
    }
    double emin = *std::min_element(E_tri.begin(), E_tri.end());
    double emax = *std::max_element(E_tri.begin(), E_tri.end());
    if (emax <= emin) emax = emin + 1.0;

    // Use log scale for E coloring when there's a large dynamic range
    const bool log_scale = (emax / std::max(emin, 1e-30) > 100.0);
    auto e_t = [&](double e) -> double {
        if (log_scale) {
            const double lo = std::log10(std::max(emin, 1e-30));
            const double hi = std::log10(std::max(emax, 1e-30));
            return (std::log10(std::max(e, 1e-30)) - lo) / std::max(hi - lo, 1e-12);
        }
        return (e - emin) / (emax - emin);
    };

    std::ofstream ofs(path);
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << total_W << "\" height=\"" << H + 20 << "\">\n"
        << "  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n";

    // Title
    ofs << "  <text x=\"" << pad << "\" y=\"16\""
        << " font-size=\"13\" font-family=\"sans-serif\" font-weight=\"bold\">"
        << "Spring constraints overlay  (n_springs=" << m_spring_fine_verts.size()
        << ", n_fine_verts=" << V.rows() << ")</text>\n";

    // Triangles colored by E (viridis)
    for (int f = 0; f < nf; ++f) {
        uint8_t r, g, b;
        cm_viridis((float)e_t(E_tri[f]), r, g, b);
        ofs << "  <polygon points=\""
            << std::fixed << std::setprecision(2)
            << px(V(F(f,0),0)) << "," << py(V(F(f,0),1)) << " "
            << px(V(F(f,1),0)) << "," << py(V(F(f,1),1)) << " "
            << px(V(F(f,2),0)) << "," << py(V(F(f,2),1)) << "\""
            << " fill=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\""
            << " stroke=\"#444\" stroke-width=\"0.3\"/>\n";
    }

    // Tiny dots at all fine vertices
    for (int i = 0; i < (int)V.rows(); ++i)
        ofs << "  <circle cx=\"" << std::fixed << std::setprecision(2)
            << px(V(i,0)) << "\" cy=\"" << py(V(i,1))
            << "\" r=\"1.2\" fill=\"#222\" opacity=\"0.5\"/>\n";

    // Spring fine vertices: large red dots with white halo + ID label
    for (size_t k = 0; k < m_spring_fine_verts.size(); ++k) {
        const int vf = m_spring_fine_verts[k];
        if (vf < 0 || vf >= (int)V.rows()) continue;
        const double sx = px(V(vf, 0)), sy = py(V(vf, 1));
        ofs << "  <circle cx=\"" << sx << "\" cy=\"" << sy
            << "\" r=\"5\" fill=\"white\" stroke=\"rgb(220,30,30)\" stroke-width=\"2\"/>\n";
        ofs << "  <circle cx=\"" << sx << "\" cy=\"" << sy
            << "\" r=\"2\" fill=\"rgb(220,30,30)\"/>\n";
        // Only label every Nth vertex if very dense (to avoid clutter)
        const int label_stride = std::max(1, (int)m_spring_fine_verts.size() / 50);
        if ((int)k % label_stride == 0) {
            ofs << "  <text x=\"" << sx + 6 << "\" y=\"" << sy - 4
                << "\" font-size=\"8\" font-family=\"monospace\""
                << " fill=\"rgb(180,20,20)\">" << vf << "</text>\n";
        }
    }

    // Colorbar (viridis)
    const int cb_x = PW + cb_gap;
    const int cb_y = pad;
    const int cb_h = H - 2 * pad;
    const int steps = 64;
    for (int s = 0; s < steps; ++s) {
        float t = 1.0f - (float)s / (float)steps;
        uint8_t r, g, b;
        cm_viridis(t, r, g, b);
        double y0 = cb_y + (double)s / steps * cb_h;
        double dy = (double)cb_h / steps + 1.0;
        ofs << "  <rect x=\"" << cb_x << "\" y=\""
            << std::fixed << std::setprecision(1)
            << y0 << "\" width=\"" << cb_w << "\" height=\"" << dy << "\""
            << " fill=\"rgb(" << (int)r << "," << (int)g << "," << (int)b << ")\"/>\n";
    }
    ofs << "  <rect x=\"" << cb_x << "\" y=\"" << cb_y
        << "\" width=\"" << cb_w << "\" height=\"" << cb_h
        << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

    // Colorbar labels
    auto label = [&](double val, double frac) {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(2) << val;
        ofs << "  <text x=\"" << cb_x + cb_w + 3
            << "\" y=\"" << std::fixed << std::setprecision(1)
            << (cb_y + (1.0 - frac) * cb_h + 4)
            << "\" font-size=\"10\" font-family=\"monospace\">"
            << ss.str() << "</text>\n";
    };
    label(emax, 1.0);
    label(log_scale ? std::sqrt(emin * emax) : 0.5 * (emin + emax), 0.5);
    label(emin, 0.0);

    // Legend for spring markers
    {
        const double lx = pad;
        const double ly = H + 5;
        ofs << "  <circle cx=\"" << lx + 6 << "\" cy=\"" << ly + 6
            << "\" r=\"5\" fill=\"white\" stroke=\"rgb(220,30,30)\" stroke-width=\"2\"/>\n";
        ofs << "  <circle cx=\"" << lx + 6 << "\" cy=\"" << ly + 6
            << "\" r=\"2\" fill=\"rgb(220,30,30)\"/>\n";
        ofs << "  <text x=\"" << lx + 18 << "\" y=\"" << ly + 10
            << "\" font-size=\"11\" font-family=\"sans-serif\">spring fine vertex</text>\n";
    }

    ofs << "</svg>\n";
    std::cout << "Saved springs overlay SVG: " << path
              << " (" << m_spring_fine_verts.size() << " markers)\n";
}

} // namespace app::remesh
