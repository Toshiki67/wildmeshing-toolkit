// ============================================================
//  CostFunctions.cpp — candidate cost evaluation
// ============================================================
//  * Timing accumulators for the Apple Accelerate Woodbury path.
//  * Static helpers apply_*_accel (file-local; share state with the timers).
//  * candidate_cost                       — full assembly + CHOLMOD direct solve.
//  * candidate_cost_woodbury_accelerated  — block-deletion + rank-s Woodbury,
//                                           using Apple's SparseFactor/Solve.
//  * cost_approx                          — O(1-ring) approximate cost.

#include "EigenEdgeCollapse.hpp"
#include "EigenEdgeCollapseInternal.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace app::remesh {

// ============================================================
//  Apple Accelerate apply_*_accel helpers (file-local).
//  Timing accumulators live in EigenEdgeCollapseInternal.hpp so the
//  simplify() loop in EigenEdgeCollapse.cpp can read them.
// ============================================================
namespace {

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

} // anonymous namespace (_accel helpers)


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

} // namespace app::remesh
