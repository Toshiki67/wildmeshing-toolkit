// ============================================================
//  FactorReuseSimplify.cpp
// ============================================================
//  Greedy edge collapse with fine-factorization reuse.
//
//  Algorithm:
//    * L_f = chol(K_f) is computed once (in init_from_obj) and reused for every
//      per-edge cost evaluation.
//    * Cumulative scalar restriction R_s (n_curr × n_fine) and prolongation
//      P_s (n_fine × n_curr) maps are maintained through the collapse history.
//      Both are sparse and indexed by stable wmtk VIDs.
//      Per collapse only one row of R_s and one column of P_s are touched.
//    * Edge costs live in a priority queue.  After each collapse only the closed
//      1-ring of edges around the merged vertex is re-scored.
//
//  Per-edge cost (4×4 trace form):
//      E_{e'} = tr( G_c^loc · A ),
//      A      = Z^T M_f Z                ∈ R^{4×4},   Z = K_f^{-1} W,
//      W      = R_c^T D_loc^T            ∈ R^{2 n_f × 4},
//      G_c^loc = E~^T P_c^T P_c E~        ∈ R^{4×4}.
//
//  D_loc rows are built from barycentric weights β^(a), β^(b) of p_a, p_b in
//  the post-collapse 1-ring of v~.  Cross-axis entries vanish, so D_loc splits
//  per axis (x, y) and the whole pipeline reduces to scalar arithmetic + four
//  fine backsubstitutions per candidate.
//
//  See the paper / latex notes for the derivation.
// ============================================================

#include "EigenEdgeCollapse.hpp"

#include <igl/write_triangle_mesh.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

namespace app::remesh {

namespace {

using EdgeKey = std::pair<int, int>; // sorted (smaller, larger) wmtk VIDs

inline EdgeKey make_edge_key(int a, int b) {
    return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
}

// Hash functor for std::unordered_{map,set}<EdgeKey, ...>.
// std::pair<int,int> has no default std::hash specialisation.
struct EdgeKeyHash {
    size_t operator()(const EdgeKey& p) const noexcept {
        return std::hash<long long>()(
            ((long long)(unsigned)p.first << 32) | (unsigned)p.second);
    }
};

// Sparse scalar vector indexed by int — used for rows of R_s and columns of P_s.
using SparseVec = std::unordered_map<int, double>;

// Inner product of two sparse vectors. Iterates the smaller one for speed.
inline double sparse_dot(const SparseVec& a, const SparseVec& b) {
    const SparseVec& small = (a.size() <= b.size()) ? a : b;
    const SparseVec& large = (a.size() <= b.size()) ? b : a;
    double s = 0.0;
    for (const auto& kv : small) {
        auto it = large.find(kv.first);
        if (it != large.end()) s += kv.second * it->second;
    }
    return s;
}

// Barycentric weights of q inside triangle (p0, p1, p2).
// Returns true iff q lies (within tolerance) inside the triangle. The weights
// w0, w1, w2 sum to 1 regardless of containment.
inline bool barycentric_inside(
    const Eigen::Vector2d& q,
    const Eigen::Vector2d& p0,
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& p2,
    double& w0, double& w1, double& w2,
    double eps = 1e-9)
{
    const double det = (p1[0] - p0[0]) * (p2[1] - p0[1])
                     - (p2[0] - p0[0]) * (p1[1] - p0[1]);
    if (std::abs(det) < 1e-18) {
        w0 = w1 = w2 = 0.0;
        return false;
    }
    const double inv = 1.0 / det;
    const double qx = q[0] - p0[0];
    const double qy = q[1] - p0[1];
    w1 = (qx * (p2[1] - p0[1]) - (p2[0] - p0[0]) * qy) * inv;
    w2 = ((p1[0] - p0[0]) * qy - qx * (p1[1] - p0[1])) * inv;
    w0 = 1.0 - w1 - w2;
    return (w0 >= -eps && w1 >= -eps && w2 >= -eps);
}

// Snapshot of the current TriMesh as compact V, F + index maps.
struct MeshSnapshot {
    Eigen::MatrixXd V;                            // n_curr × 2
    Eigen::MatrixXi F;                            // m_curr × 3 (compact indices)
    std::vector<int> compact_to_wmtk;             // n_curr → wmtk VID
    std::vector<int> wmtk_to_compact;             // vert_capacity → compact (or -1)
    std::vector<std::vector<int>> v_faces;        // compact vid → adjacent face indices
    std::vector<int> compact_to_fid;              // n_curr_face → stable wmtk FID
};

// Closed-form r ∈ [0, 1] such that cpos ≈ (1-r)*p_a + r*p_b. When cpos is on
// the line through p_a, p_b this is exact; otherwise it returns the orthogonal
// projection's ratio (clamped). Used purely as the (1-r, r) split factor in the
// barycentric reconstruction; off-line cpos still works under this convention.
inline double projection_ratio(
    const Eigen::Vector2d& cpos,
    const Eigen::Vector2d& pa,
    const Eigen::Vector2d& pb)
{
    const Eigen::Vector2d e = pb - pa;
    const double len2 = e.squaredNorm();
    if (len2 < 1e-20) return 0.5;
    double r = (cpos - pa).dot(e) / len2;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    return r;
}

} // anonymous namespace


// ============================================================
//  Public entry
// ============================================================

int EigenEdgeCollapse::simplify_factor_reuse(int target_vertices) {
    using Clock = std::chrono::high_resolution_clock;
    const auto t_total_start = Clock::now();

    const int n_fine  = (int)m_V_fine.rows();
    const int ndof_f  = 2 * n_fine;
    const int nf_free = (int)m_free_fine.size();
    const int vcap0   = (int)vert_capacity();

    if (nf_free == 0) {
        std::cerr << "simplify_factor_reuse: no free fine DOFs\n";
        return 0;
    }

    // ── Pre-computed fine-mesh quantities ────────────────────────────────────
    std::vector<int> dof_full_to_free(ndof_f, -1);
    for (int k = 0; k < nf_free; ++k) dof_full_to_free[m_free_fine[k]] = k;

    Eigen::VectorXd Mff(nf_free);
    for (int k = 0; k < nf_free; ++k) Mff[k] = m_M_diag[m_free_fine[k]];

    // ── Low-rank K_f^{-1} approximation from the standard eigenpencil ──────
    // m_evals_std / m_evecs_std hold (λ_i, φ_i) of  K_ff φ = λ φ  with the
    // eigenvectors L²-orthonormal (φ_i^T φ_j = δ_ij). Then
    //     K_ff^{-1} = Σ_i (1/λ_i) φ_i φ_i^T ≈ Φ_k Λ_k^{-1} Φ_k^T,
    // truncated to the k smallest λ. Applied to W (nf_free × 4):
    //     Z ≈ Φ_k * (Λ_k^{-1} * (Φ_k^T W)).
    //
    // Per-edge cost drops from one sparse backsubstitution to two dense
    // (k × nf_free) matmuls + a k-vector diagonal scale.  Truncation index is
    // m_p.cost_modes (capped at how many converged modes we have).
    Eigen::MatrixXd Phi_free;     // nf_free × N
    Eigen::VectorXd inv_lambda;   // N
    {
        const int N_avail = (int)m_evals_std.size();
        const int N_use   = std::min(m_p.cost_modes > 0 ? m_p.cost_modes : N_avail,
                                     N_avail);
        Phi_free.resize(nf_free, N_use);
        inv_lambda.resize(N_use);
        for (int j = 0; j < N_use; ++j) {
            const double lam = m_evals_std[j];
            inv_lambda[j] = std::isfinite(lam) && std::abs(lam) > 1e-30
                ? 1.0 / lam
                : 0.0;
            for (int k = 0; k < nf_free; ++k)
                Phi_free(k, j) = m_evecs_std(m_free_fine[k], j);
        }
        std::cout << "[factor_reuse] using " << N_use
                  << " standard modes for K^{-1} low-rank approximation\n";
    }

    // ── Mean relative error of the spectral-expansion cost vs the exact
    //    sparse-solve cost. Accumulated per edge inside edge_cost; final mean
    //    printed at the end of simplify_factor_reuse.
    long long err_n   = 0;
    double    err_sum = 0.0;

    // ── R_s, P_s state ──────────────────────────────────────────────────────
    // Both are indexed by wmtk VID (which is stable across collapses; the
    // removed endpoint becomes "invalid" but the index space does not shrink).
    // Initially identity on the fine vertices.
    std::vector<SparseVec> R_rows(vcap0);
    std::vector<SparseVec> P_cols(vcap0);
    for (int v = 0; v < n_fine; ++v) {
        R_rows[v].emplace(v, 1.0);
        P_cols[v].emplace(v, 1.0);
    }

    // ── Snapshot helper ─────────────────────────────────────────────────────
    auto build_snapshot = [&]() {
        // One pass each over get_vertices() / get_faces().  The previous version
        // called both three times (twice inside extract_current_mesh + once
        // here) — each call materialises a fresh vector<Tuple>, so this fusion
        // gives roughly a 3× speedup on the snapshot build alone.
        MeshSnapshot s;

        const auto vtuples = get_vertices();
        const int n_vert = (int)vtuples.size();
        s.compact_to_wmtk.reserve(n_vert);
        s.wmtk_to_compact.assign((int)vert_capacity(), -1);
        s.V.resize(n_vert, 2);
        for (int i = 0; i < n_vert; ++i) {
            const int vid = (int)vtuples[i].vid(*this);
            s.compact_to_wmtk.push_back(vid);
            s.wmtk_to_compact[vid] = i;
            s.V.row(i) = vertex_attrs[vid].pos;
        }

        const auto ftuples = get_faces();
        const int n_face = (int)ftuples.size();
        s.F.resize(n_face, 3);
        s.compact_to_fid.reserve(n_face);
        s.v_faces.assign(n_vert, {});
        for (int f = 0; f < n_face; ++f) {
            const auto& t = ftuples[f];
            const auto vs = oriented_tri_vertices(t);
            const int a = s.wmtk_to_compact[(int)vs[0].vid(*this)];
            const int b = s.wmtk_to_compact[(int)vs[1].vid(*this)];
            const int c = s.wmtk_to_compact[(int)vs[2].vid(*this)];
            s.F(f, 0) = a;
            s.F(f, 1) = b;
            s.F(f, 2) = c;
            s.compact_to_fid.push_back((int)t.fid(*this));
            s.v_faces[a].push_back(f);
            s.v_faces[b].push_back(f);
            s.v_faces[c].push_back(f);
        }
        return s;
    };

    // ── Constrained collapse position (mirrors what simplify() does) ────────
    auto compute_cpos = [&](int ci, int cj, const MeshSnapshot& s,
                            const BoundaryInfo& binfo) -> Eigen::Vector2d {
        return constrained_pos(std::min(ci, cj), std::max(ci, cj), s.V, s.F, binfo);
    };

    // ── Per-edge cost evaluator ─────────────────────────────────────────────
    // Returns +inf when the edge is geometrically infeasible (e.g. p_a or p_b
    // is not contained in any post-collapse 1-ring triangle).
    //
    // Outputs the constrained position and the projection ratio so the caller
    // can pass them to collapse_edge later.
    auto edge_cost = [&](int ci, int cj,
                         const MeshSnapshot& s,
                         const BoundaryInfo& binfo,
                         Eigen::Vector2d* out_cpos,
                         double* out_ratio) -> double
    {
        const int n_curr = (int)s.V.rows();
        if (ci < 0 || cj < 0 || ci >= n_curr || cj >= n_curr || ci == cj)
            return std::numeric_limits<double>::infinity();

        const Eigen::Vector2d pa = s.V.row(ci);
        const Eigen::Vector2d pb = s.V.row(cj);

        const Eigen::Vector2d cpos = compute_cpos(ci, cj, s, binfo);
        const double r = projection_ratio(cpos, pa, pb);
        if (out_cpos)  *out_cpos  = cpos;
        if (out_ratio) *out_ratio = r;

        // ── Post-collapse 1-ring tris ───────────────────────────────────────
        // Triangles incident to v_a OR v_b in the current mesh, minus those
        // incident to BOTH (the 1 or 2 tris that disappear in the collapse).
        std::set<int> ring;
        for (int f : s.v_faces[ci]) ring.insert(f);
        for (int f : s.v_faces[cj]) ring.insert(f);

        struct PostTri {
            int w1, w2;                // "other" corners (compact indices)
            Eigen::Vector2d p_w1, p_w2;
        };
        std::vector<PostTri> post_tris;
        post_tris.reserve(ring.size());
        for (int f : ring) {
            const int a = s.F(f, 0), b = s.F(f, 1), c = s.F(f, 2);
            const int n_collapse = (int)(a == ci || a == cj)
                                 + (int)(b == ci || b == cj)
                                 + (int)(c == ci || c == cj);
            if (n_collapse != 1) continue;     // 2 → collapsed face, 0 → impossible
            PostTri pt{};
            int idx = 0;
            for (int k = 0; k < 3; ++k) {
                const int v = s.F(f, k);
                if (v == ci || v == cj) continue;
                if (idx == 0) { pt.w1 = v; pt.p_w1 = s.V.row(v); }
                else          { pt.w2 = v; pt.p_w2 = s.V.row(v); }
                ++idx;
            }
            post_tris.push_back(pt);
        }
        if (post_tris.empty()) return std::numeric_limits<double>::infinity();

        // ── Coverage feasibility (boundary-boundary edges only) ─────────────
        // If any fine-boundary vertex currently bound to a 1-ring face would
        // NOT be inside any post-collapse triangle at this cpos, the collapse
        // would leak coverage.  Mark the candidate infeasible (cost = ∞) so it
        // never gets popped.  Interior / one-boundary edges don't move the
        // coarse boundary, so this check only matters when both endpoints sit
        // on the boundary.
        if (binfo.on_boundary[ci] && binfo.on_boundary[cj]) {
            std::set<int> ring_fbvs;
            for (int f : ring) {
                if (f < 0 || f >= (int)m_coarse_face_fine_bdry_verts.size()) continue;
                for (int fbv : m_coarse_face_fine_bdry_verts[f])
                    ring_fbvs.insert(fbv);
            }
            for (int fbv : ring_fbvs) {
                const Eigen::Vector2d q = m_V_fine.row(fbv);
                bool covered = false;
                for (const auto& pt : post_tris) {
                    double w0, w1, w2;
                    if (barycentric_inside(q, cpos, pt.p_w1, pt.p_w2,
                                           w0, w1, w2, 1e-9)) {
                        covered = true;
                        break;
                    }
                }
                if (!covered)
                    return std::numeric_limits<double>::infinity();
            }
        }

        // ── α^(v): bary weights of p_v in some post-collapse triangle ───────
        // Corners are (v~ at cpos, w1, w2). Returns the post-tri index and the
        // three α weights (α_tilde, α_w1, α_w2).
        auto find_alpha = [&](const Eigen::Vector2d& q,
                              double& a_tilde, double& a_w1, double& a_w2,
                              int& pt_idx) -> bool {
            // First pass: strict containment.
            for (size_t i = 0; i < post_tris.size(); ++i) {
                const auto& pt = post_tris[i];
                if (barycentric_inside(q, cpos, pt.p_w1, pt.p_w2,
                                       a_tilde, a_w1, a_w2, 1e-9)) {
                    pt_idx = (int)i;
                    return true;
                }
            }
            // Second pass: pick the tri with the *largest* min-weight (least
            // negative), i.e. the "closest" to containing q. Bary weights are
            // still computed and used; this stays robust on the boundary.
            double best_min = -std::numeric_limits<double>::infinity();
            int best_i = -1;
            double best_t = 0, best_1 = 0, best_2 = 0;
            for (size_t i = 0; i < post_tris.size(); ++i) {
                const auto& pt = post_tris[i];
                double t0, t1, t2;
                barycentric_inside(q, cpos, pt.p_w1, pt.p_w2, t0, t1, t2, 1.0);
                const double mn = std::min({t0, t1, t2});
                if (mn > best_min) {
                    best_min = mn;
                    best_i = (int)i;
                    best_t = t0; best_1 = t1; best_2 = t2;
                }
            }
            if (best_i < 0) return false;
            a_tilde = best_t; a_w1 = best_1; a_w2 = best_2;
            pt_idx = best_i;
            return true;
        };

        double a_til = 0, a_w1 = 0, a_w2 = 0;
        double b_til = 0, b_w1 = 0, b_w2 = 0;
        int pt_a = -1, pt_b = -1;
        if (!find_alpha(pa, a_til, a_w1, a_w2, pt_a)) return std::numeric_limits<double>::infinity();
        if (!find_alpha(pb, b_til, b_w1, b_w2, pt_b)) return std::numeric_limits<double>::infinity();

        const int avw1 = post_tris[pt_a].w1, avw2 = post_tris[pt_a].w2;
        const int bvw1 = post_tris[pt_b].w1, bvw2 = post_tris[pt_b].w2;

        // ──────────────────────────────────────────────────────────────────
        //  Natural D (axis-separable barycentric reconstruction) — kept for
        //  reference. Below is the physics-aware D^{phys} replacement.
        // ──────────────────────────────────────────────────────────────────
        // ── β^(v): split α_tilde into (1-r, r) at v_a, v_b ──────────────
        std::map<int, double> beta_a, beta_b;
        beta_a[ci] += (1.0 - r) * a_til;
        beta_a[cj] += r * a_til;
        beta_a[avw1] += a_w1;
        beta_a[avw2] += a_w2;
        beta_b[ci] += (1.0 - r) * b_til;
        beta_b[cj] += r * b_til;
        beta_b[bvw1] += b_w1;
        beta_b[bvw2] += b_w2;
        
        // d^(v)_f := (R_s)_{v_a, f} - Σ_w β^(v)_w (R_s)_{w, f}
        auto build_d_scalar = [&](int vc, const std::map<int, double>& beta) {
            SparseVec d;
            const int wmtk_vc = s.compact_to_wmtk[vc];
            for (const auto& kv : R_rows[wmtk_vc])
                d[kv.first] += kv.second;
            for (const auto& [w, weight] : beta) {
                const int wmtk_w = s.compact_to_wmtk[w];
                for (const auto& kv : R_rows[wmtk_w])
                    d[kv.first] -= weight * kv.second;
            }
            return d;
        };
        const SparseVec d_a = build_d_scalar(ci, beta_a);
        const SparseVec d_b = build_d_scalar(cj, beta_b);
        
        // W_free: axis-pure columns (no cross-axis coupling)
        Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nf_free, 4);
        for (const auto& [fv, val] : d_a) {
            if (fv < 0 || fv >= n_fine) continue;
            const int fx = dof_full_to_free[2 * fv];
            const int fy = dof_full_to_free[2 * fv + 1];
            if (fx >= 0) W(fx, 0) = val;
            if (fy >= 0) W(fy, 1) = val;
        }
        for (const auto& [fv, val] : d_b) {
            if (fv < 0 || fv >= n_fine) continue;
            const int fx = dof_full_to_free[2 * fv];
            const int fy = dof_full_to_free[2 * fv + 1];
            if (fx >= 0) W(fx, 2) = val;
            if (fy >= 0) W(fy, 3) = val;
        }

        // // ──────────────────────────────────────────────────────────────────
        // //  Harmonic-extension residual  ρ = A · u_I  +  B · u_O
        // //
        // //  Patch:  I = {v_a, v_b} (4 DOFs)   O = 1-ring of (v_a, v_b) \ I.
        // //  Stiffness blocks (pre-collapse):
        // //      H_II   ∈ ℝ^{4×4},            H_IO ∈ ℝ^{4×|2O|}
        // //  Stiffness blocks (post-collapse, cpos as merged vertex):
        // //      H^post_{vv} ∈ ℝ^{2×2},       H^post_{vO} ∈ ℝ^{2×|2O|}
        // //  Barycentric prolongation slices:
        // //      P_e^v  ∈ ℝ^{4×2}  (weights at v~ for v_a, v_b)
        // //      P_e^O  ∈ ℝ^{4×|2O|} (weights at 1-ring corners for v_a, v_b)
        // //
        // //  Then (with M := (H^post_{vv})^{-1}):
        // //      A = I_4 - P_e^v M (P_e^v)^T H_II
        // //      B = P_e^v M (H^post_{vO} - (P_e^v)^T H_IO) - P_e^O
        // //
        // //  Unlike the previous "1-ring pinned to 0" form this keeps the 1-ring
        // //  at its actual global displacement u_O.  Harmonic extension passes
        // //  rigid bodies through (f_I = 0 in that case → u_~v is the rigid-body
        // //  value at cpos → ρ = 0), so translation and rotation are kernel.
        // //  Internal deformations (f_I ≠ 0) get amplified by M = (H^post_{vv})^{-1},
        // //  so slim triangles produce large residuals.
        // // ──────────────────────────────────────────────────────────────────
        // auto compute_Ke = [&](const Eigen::Vector2d& p0,
        //                       const Eigen::Vector2d& p1,
        //                       const Eigen::Vector2d& p2,
        //                       double E_mat, double nu_mat)
        //                       -> Eigen::Matrix<double, 6, 6>
        // {
        //     const double x1 = p0[0], y1 = p0[1];
        //     const double x2 = p1[0], y2 = p1[1];
        //     const double x3 = p2[0], y3 = p2[1];
        //     const double signed_area2 = (x2 - x1) * (y3 - y1)
        //                               - (x3 - x1) * (y2 - y1);
        //     const double area = 0.5 * std::abs(signed_area2);
        //     if (area <= 1e-14) return Eigen::Matrix<double, 6, 6>::Zero();
        //     Eigen::Vector3d dNdx, dNdy;
        //     dNdx << (y2 - y3), (y3 - y1), (y1 - y2);
        //     dNdy << (x3 - x2), (x1 - x3), (x2 - x1);
        //     dNdx /= signed_area2;
        //     dNdy /= signed_area2;
        //     Eigen::Matrix<double, 3, 6> B = Eigen::Matrix<double, 3, 6>::Zero();
        //     for (int a = 0; a < 3; ++a) {
        //         B(0, 2 * a)     = dNdx[a];
        //         B(1, 2 * a + 1) = dNdy[a];
        //         B(2, 2 * a)     = dNdy[a];
        //         B(2, 2 * a + 1) = dNdx[a];
        //     }
        //     return area * (B.transpose() * elasticity_D(E_mat, nu_mat) * B);
        // };

        // // ── Patch indexing ──────────────────────────────────────────────────
        // // Collect O = 1-ring vertices around (v_a, v_b), excluding v_a, v_b.
        // std::vector<int> O_verts;
        // std::unordered_map<int, int> O_idx;  // compact vid → index in O_verts
        // {
        //     std::set<int> O_set;
        //     for (int f : ring) {
        //         for (int k = 0; k < 3; ++k) {
        //             const int v = s.F(f, k);
        //             if (v != ci && v != cj) O_set.insert(v);
        //         }
        //     }
        //     O_verts.assign(O_set.begin(), O_set.end());
        //     for (int i = 0; i < (int)O_verts.size(); ++i) O_idx[O_verts[i]] = i;
        // }
        // const int nO     = (int)O_verts.size();
        // const int O_dim  = 2 * nO;

        // // Endpoint DOF index in the 4-DOF I block (v_a → 0,1; v_b → 2,3; else -1).
        // auto I_dof = [&](int v_compact, int axis) -> int {
        //     if (v_compact == ci) return axis;       // 0,1
        //     if (v_compact == cj) return 2 + axis;   // 2,3
        //     return -1;
        // };
        // // O DOF index (2*O_idx[v] + axis, or -1).
        // auto O_dof = [&](int v_compact, int axis) -> int {
        //     auto it = O_idx.find(v_compact);
        //     if (it == O_idx.end()) return -1;
        //     return 2 * it->second + axis;
        // };

        // // ── H_II (4×4) and H_IO (4 × O_dim) from old 1-ring (no collapsed faces).
        // Eigen::Matrix4d H_II = Eigen::Matrix4d::Zero();
        // Eigen::MatrixXd H_IO = Eigen::MatrixXd::Zero(4, O_dim);
        // for (int f : ring) {
        //     const int v0 = s.F(f, 0), v1 = s.F(f, 1), v2 = s.F(f, 2);
        //     const int n_collapse = (int)(v0 == ci || v0 == cj)
        //                          + (int)(v1 == ci || v1 == cj)
        //                          + (int)(v2 == ci || v2 == cj);
        //     if (n_collapse == 2) continue;
        //     const Eigen::Vector2d p0 = s.V.row(v0);
        //     const Eigen::Vector2d p1 = s.V.row(v1);
        //     const Eigen::Vector2d p2 = s.V.row(v2);
        //     const double E_mat  = material_E (s.V, s.F.row(f));
        //     const double nu_mat = material_nu(s.V, s.F.row(f));
        //     const auto Ke = compute_Ke(p0, p1, p2, E_mat, nu_mat);

        //     const std::array<int, 3> tv = {v0, v1, v2};
        //     std::array<int, 6> Im, Om;
        //     for (int k = 0; k < 3; ++k) {
        //         Im[2*k]   = I_dof(tv[k], 0);
        //         Im[2*k+1] = I_dof(tv[k], 1);
        //         Om[2*k]   = O_dof(tv[k], 0);
        //         Om[2*k+1] = O_dof(tv[k], 1);
        //     }
        //     for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) {
        //         if (Im[a] >= 0 && Im[b] >= 0) H_II(Im[a], Im[b]) += Ke(a, b);
        //         if (Im[a] >= 0 && Om[b] >= 0) H_IO(Im[a], Om[b]) += Ke(a, b);
        //     }
        // }

        // // ── H^post_{vv} (2×2) and H^post_{vO} (2 × O_dim) from post_tris.
        // Eigen::Matrix2d H_post_vv = Eigen::Matrix2d::Zero();
        // Eigen::MatrixXd H_post_vO = Eigen::MatrixXd::Zero(2, O_dim);
        // for (const auto& pt : post_tris) {
        //     Eigen::MatrixXd V_local(3, 2);
        //     V_local.row(0) = cpos.transpose();
        //     V_local.row(1) = pt.p_w1.transpose();
        //     V_local.row(2) = pt.p_w2.transpose();
        //     const Eigen::Vector3i tri_local(0, 1, 2);
        //     const double E_mat  = material_E (V_local, tri_local);
        //     const double nu_mat = material_nu(V_local, tri_local);
        //     const auto Ke = compute_Ke(cpos, pt.p_w1, pt.p_w2, E_mat, nu_mat);

        //     // Local DOFs in Ke: tv at 0,1; w1 at 2,3; w2 at 4,5.
        //     H_post_vv += Ke.block<2, 2>(0, 0);
        //     const int w1x = O_dof(pt.w1, 0), w1y = O_dof(pt.w1, 1);
        //     const int w2x = O_dof(pt.w2, 0), w2y = O_dof(pt.w2, 1);
        //     for (int a = 0; a < 2; ++a) {
        //         if (w1x >= 0) H_post_vO(a, w1x) += Ke(a, 2);
        //         if (w1y >= 0) H_post_vO(a, w1y) += Ke(a, 3);
        //         if (w2x >= 0) H_post_vO(a, w2x) += Ke(a, 4);
        //         if (w2y >= 0) H_post_vO(a, w2y) += Ke(a, 5);
        //     }
        // }

        // // ── P_e^v (4×2) and P_e^O (4 × O_dim) ───────────────────────────────
        // Eigen::Matrix<double, 4, 2> P_e_v =
        //     Eigen::Matrix<double, 4, 2>::Zero();
        // P_e_v(0, 0) = a_til; P_e_v(1, 1) = a_til;
        // P_e_v(2, 0) = b_til; P_e_v(3, 1) = b_til;

        // Eigen::MatrixXd P_e_O = Eigen::MatrixXd::Zero(4, O_dim);
        // {
        //     // Row v_a (rows 0,1): bary weights at avw1, avw2 (= a_w1, a_w2).
        //     const int avw1x = O_dof(avw1, 0), avw1y = O_dof(avw1, 1);
        //     const int avw2x = O_dof(avw2, 0), avw2y = O_dof(avw2, 1);
        //     if (avw1x >= 0) P_e_O(0, avw1x) += a_w1;
        //     if (avw1y >= 0) P_e_O(1, avw1y) += a_w1;
        //     if (avw2x >= 0) P_e_O(0, avw2x) += a_w2;
        //     if (avw2y >= 0) P_e_O(1, avw2y) += a_w2;
        //     // Row v_b (rows 2,3): bary weights at bvw1, bvw2 (= b_w1, b_w2).
        //     const int bvw1x = O_dof(bvw1, 0), bvw1y = O_dof(bvw1, 1);
        //     const int bvw2x = O_dof(bvw2, 0), bvw2y = O_dof(bvw2, 1);
        //     if (bvw1x >= 0) P_e_O(2, bvw1x) += b_w1;
        //     if (bvw1y >= 0) P_e_O(3, bvw1y) += b_w1;
        //     if (bvw2x >= 0) P_e_O(2, bvw2x) += b_w2;
        //     if (bvw2y >= 0) P_e_O(3, bvw2y) += b_w2;
        // }

        // // ── A (4×4), B (4 × O_dim) ──────────────────────────────────────────
        // Eigen::Matrix4d A_mat = Eigen::Matrix4d::Identity();
        // Eigen::MatrixXd B_mat = -P_e_O;   // fallback when H^post_vv is singular
        // {
        //     const double det_Hvv = H_post_vv.determinant();
        //     if (std::abs(det_Hvv) > 1e-30) {
        //         const Eigen::Matrix2d M = H_post_vv.inverse();
        //         A_mat = Eigen::Matrix4d::Identity()
        //               - P_e_v * M * P_e_v.transpose() * H_II;
        //         B_mat = P_e_v * M * (H_post_vO - P_e_v.transpose() * H_IO) - P_e_O;
        //     }
        // }

        // // ── Pack tildeD = [A | B] ∈ ℝ^{4 × (4 + O_dim)} ────────────────────
        // // Columns are indexed by patch DOFs, in the order:
        // //    [(v_a,x),(v_a,y),(v_b,x),(v_b,y), (O[0],x),(O[0],y), …].
        // Eigen::MatrixXd tildeD(4, 4 + O_dim);
        // tildeD.leftCols<4>() = A_mat;
        // if (O_dim > 0) tildeD.rightCols(O_dim) = B_mat;

        // // ── W (nf_free × 4) ────────────────────────────────────────────────
        // // Column k of W: W[(f, γ), k] = Σ_{v ∈ patch} tildeD[k, (v, γ)] R_s[v, f].
        // // For each patch vertex, multiply its 4 (per axis) coefficients into
        // // the rows of W from the corresponding R_s row.
        // Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nf_free, 4);
        // {
        //     const auto accumulate = [&](int v_compact, int col_x, int col_y) {
        //         const int wmtk_v = s.compact_to_wmtk[v_compact];
        //         const SparseVec& rs_row = R_rows[wmtk_v];
        //         const Eigen::Vector4d cx = tildeD.col(col_x);
        //         const Eigen::Vector4d cy = tildeD.col(col_y);
        //         for (const auto& kv : rs_row) {
        //             const int fv = kv.first;
        //             if (fv < 0 || fv >= n_fine) continue;
        //             const double rs = kv.second;
        //             const int fx = dof_full_to_free[2 * fv];
        //             const int fy = dof_full_to_free[2 * fv + 1];
        //             for (int k = 0; k < 4; ++k) {
        //                 if (fx >= 0) W(fx, k) += cx[k] * rs;
        //                 if (fy >= 0) W(fy, k) += cy[k] * rs;
        //             }
        //         }
        //     };
        //     // I: v_a → tildeD cols 0,1.   v_b → tildeD cols 2,3.
        //     accumulate(ci, 0, 1);
        //     accumulate(cj, 2, 3);
        //     // O: each 1-ring vertex → tildeD cols 4 + 2*i, 4 + 2*i + 1.
        //     for (int i = 0; i < nO; ++i)
        //         accumulate(O_verts[i], 4 + 2*i, 4 + 2*i + 1);
        // }

        // ── Z = K_f^{-1} W  ─────────────────────────────────────────────────
        // // (exact) Four backsubstitutions against the fixed LDLT factor.
        Eigen::MatrixXd Z = m_K_fine_ff_solver.solve(W);
        if (m_K_fine_ff_solver.info() != Eigen::Success)
            return std::numeric_limits<double>::infinity();
        //
        // (approx) Low-rank spectral expansion using the precomputed standard
        // eigenpencil (m_evals_std, m_evecs_std): K_ff^{-1} ≈ Φ_k Λ_k^{-1} Φ_k^T.
        // Two small dense matmuls + a diagonal scale; no per-edge sparse solve.
        // Eigen::MatrixXd Z = Phi_free * (inv_lambda.asDiagonal()
        //                                 * (Phi_free.transpose() * W));

        // ── A = Z^T M_f Z  (4×4) ────────────────────────────────────────────
        Eigen::MatrixXd MZ = Mff.asDiagonal() * Z;
        Eigen::Matrix4d A  = Z.transpose() * MZ;

        // ── G_c^loc ─ block structure: (g_aa I_2, g_ab I_2; g_ba I_2, g_bb I_2)
        const int wmtk_a = s.compact_to_wmtk[ci];
        const int wmtk_b = s.compact_to_wmtk[cj];
        const double g_aa = sparse_dot(P_cols[wmtk_a], P_cols[wmtk_a]);
        const double g_ab = sparse_dot(P_cols[wmtk_a], P_cols[wmtk_b]);
        const double g_bb = sparse_dot(P_cols[wmtk_b], P_cols[wmtk_b]);

        // ── tr(G_c^loc · A) ─────────────────────────────────────────────────
        // Using the block factorisation; trace of cross blocks doubled by symm.
        const double cost = g_aa * (A(0, 0) + A(1, 1))
                          + 2.0 * g_ab * (A(0, 2) + A(1, 3))
                          + g_bb * (A(2, 2) + A(3, 3));

        // ─── BEGIN exact-cost relative-error tracking (comment out for timing) ───
        // Adds one per-edge sparse backsubstitution. Comment out the entire
        // block below (from BEGIN to END) to drop that overhead.
        // {
        //     Eigen::MatrixXd Z_ex = m_K_fine_ff_solver.solve(W);
        //     if (m_K_fine_ff_solver.info() == Eigen::Success) {
        //         Eigen::MatrixXd MZ_ex = Mff.asDiagonal() * Z_ex;
        //         Eigen::Matrix4d A_ex  = Z_ex.transpose() * MZ_ex;
        //         const double cost_ex = g_aa * (A_ex(0, 0) + A_ex(1, 1))
        //                              + 2.0 * g_ab * (A_ex(0, 2) + A_ex(1, 3))
        //                              + g_bb * (A_ex(2, 2) + A_ex(3, 3));
        //         if (std::isfinite(cost) && std::isfinite(cost_ex)
        //             && std::abs(cost_ex) > 1e-300) {
        //             const double rel = std::abs(cost - cost_ex) / std::abs(cost_ex);
        //             #pragma omp atomic
        //             ++err_n;
        //             #pragma omp atomic
        //             err_sum += rel;
        //         }
        //     }
        // }
        // ─── END exact-cost relative-error tracking ─────────────────────────────

        if (!std::isfinite(cost)) return std::numeric_limits<double>::infinity();
        return cost;
    };

    // ── Rebuild P_cols[vid_tilde] from the new 1-ring tris around v~ ─────────
    // Iterates fine vertices contained in the closed 1-ring of v~ in the current
    // mesh, sets P_cols[v~][f] = α_tilde for f in that 1-ring (zero elsewhere).
    auto update_P_at = [&](int vid_tilde, const MeshSnapshot& s) {
        P_cols[vid_tilde].clear();
        const int ct = s.wmtk_to_compact[vid_tilde];
        if (ct < 0) return;

        // bounding box of the closed 1-ring
        double bb_min[2] = { std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::infinity() };
        double bb_max[2] = {-std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()};
        for (int f : s.v_faces[ct]) {
            for (int k = 0; k < 3; ++k) {
                const Eigen::Vector2d p = s.V.row(s.F(f, k));
                bb_min[0] = std::min(bb_min[0], p[0]);
                bb_min[1] = std::min(bb_min[1], p[1]);
                bb_max[0] = std::max(bb_max[0], p[0]);
                bb_max[1] = std::max(bb_max[1], p[1]);
            }
        }
        const double eps = 1e-10;
        bb_min[0] -= eps; bb_min[1] -= eps;
        bb_max[0] += eps; bb_max[1] += eps;

        // For each fine vertex in the bbox, find its containing 1-ring tri and
        // record the barycentric weight at v~.
        for (int fv = 0; fv < n_fine; ++fv) {
            const Eigen::Vector2d q = m_V_fine.row(fv);
            if (q[0] < bb_min[0] || q[0] > bb_max[0] ||
                q[1] < bb_min[1] || q[1] > bb_max[1]) continue;

            for (int f : s.v_faces[ct]) {
                const int a = s.F(f, 0), b = s.F(f, 1), c = s.F(f, 2);
                // Order corners so the v~ corner comes first → its bary weight
                // is w0.
                int idx_tilde = -1;
                if      (a == ct) idx_tilde = 0;
                else if (b == ct) idx_tilde = 1;
                else if (c == ct) idx_tilde = 2;
                if (idx_tilde < 0) continue;
                const int c1 = (idx_tilde + 1) % 3;
                const int c2 = (idx_tilde + 2) % 3;
                const Eigen::Vector2d p0 = s.V.row(s.F(f, idx_tilde));
                const Eigen::Vector2d p1 = s.V.row(s.F(f, c1));
                const Eigen::Vector2d p2 = s.V.row(s.F(f, c2));
                double w0, w1, w2;
                if (barycentric_inside(q, p0, p1, p2, w0, w1, w2, 1e-9)) {
                    if (w0 > 1e-15) P_cols[vid_tilde].emplace(fv, w0);
                    break;
                }
            }
        }
    };

    // ── Update R_rows[vid_tilde] = (1-r) R_rows[vid_i] + r R_rows[vid_j] ────
    auto update_R_at = [&](int vid_tilde, int vid_i, int vid_j, double r) {
        SparseVec merged;
        if (vid_i != vid_tilde && vid_j != vid_tilde) {
            // Neither parent IS the merged vertex — accumulate fresh.
            for (const auto& kv : R_rows[vid_i]) merged[kv.first] += (1.0 - r) * kv.second;
            for (const auto& kv : R_rows[vid_j]) merged[kv.first] += r * kv.second;
        } else {
            // The merged vertex re-uses one parent's VID; compute into a temp
            // first to avoid trashing it mid-iteration.
            SparseVec acc;
            for (const auto& kv : R_rows[vid_i]) acc[kv.first] += (1.0 - r) * kv.second;
            for (const auto& kv : R_rows[vid_j]) acc[kv.first] += r * kv.second;
            merged.swap(acc);
        }
        R_rows[vid_tilde] = std::move(merged);
    };

    // ── Initial setup ───────────────────────────────────────────────────────
    // (The coarse–fine overlap is built further below, after fbv_per_fid is set
    //  up; both the boundary and overlap states are kept VID/FID-indexed so
    //  that subsequent collapses can update them in O(1-ring) instead of O(mesh).)
    MeshSnapshot snap = build_snapshot();
    BoundaryInfo binfo = classify_boundary(snap.V, snap.F);

    // ── Cascade-MG prolongation tracking ───────────────────────────────────
    // For each level transition C_l → C_{l+1}, track the containing FID of
    // every C_l vertex through the evolving coarse mesh.  Updates are
    // 1-ring-local (fold-robust via topological restriction).  At stage end
    // (= bundle threshold or final), barycentric weights are read off from
    // the final coarse rest positions and stored as a triplet matrix.
    struct CascadeLevel {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        std::vector<int> compact_to_wmtk;
        std::vector<int> wmtk_to_compact;
        std::vector<int> compact_to_fid;
    };
    struct CascadeStage {
        int n_fine = 0;                                    // |C_l|
        Eigen::MatrixXd fine_pos;                          // n_fine × 2 (positions at stage start)
        std::vector<int> fine_to_fid;                      // current FID per fine vertex (-1 if unassigned)
        std::unordered_map<int, std::vector<int>> fid_to_fines;
    };
    struct CascadeStageResult {
        int level_l = 0;                                   // finer level index
        int n_l = 0;
        int n_lp1 = 0;
        std::vector<std::tuple<int, int, double>> triplets; // (row=fine_compact, col=coarse_compact, val=weight)
    };

    const bool track_cascade = !m_cascade_prefix.empty();
    std::vector<CascadeLevel> cascade_levels;
    CascadeStage cascade_stage;
    std::vector<CascadeStageResult> cascade_results;

    auto cascade_capture_level = [&]() {
        if (!track_cascade) return;
        CascadeLevel L;
        L.V = snap.V;
        L.F = snap.F;
        L.compact_to_wmtk = snap.compact_to_wmtk;
        L.wmtk_to_compact = snap.wmtk_to_compact;
        L.compact_to_fid  = snap.compact_to_fid;
        cascade_levels.push_back(std::move(L));
    };

    auto cascade_start_stage = [&]() {
        if (!track_cascade) return;
        cascade_stage.n_fine  = (int)snap.V.rows();
        cascade_stage.fine_pos = snap.V;
        cascade_stage.fine_to_fid.assign(cascade_stage.n_fine, -1);
        cascade_stage.fid_to_fines.clear();
        // Assign each C_l vertex to any incident face (corner case — bary will
        // be (1,0,0) at this stage).
        for (int ci = 0; ci < cascade_stage.n_fine; ++ci) {
            if (snap.v_faces[ci].empty()) continue;
            const int fc  = snap.v_faces[ci][0];
            const int fid = snap.compact_to_fid[fc];
            cascade_stage.fine_to_fid[ci] = fid;
            cascade_stage.fid_to_fines[fid].push_back(ci);
        }
    };

    // Local point-location helper used by both stage init / collapse updates
    // and end-of-stage weight readout.  Allows negative weights (extrapolation)
    // when no face contains the point.
    auto bary_in_face = [&](const Eigen::Vector2d& p, int fc,
                            double& w0, double& w1, double& w2) -> bool {
        const Eigen::Vector2d a  = snap.V.row(snap.F(fc, 0));
        const Eigen::Vector2d b  = snap.V.row(snap.F(fc, 1));
        const Eigen::Vector2d c  = snap.V.row(snap.F(fc, 2));
        const double det = (b[0] - a[0]) * (c[1] - a[1])
                         - (c[0] - a[0]) * (b[1] - a[1]);
        if (std::abs(det) < 1e-18) { w0 = w1 = w2 = 0.0; return false; }
        const double inv = 1.0 / det;
        const double qx  = p[0] - a[0];
        const double qy  = p[1] - a[1];
        w1 = (qx * (c[1] - a[1]) - (c[0] - a[0]) * qy) * inv;
        w2 = ((b[0] - a[0]) * qy - qx * (b[1] - a[1])) * inv;
        w0 = 1.0 - w1 - w2;
        return true;
    };

    auto cascade_finalize_stage = [&](int level_l_idx) {
        if (!track_cascade) return;
        CascadeStageResult res;
        res.level_l = level_l_idx;
        res.n_l     = cascade_stage.n_fine;
        res.n_lp1   = (int)snap.V.rows();

        // wmtk FID → compact face index map in current (coarse) snap.
        std::unordered_map<int, int> fid_to_fc;
        for (int fc = 0; fc < (int)snap.compact_to_fid.size(); ++fc)
            fid_to_fc[snap.compact_to_fid[fc]] = fc;

        for (int fi = 0; fi < cascade_stage.n_fine; ++fi) {
            const int fid = cascade_stage.fine_to_fid[fi];
            if (fid < 0) continue;
            const auto it = fid_to_fc.find(fid);
            if (it == fid_to_fc.end()) continue;
            const int fc = it->second;

            const Eigen::Vector2d p = cascade_stage.fine_pos.row(fi);
            double w0, w1, w2;
            if (!bary_in_face(p, fc, w0, w1, w2)) continue;

            res.triplets.emplace_back(fi, snap.F(fc, 0), w0);
            res.triplets.emplace_back(fi, snap.F(fc, 1), w1);
            res.triplets.emplace_back(fi, snap.F(fc, 2), w2);
        }
        cascade_results.push_back(std::move(res));
    };

    // Capture level 0 = finest, then start stage 0.
    cascade_capture_level();
    cascade_start_stage();

    // ── Persistent VID-indexed edge-face count, used to update binfo
    //    incrementally after each collapse without re-scanning the whole mesh.
    //    Seeded once here from the initial snapshot; touched O(1-ring degree)
    //    times per collapse afterwards.
    std::unordered_map<EdgeKey, int, EdgeKeyHash> efc_vid;
    std::unordered_set<EdgeKey, EdgeKeyHash>      bdry_edges_vid;
    std::vector<int>                              bdry_deg_vid(vcap0, 0);

    auto edge_inc = [&](int a, int b) {
        EdgeKey ek = (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
        int& c = efc_vid[ek];
        ++c;
        if (c == 1) {
            bdry_edges_vid.insert(ek);
            ++bdry_deg_vid[ek.first];
            ++bdry_deg_vid[ek.second];
        } else if (c == 2) {
            bdry_edges_vid.erase(ek);
            --bdry_deg_vid[ek.first];
            --bdry_deg_vid[ek.second];
        }
    };
    auto edge_dec = [&](int a, int b) {
        EdgeKey ek = (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
        auto it = efc_vid.find(ek);
        if (it == efc_vid.end()) return;
        --it->second;
        if (it->second == 1) {
            bdry_edges_vid.insert(ek);
            ++bdry_deg_vid[ek.first];
            ++bdry_deg_vid[ek.second];
        } else if (it->second == 0) {
            bdry_edges_vid.erase(ek);
            --bdry_deg_vid[ek.first];
            --bdry_deg_vid[ek.second];
            efc_vid.erase(it);
        }
    };
    auto face_add = [&](int v0, int v1, int v2) {
        edge_inc(v0, v1); edge_inc(v1, v2); edge_inc(v2, v0);
    };
    auto face_remove = [&](int v0, int v1, int v2) {
        edge_dec(v0, v1); edge_dec(v1, v2); edge_dec(v2, v0);
    };

    // Project the VID-indexed state into a compact-indexed BoundaryInfo
    // matching classify_boundary()'s output for the given snapshot.
    auto make_binfo = [&](const MeshSnapshot& s) -> BoundaryInfo {
        BoundaryInfo info;
        const int nc = (int)s.V.rows();
        info.on_boundary.assign(nc, false);
        info.on_corner.assign(nc, false);

        if (!m_p.general_mesh) {
            const double xmin = s.V.col(0).minCoeff();
            const double xmax = s.V.col(0).maxCoeff();
            const double ymin = s.V.col(1).minCoeff();
            const double ymax = s.V.col(1).maxCoeff();
            const double span = std::max({xmax - xmin, ymax - ymin, 1.0});
            const double eps  = m_p.boundary_tol * span;
            for (int c = 0; c < nc; ++c) {
                const bool onL = std::abs(s.V(c, 0) - xmin) <= eps;
                const bool onR = std::abs(s.V(c, 0) - xmax) <= eps;
                const bool onB = std::abs(s.V(c, 1) - ymin) <= eps;
                const bool onT = std::abs(s.V(c, 1) - ymax) <= eps;
                info.on_boundary[c] = onL || onR || onB || onT;
                info.on_corner[c]   = (onL || onR) && (onB || onT);
            }
        } else {
            for (int c = 0; c < nc; ++c) {
                const int v   = s.compact_to_wmtk[c];
                const int deg = (v >= 0 && v < (int)bdry_deg_vid.size())
                                ? bdry_deg_vid[v] : 0;
                info.on_boundary[c] = (deg > 0);
                info.on_corner[c]   = (deg > 0 && deg != 2);
            }
        }
        for (const auto& ek : bdry_edges_vid) {
            if (ek.first  >= (int)s.wmtk_to_compact.size()) continue;
            if (ek.second >= (int)s.wmtk_to_compact.size()) continue;
            const int c1 = s.wmtk_to_compact[ek.first];
            const int c2 = s.wmtk_to_compact[ek.second];
            if (c1 >= 0 && c2 >= 0)
                info.boundary_edges.insert({std::min(c1, c2), std::max(c1, c2)});
        }
        return info;
    };

    // Seed efc_vid (and friends) from the initial snapshot.
    for (int f = 0; f < (int)snap.F.rows(); ++f) {
        face_add(snap.compact_to_wmtk[snap.F(f, 0)],
                 snap.compact_to_wmtk[snap.F(f, 1)],
                 snap.compact_to_wmtk[snap.F(f, 2)]);
    }

    // ── Persistent FID-indexed coarse-face → fine-boundary-verts map ───────
    // Updated incrementally per collapse on the closed 1-ring of v~. Projected
    // back to the compact-indexed `m_coarse_face_fine_bdry_verts` after each
    // collapse so callers (enclosure_pos) see the same API.
    std::vector<std::vector<int>> fbv_per_fid((int)tri_capacity());

    // Test whether `p` falls inside the compact triangle `c` of `s`, with the
    // same tolerance / sign convention as rebuild_coarse_fine_overlap().
    auto fbv_inside_face = [&](const MeshSnapshot& s, int c,
                               const Eigen::Vector2d& p) -> bool {
        const double eps = 1e-10;
        const Eigen::Vector2d a  = s.V.row(s.F(c, 0));
        const Eigen::Vector2d b  = s.V.row(s.F(c, 1));
        const Eigen::Vector2d cc = s.V.row(s.F(c, 2));
        // bbox prefilter
        const double xmin = std::min({a[0], b[0], cc[0]}) - eps;
        const double xmax = std::max({a[0], b[0], cc[0]}) + eps;
        const double ymin = std::min({a[1], b[1], cc[1]}) - eps;
        const double ymax = std::max({a[1], b[1], cc[1]}) + eps;
        if (p[0] < xmin || p[0] > xmax || p[1] < ymin || p[1] > ymax) return false;
        const double den = (b - a)[0] * (cc - a)[1] - (cc - a)[0] * (b - a)[1];
        if (std::abs(den) < 1e-14) return false;
        const double inv = 1.0 / den;
        const Eigen::Vector2d v2 = p - a;
        const double u  = (v2[0] * (cc - a)[1] - (cc - a)[0] * v2[1]) * inv;
        const double v_ = ((b - a)[0] * v2[1] - v2[0] * (b - a)[1]) * inv;
        const double w  = 1.0 - u - v_;
        return (u >= -eps && v_ >= -eps && w >= -eps);
    };

    // Project FID-indexed state into the compact-indexed member.  O(m_curr)
    // shallow vector copies, no geometry tests.
    auto project_overlap = [&](const MeshSnapshot& s) {
        const int nc = (int)s.F.rows();
        m_coarse_face_fine_bdry_verts.assign(nc, {});
        for (int c = 0; c < nc; ++c) {
            const int fid = s.compact_to_fid[c];
            if (fid >= 0 && fid < (int)fbv_per_fid.size())
                m_coarse_face_fine_bdry_verts[c] = fbv_per_fid[fid];
        }
    };

    // Initial population: full O(|fbv| × m_curr) scan, same logic as
    // rebuild_coarse_fine_overlap, but indexed by FID.
    for (int fbv : m_fine_boundary_verts) {
        const Eigen::Vector2d p = m_V_fine.row(fbv);
        for (int c = 0; c < (int)snap.F.rows(); ++c) {
            if (fbv_inside_face(snap, c, p))
                fbv_per_fid[snap.compact_to_fid[c]].push_back(fbv);
        }
    }
    project_overlap(snap);

    // Edge identification: stable wmtk VID pair (sorted).
    std::map<EdgeKey, double>          cur_cost;          // current cost per edge
    std::map<EdgeKey, Eigen::Vector2d> cur_pos;           // cached cpos per edge
    std::map<EdgeKey, double>          cur_ratio;         // cached r per edge

    using PQEntry = std::tuple<double, int, int>;        // (cost, va_vid, vb_vid)
    auto pq_cmp = [](const PQEntry& a, const PQEntry& b) {
        return std::get<0>(a) > std::get<0>(b);          // min-heap
    };
    std::priority_queue<PQEntry, std::vector<PQEntry>, decltype(pq_cmp)> pq(pq_cmp);

    auto enqueue_edge = [&](int va_vid, int vb_vid) {
        const int ci = snap.wmtk_to_compact[va_vid];
        const int cj = snap.wmtk_to_compact[vb_vid];
        if (ci < 0 || cj < 0) return;

        // Reject corner–corner edges (consistent with simplify()).
        if (vertex_attrs[va_vid].freeze && vertex_attrs[vb_vid].freeze) return;

        // Reject boundary-interior-crossing edges on general meshes
        // (consistent with simplify()).
        if (binfo.on_boundary[ci] && binfo.on_boundary[cj]) {
            auto key = std::make_pair(std::min(ci, cj), std::max(ci, cj));
            if (!binfo.boundary_edges.count(key)) return;
        }

        // wmtk's topological link condition.  Filter here so the priority
        // queue never holds edges that collapse_edge() would reject.
        // tuple_from_edge() and check_link_condition() are read-only on the
        // connectivity and safe to call from parallel enqueue_edge calls.
        {
            std::array<size_t, 2> vids{(size_t)va_vid, (size_t)vb_vid};
            auto [e_tuple, fid_unused] = tuple_from_edge(vids);
            (void)fid_unused;
            if (!e_tuple.is_valid(*this))       return;
            if (!check_link_condition(e_tuple)) return;
        }

        Eigen::Vector2d cpos;
        double ratio;
        // edge_cost only reads shared state (R_rows / P_cols / snap / Phi_free
        // / m_K_fine_ff_solver) — safe to call concurrently from many threads.
        const double c = edge_cost(ci, cj, snap, binfo, &cpos, &ratio);
        if (!std::isfinite(c)) return;

        const EdgeKey ek = make_edge_key(va_vid, vb_vid);
        // Serial section: std::map and std::priority_queue are not thread-safe.
        #pragma omp critical(fr_pq)
        {
            cur_cost[ek]  = c;
            cur_pos[ek]   = cpos;
            cur_ratio[ek] = ratio;
            pq.emplace(c, ek.first, ek.second);
        }
    };

    {
        const auto t0 = Clock::now();
        const auto edges_init = get_edges();
        #pragma omp parallel for
        for (int i = 0; i < (int)edges_init.size(); ++i) {
            const auto& et = edges_init[i];
            const int vi = (int)et.vid(*this);
            const int vj = (int)et.switch_vertex(*this).vid(*this);
            enqueue_edge(vi, vj);
        }
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        std::cout << "[factor_reuse] initial queue: " << cur_cost.size()
                  << " edges in " << dt << " s\n";
        if (err_n > 0) {
            std::cout << "[factor_reuse] initial mean rel-err vs exact: "
                      << (err_sum / (double)err_n)
                      << "  (n=" << err_n << ")\n";
            err_n = 0;
            err_sum = 0.0;
        }
    }

    // ── Per-phase timing accumulators (averaged at the end) ─────────────────
    //   t_pop      pop cheapest + stale skip from the priority queue
    //   t_tuple    tuple_from_edge() + Tuple is_valid() check
    //   t_png      edge-cost PNG render & save
    //   t_collapse wmtk collapse_edge() call
    //   t_snap     snapshot rebuild (extract_current_mesh + index maps + v_faces)
    //   t_bdry     classify_boundary
    //   t_ovlp     rebuild_coarse_fine_overlap
    //   t_rp       update R_s row + P_s column at v~
    //   t_ring     collect closed 1-ring edges + erase stale entries
    //   t_reeval   parallel 1-ring re-scoring
    //   t_io       per-step OBJ + energy SVG output
    long long ns_pop      = 0;
    long long ns_tuple    = 0;
    long long ns_png      = 0;
    long long ns_collapse = 0;
    long long ns_snap     = 0;
    long long ns_bdry     = 0;
    long long ns_ovlp     = 0;
    long long ns_rp       = 0;
    long long ns_ring     = 0;
    long long ns_reeval   = 0;
    long long ns_io       = 0;
    auto since = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };

    // ── Intermediate-bundle threshold sequence ─────────────────────────────
    // Geometric sequence base, 4·base, 16·base, ...  We track the *next*
    // (largest remaining) threshold to fire; after each save it is divided by
    // 4 until it falls below `base` or below the target.
    int next_bundle_threshold = 0;
    if (m_bundle_base > 0 && !m_bundle_prefix_base.empty()) {
        const int v_init = (int)get_vertices().size();
        int t = m_bundle_base;
        while ((long long)t * 4 < v_init) t *= 4;
        if (t < v_init && t > target_vertices) next_bundle_threshold = t;
    }

    // ── Main loop ───────────────────────────────────────────────────────────
    int collapses = 0;
    while ((int)get_vertices().size() > target_vertices && !pq.empty()) {
        const auto t_pop0 = Clock::now();
        // Pop cheapest non-stale entry.
        double cost = std::numeric_limits<double>::infinity();
        int va_vid = -1, vb_vid = -1;
        bool found = false;
        while (!pq.empty()) {
            auto [c, a, b] = pq.top();
            pq.pop();
            const EdgeKey ek{a, b};
            auto it = cur_cost.find(ek);
            if (it == cur_cost.end()) continue;
            // Stale (re-inserted with a different cost).
            if (std::abs(it->second - c) > 1e-12 * std::max(std::abs(c), 1.0)) continue;
            // Endpoints still alive?
            if (a >= (int)snap.wmtk_to_compact.size() ||
                b >= (int)snap.wmtk_to_compact.size() ||
                snap.wmtk_to_compact[a] < 0 || snap.wmtk_to_compact[b] < 0) {
                cur_cost.erase(it); continue;
            }

            // ── Lazy re-verification for boundary edges ────────────────────
            // The cached cpos / cost for a boundary edge can go stale when a
            // distant collapse reshapes its 1-ring (cpos depends on
            // enclosure_pos, which reads the surrounding geometry). Recompute
            // edge_cost on the current state; if the refreshed cost is still
            // ≤ the next entry in the queue, accept it. Otherwise push the
            // updated value back and pop another candidate.
            const int ci = snap.wmtk_to_compact[a];
            const int cj = snap.wmtk_to_compact[b];
            if (binfo.on_boundary[ci] && binfo.on_boundary[cj]) {
                Eigen::Vector2d new_cpos;
                double          new_ratio;
                const double new_c = edge_cost(ci, cj, snap, binfo,
                                               &new_cpos, &new_ratio);
                if (!std::isfinite(new_c)) {
                    cur_cost.erase(it);
                    cur_pos.erase(ek);
                    cur_ratio.erase(ek);
                    continue;
                }
                cur_cost[ek]  = new_c;
                cur_pos[ek]   = new_cpos;
                cur_ratio[ek] = new_ratio;

                const double next_top =
                    pq.empty() ? std::numeric_limits<double>::infinity()
                               : std::get<0>(pq.top());
                if (new_c > next_top) {
                    // Some other edge in the queue is (claimed) cheaper.
                    // Re-queue this one and try the next.
                    pq.emplace(new_c, ek.first, ek.second);
                    continue;
                }
                // Still the cheapest — fall through with refreshed cost.
                cost = new_c; va_vid = a; vb_vid = b; found = true;
                break;
            }

            cost = c; va_vid = a; vb_vid = b; found = true;
            break;
        }
        if (!found) break;

        const EdgeKey ek_top = make_edge_key(va_vid, vb_vid);
        m_pending_pos = cur_pos[ek_top];
        ns_pop += since(t_pop0, Clock::now());

        // ── Pre-collapse edge-cost PNG ──────────────────────────────────────
        // Mirrors simplify(): every cached edge → (compact_vi, compact_vj),
        // sorted ascending by cost so the popped edge sits at index 0.
        const auto t_png0 = Clock::now();
        // if (!m_output_dir.empty()) {
        //     std::vector<std::pair<int, int>> edge_pairs;
        //     std::vector<double> edge_costs;
        //     edge_pairs.reserve(cur_cost.size());
        //     edge_costs.reserve(cur_cost.size());

        //     int best_idx = -1;
        //     for (const auto& [ek, c] : cur_cost) {
        //         const int ca = snap.wmtk_to_compact[ek.first];
        //         const int cb = snap.wmtk_to_compact[ek.second];
        //         if (ca < 0 || cb < 0) continue;
        //         if (ek == ek_top) best_idx = (int)edge_pairs.size();
        //         edge_pairs.emplace_back(ca, cb);
        //         edge_costs.push_back(c);
        //     }
        //     // Sort ascending by cost; track which index becomes the popped edge.
        //     std::vector<int> order(edge_pairs.size());
        //     std::iota(order.begin(), order.end(), 0);
        //     std::sort(order.begin(), order.end(),
        //               [&](int a, int b) { return edge_costs[a] < edge_costs[b]; });
        //     std::vector<std::pair<int, int>> sorted_pairs;
        //     std::vector<double> sorted_costs;
        //     sorted_pairs.reserve(order.size());
        //     sorted_costs.reserve(order.size());
        //     int sorted_best = 0;
        //     for (size_t k = 0; k < order.size(); ++k) {
        //         if (order[k] == best_idx) sorted_best = (int)k;
        //         sorted_pairs.push_back(edge_pairs[order[k]]);
        //         sorted_costs.push_back(edge_costs[order[k]]);
        //     }
        //     save_edge_cost_png(collapses, snap.V, snap.F,
        //                        sorted_pairs, sorted_costs, sorted_best);
        // }
        const auto t_png1 = Clock::now();
        ns_png += since(t_png0, t_png1);

        // Resolve a wmtk Tuple for this edge.
        const auto t_tuple0 = Clock::now();
        std::array<size_t, 2> vids{(size_t)va_vid, (size_t)vb_vid};
        auto [e_tuple, fid] = tuple_from_edge(vids);
        (void)fid;
        const bool tuple_ok = e_tuple.is_valid(*this);
        ns_tuple += since(t_tuple0, Clock::now());
        if (!tuple_ok) {
            cur_cost.erase(ek_top);
            cur_pos.erase(ek_top);
            cur_ratio.erase(ek_top);
            continue;
        }

        const auto t_col0 = Clock::now();
        std::vector<Tuple> new_tris;
        const bool ok = collapse_edge(e_tuple, new_tris);
        ns_collapse += since(t_col0, Clock::now());
        if (!ok) {
            // TriMesh rejected (corner-corner, link cond., face flip, etc.).
            // Drop this edge entirely; the queue will pick the next.
            cur_cost.erase(ek_top);
            cur_pos.erase(ek_top);
            cur_ratio.erase(ek_top);
            continue;
        }

        ++collapses;
        m_energy_log.push_back(cost);

        // Save old-affected face VID triples + FIDs (snap is still pre-collapse).
        // The 1-rings of va and vb cover every triangle that disappears or
        // changes label during the collapse — the delta needed to patch
        // efc_vid / bdry_edges_vid / bdry_deg_vid and fbv_per_fid below.
        const auto t_bdry_save0 = Clock::now();
        std::vector<std::array<int, 3>> old_affected_vid;
        std::vector<int>                old_ring_fids;
        {
            const int ci_old = snap.wmtk_to_compact[va_vid];
            const int cj_old = snap.wmtk_to_compact[vb_vid];
            std::set<int> tris;
            if (ci_old >= 0) for (int f : snap.v_faces[ci_old]) tris.insert(f);
            if (cj_old >= 0) for (int f : snap.v_faces[cj_old]) tris.insert(f);
            old_affected_vid.reserve(tris.size());
            old_ring_fids.reserve(tris.size());
            for (int f : tris) {
                old_affected_vid.push_back({
                    snap.compact_to_wmtk[snap.F(f, 0)],
                    snap.compact_to_wmtk[snap.F(f, 1)],
                    snap.compact_to_wmtk[snap.F(f, 2)]
                });
                old_ring_fids.push_back(snap.compact_to_fid[f]);
            }
        }
        ns_bdry += since(t_bdry_save0, Clock::now());

        // Identify the merged VID (one of the two parents).
        // After collapse, exactly one of {va_vid, vb_vid} remains valid.
        const auto t_snap0 = Clock::now();
        snap = build_snapshot();
        const auto t_snap1 = Clock::now();
        ns_snap += since(t_snap0, t_snap1);

        int tilde_vid;
        if      (snap.wmtk_to_compact[va_vid] >= 0) tilde_vid = va_vid;
        else if (snap.wmtk_to_compact[vb_vid] >= 0) tilde_vid = vb_vid;
        else {
            std::cerr << "simplify_factor_reuse: merged vertex disappeared?\n";
            break;
        }

        // ── Incremental boundary update (replaces classify_boundary) ───────
        // Remove old-affected face contributions and add new-affected ones.
        // Result is materialised into the compact-indexed `binfo` via
        // make_binfo().
        const auto t_bdry0 = Clock::now();
        {
            std::vector<std::array<int, 3>> new_affected_vid;
            const int ct_new = snap.wmtk_to_compact[tilde_vid];
            if (ct_new >= 0) {
                new_affected_vid.reserve(snap.v_faces[ct_new].size());
                for (int f : snap.v_faces[ct_new]) {
                    new_affected_vid.push_back({
                        snap.compact_to_wmtk[snap.F(f, 0)],
                        snap.compact_to_wmtk[snap.F(f, 1)],
                        snap.compact_to_wmtk[snap.F(f, 2)]
                    });
                }
            }
            for (const auto& t : old_affected_vid) face_remove(t[0], t[1], t[2]);
            for (const auto& t : new_affected_vid) face_add   (t[0], t[1], t[2]);
            binfo = make_binfo(snap);
        }
        const auto t_bdry1 = Clock::now();
        ns_bdry += since(t_bdry0, t_bdry1);

        // ── Incremental coarse–fine overlap (replaces rebuild_coarse_fine_overlap)
        //
        // 1. Affected FIDs = old 1-ring (collapsed + reshaped) ∪ new 1-ring of v~.
        //    All other FIDs keep their fbv lists untouched (their geometry did
        //    not change, so their containment lists are still correct).
        // 2. Clear the fbv lists for every affected FID.
        // 3. Re-test ALL fine boundary verts against the new 1-ring faces and
        //    bind whichever ones contain each fbv.
        //    Why "all"?  Because the new 1-ring can now cover an fbv that was
        //    previously uncovered (e.g. cpos extended outward) or bound only
        //    to a non-affected face.  Limiting the re-test to fbvs previously
        //    bound to affected faces would silently miss those — and that is
        //    exactly the leak that lets the coarse mesh stop wrapping the fine
        //    boundary over time.
        // 4. Project the FID-indexed state into m_coarse_face_fine_bdry_verts
        //    so enclosure_pos() sees the new compact indexing.
        {
            const int ct_new = snap.wmtk_to_compact[tilde_vid];
            std::vector<int> new_ring_faces;     // compact face indices
            if (ct_new >= 0) new_ring_faces = snap.v_faces[ct_new];

            std::set<int> affected_fids;
            for (int fid : old_ring_fids) affected_fids.insert(fid);
            for (int f : new_ring_faces) affected_fids.insert(snap.compact_to_fid[f]);
            for (int fid : affected_fids) {
                if (fid >= 0 && fid < (int)fbv_per_fid.size())
                    fbv_per_fid[fid].clear();
            }

            for (int fbv : m_fine_boundary_verts) {
                const Eigen::Vector2d p = m_V_fine.row(fbv);
                for (int c : new_ring_faces) {
                    if (fbv_inside_face(snap, c, p))
                        fbv_per_fid[snap.compact_to_fid[c]].push_back(fbv);
                }
            }
            project_overlap(snap);
        }
        ns_ovlp += since(t_bdry1, Clock::now());

        // ── Cascade tracking: 1-ring local re-point-location ───────────────
        // Collect C_l fine vertices currently assigned to any old-affected
        // FID, then re-point-locate each in the new 1-ring fan around v~.
        // Restricting candidates to the local fan ensures we stay on the
        // topologically correct sheet even when the coarsened rest geometry
        // self-intersects elsewhere.
        if (track_cascade) {
            std::vector<int> redistribute;
            for (int fid : old_ring_fids) {
                auto it = cascade_stage.fid_to_fines.find(fid);
                if (it == cascade_stage.fid_to_fines.end()) continue;
                for (int fi : it->second) {
                    redistribute.push_back(fi);
                    cascade_stage.fine_to_fid[fi] = -1;
                }
                cascade_stage.fid_to_fines.erase(it);
            }

            const int ct_new = snap.wmtk_to_compact[tilde_vid];
            if (ct_new >= 0) {
                const auto& fan = snap.v_faces[ct_new];
                const double eps = 1e-9;
                for (int fi : redistribute) {
                    const Eigen::Vector2d p = cascade_stage.fine_pos.row(fi);
                    int    best_fc    = -1;
                    double best_min_w = -std::numeric_limits<double>::infinity();
                    for (int fc : fan) {
                        double w0, w1, w2;
                        if (!bary_in_face(p, fc, w0, w1, w2)) continue;
                        const double mw = std::min({w0, w1, w2});
                        if (mw >= -eps) { best_fc = fc; best_min_w = mw; break; }
                        if (mw > best_min_w) { best_min_w = mw; best_fc = fc; }
                    }
                    if (best_fc >= 0) {
                        const int fid = snap.compact_to_fid[best_fc];
                        cascade_stage.fine_to_fid[fi] = fid;
                        cascade_stage.fid_to_fines[fid].push_back(fi);
                    }
                }
            }
        }

        const double r = cur_ratio[ek_top];

        // ── Maintain R_s, P_s ───────────────────────────────────────────────
        const auto t_rp0 = Clock::now();
        // R_s row at v~ becomes the (1-r, r) blend of the two parent rows.
        update_R_at(tilde_vid, va_vid, vb_vid, r);

        // The non-surviving parent's row/column become dead — clear them so
        // subsequent inner products don't see stale data (and stale memory is
        // freed).
        const int dead_vid = (tilde_vid == va_vid) ? vb_vid : va_vid;
        R_rows[dead_vid].clear();
        P_cols[dead_vid].clear();

        // P_s column at v~ rebuilt from the new 1-ring barycentric rule.
        update_P_at(tilde_vid, snap);
        ns_rp += since(t_rp0, Clock::now());

        // ── Re-evaluate the closed 1-ring of edges around v~ ────────────────
        // Collect their (wmtk VID, wmtk VID) keys, then re-score.
        const auto t_ring0 = Clock::now();
        std::set<EdgeKey> ring_edges;
        Tuple tilde_t = tuple_from_vertex((size_t)tilde_vid);
        for (const auto& ring_t : get_one_ring_tris_for_vertex(tilde_t)) {
            auto vs = oriented_tri_vertices(ring_t);
            for (int k = 0; k < 3; ++k) {
                const int u = (int)vs[k].vid(*this);
                const int v = (int)vs[(k + 1) % 3].vid(*this);
                ring_edges.insert(make_edge_key(u, v));
            }
        }

        // First, drop any cached entry for an edge whose endpoints are stale.
        // Then re-score the survivors.
        for (auto it = cur_cost.begin(); it != cur_cost.end(); ) {
            const int a = it->first.first, b = it->first.second;
            if (a >= (int)snap.wmtk_to_compact.size() ||
                b >= (int)snap.wmtk_to_compact.size() ||
                snap.wmtk_to_compact[a] < 0 || snap.wmtk_to_compact[b] < 0) {
                cur_pos.erase(it->first);
                cur_ratio.erase(it->first);
                it = cur_cost.erase(it);
            } else ++it;
        }

        // Erase stale entries first (sequential — map mutation), then re-score
        // the survivors in parallel.
        std::vector<EdgeKey> ring_list(ring_edges.begin(), ring_edges.end());
        for (const auto& ek : ring_list) {
            cur_cost.erase(ek);
            cur_pos.erase(ek);
            cur_ratio.erase(ek);
        }
        ns_ring += since(t_ring0, Clock::now());

        const auto t_reev0 = Clock::now();
        #pragma omp parallel for
        for (int i = 0; i < (int)ring_list.size(); ++i) {
            enqueue_edge(ring_list[i].first, ring_list[i].second);
        }
        ns_reeval += since(t_reev0, Clock::now());

        // Per-collapse mean relative error (re-evaluated 1-ring only).
        if (err_n > 0) {
            std::cout << "[factor_reuse] step " << collapses
                      << "  mean rel-err vs exact: "
                      << (err_sum / (double)err_n)
                      << "  (n=" << err_n << ")\n";
            err_n = 0;
            err_sum = 0.0;
        }

        // ── Intermediate simulation bundle (4^k · base sequence) ───────────
        // Fires when the live vertex count hits the next threshold in
        // {base, 4·base, 16·base, ...}; after firing, step down by a factor of 4.
        if (next_bundle_threshold > 0) {
            const int v_now = (int)get_vertices().size();
            if (v_now == next_bundle_threshold && v_now > target_vertices) {
                Eigen::VectorXd face_E(snap.F.rows());
                Eigen::VectorXd face_nu(snap.F.rows());
                for (int f = 0; f < (int)snap.F.rows(); ++f) {
                    face_E[f]  = material_E (snap.V, snap.F.row(f));
                    face_nu[f] = material_nu(snap.V, snap.F.row(f));
                }
                std::ostringstream ss;
                ss << m_bundle_prefix_base << "_v"
                   << std::setw(5) << std::setfill('0') << v_now;
                save_simulation_bundle(ss.str(), snap.V, snap.F, face_E, face_nu);

                // ── Cascade: end current stage and start the next ─────────
                // Snap here is the coarser side of this stage AND the finer
                // side of the next.  The captured level becomes both.
                if (track_cascade) {
                    const int prev_level_idx = (int)cascade_levels.size() - 1;
                    cascade_finalize_stage(prev_level_idx);
                    cascade_capture_level();
                    cascade_start_stage();
                }

                const int n = next_bundle_threshold / 4;
                next_bundle_threshold =
                    (n >= m_bundle_base && n > target_vertices) ? n : 0;
            }
        }

        // ── Per-step OBJ + energy SVG ───────────────────────────────────────
        const auto t_io0 = Clock::now();
        // if (!m_output_dir.empty()) {
        //     std::filesystem::create_directories(m_output_dir + "/steps");
        //     std::ostringstream ss;
        //     ss << m_output_dir << "/steps/step_"
        //        << std::setw(4) << std::setfill('0') << collapses << ".obj";
        //     Eigen::MatrixXd V3 = Eigen::MatrixXd::Zero(snap.V.rows(), 3);
        //     V3.leftCols(2) = snap.V;
        //     igl::write_triangle_mesh(ss.str(), V3, snap.F);
        //     save_energy_svg(collapses);
        // }
        ns_io += since(t_io0, Clock::now());

        if ((collapses % 50) == 0) {
            std::cout << "[factor_reuse] step " << collapses
                      << "  verts=" << get_vertices().size()
                      << "  cost=" << cost
                      << "  queue=" << cur_cost.size() << "\n";
        }
    }

    const double t_total = std::chrono::duration<double>(Clock::now() - t_total_start).count();
    std::cout << "[factor_reuse] done. " << collapses << " collapses in "
              << t_total << " s\n";

    // ── Cascade: finalize the last stage (coarsest) and write triplets ─────
    // Mesh OBJs are not written here — they are already covered by the
    // simulation bundle outputs (fine / intermediate / simplified).  The
    // level→vertex-count mapping is printed so callers can pair triplet
    // files with the corresponding bundle OBJs.
    if (track_cascade) {
        const int prev_level_idx = (int)cascade_levels.size() - 1;
        cascade_finalize_stage(prev_level_idx);
        cascade_capture_level();

        const std::string prefix = m_cascade_prefix;
        const auto parent_dir = std::filesystem::path(prefix).parent_path();
        if (!parent_dir.empty())
            std::filesystem::create_directories(parent_dir);

        std::cout << "[cascade] level → vertex count:\n";
        for (int l = 0; l < (int)cascade_levels.size(); ++l)
            std::cout << "          level " << l
                      << " : " << cascade_levels[l].V.rows() << " vertices\n";

        // Triplets per adjacent-pair prolongation P̂_{l+1→l}.
        //   Header line:  "rows cols nnz"
        //   Body lines:   "row col val"
        // rows  = |C_l| = number of finer vertices (= cascade_levels[l].V.rows())
        // cols  = |C_{l+1}| = number of coarser vertices
        for (const auto& res : cascade_results) {
            std::ostringstream ss;
            ss << prefix << "_P_"
               << std::setw(2) << std::setfill('0') << (res.level_l + 1)
               << "_to_"
               << std::setw(2) << std::setfill('0') << res.level_l << ".txt";
            std::ofstream ofs(ss.str());
            ofs << std::setprecision(17);
            ofs << res.n_l << " " << res.n_lp1 << " "
                << res.triplets.size() << "\n";
            for (const auto& [r, c, v] : res.triplets)
                ofs << r << " " << c << " " << v << "\n";
            std::cout << "[cascade] wrote " << ss.str()
                      << "  " << res.n_l << "x" << res.n_lp1
                      << "  nnz=" << res.triplets.size() << "\n";
        }
    }

    // ── Per-collapse timing breakdown ───────────────────────────────────────
    // Totals over the run, averaged over the number of successful collapses.
    // Percentages are relative to (pop + png + collapse + snap + rp + ring +
    // reeval + io); the wall-clock total above also includes setup costs that
    // are not bucketed here.
    if (collapses > 0) {
        struct Bucket { const char* name; long long ns; };
        const Bucket buckets[] = {
            {"pop+skip",       ns_pop},
            {"tuple_from_edge", ns_tuple},
            {"edge-cost PNG",  ns_png},
            {"collapse_edge",  ns_collapse},
            {"snapshot",       ns_snap},
            {"classify_bdry",  ns_bdry},
            {"coarse-fine ovl",ns_ovlp},
            {"update R/P",     ns_rp},
            {"1-ring collect", ns_ring},
            {"1-ring re-eval", ns_reeval},
            {"OBJ + energy",   ns_io},
        };
        long long sum_ns = 0;
        for (const auto& b : buckets) sum_ns += b.ns;

        std::cout << "[factor_reuse] per-collapse timing (avg over "
                  << collapses << " collapses):\n";
        for (const auto& b : buckets) {
            const double mean_ms = b.ns * 1e-6 / collapses;
            const double pct     = sum_ns > 0 ? 100.0 * b.ns / sum_ns : 0.0;
            std::cout << "    " << std::left << std::setw(17) << b.name
                      << "  mean " << std::right << std::setw(9)
                      << std::fixed << std::setprecision(3) << mean_ms
                      << " ms   total " << std::setw(8)
                      << (b.ns * 1e-9) << " s   ("
                      << std::setw(5) << std::setprecision(1) << pct << "%)\n";
        }
        std::cout << "    " << std::left << std::setw(17) << "sum (bucketed)"
                  << "  mean " << std::right << std::setw(9)
                  << std::fixed << std::setprecision(3)
                  << (sum_ns * 1e-6 / collapses)
                  << " ms   total " << std::setw(8)
                  << (sum_ns * 1e-9) << " s\n";
        std::cout << std::defaultfloat;
    }

    return collapses;
}

} // namespace app::remesh
