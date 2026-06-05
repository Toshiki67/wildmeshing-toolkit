#include "EigenEdgeCollapse.hpp"
#include "EigenEdgeCollapseInternal.hpp"

#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Sparse symmetric eigensolver (Spectra is header-only on top of Eigen).
// Spectra 0.6.2 ships its headers at the include root, not under a Spectra/
// subdirectory, so the include path is just the bare filename.
#include <SymEigsSolver.h>

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

namespace {

// ── Spectra operators ───────────────────────────────────────────────────
//
// Two flavours, selected by USE_GENERALIZED_PENCIL below:
//
//   (A)  K_ff φ = λ φ
//        Spectra runs LARGEST_ALGE on K_ff^{-1}. Eigenvectors are unchanged
//        between K_ff and K_ff^{-1}, so no post-transform is needed.
//
//   (B)  K_ff φ = λ M_ff φ
//        Change of variables ψ = M^{1/2} φ symmetrises the pencil:
//           M^{-1/2} K_ff M^{-1/2} ψ = λ ψ.
//        Spectra runs LARGEST_ALGE on the inverse operator
//           K̃^{-1} = M^{1/2} K_ff^{-1} M^{1/2},
//        whose largest eigenvalues are 1/λ for the smallest generalized λ.
//        After the solve we recover φ = M^{-1/2} ψ.
//
// Both operators apply K_ff^{-1} through the already-built LDLT factor
// (m_K_fine_ff_solver), so no extra factorisation is needed.

struct InvK_Op {
    using Scalar = double;

    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>* solver = nullptr;
    int n = 0;

    int rows() const { return n; }
    int cols() const { return n; }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in,  n);
        Eigen::Map<Eigen::VectorXd>       y(y_out, n);
        y = solver->solve(x);
    }
};

struct InvKM_SymOp {
    using Scalar = double;

    const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>* solver = nullptr;
    Eigen::VectorXd M_sqrt;   // sqrt(M_diag[m_free_fine[k]]), length n
    int n = 0;

    int rows() const { return n; }
    int cols() const { return n; }

    void perform_op(const Scalar* x_in, Scalar* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in,  n);
        Eigen::Map<Eigen::VectorXd>       y(y_out, n);
        Eigen::VectorXd t = M_sqrt.cwiseProduct(x);   // M^{1/2} x
        t = solver->solve(t);                          // K_ff^{-1} M^{1/2} x
        y = M_sqrt.cwiseProduct(t);                    // M^{1/2} K_ff^{-1} M^{1/2} x
    }
};

} // anonymous namespace


EigenEdgeCollapse::EigenEdgeCollapse() {
    p_vertex_attrs = &vertex_attrs;
}

EigenEdgeCollapse::~EigenEdgeCollapse() = default;

// ---------------------------------------------------------- helper free functions



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

    // ── Standard symmetric eigenproblem: K_eff φ = λ φ ────────────────────
    const int k_eig = std::min(p.num_modes, nf);
    std::cout << "Computing " << k_eig << " eigenmodes on "
              << nf << " free DOFs (Spectra / sparse)...\n";

    // // ── (legacy) Dense generalized eigensolve, kept for reference ──────
    // Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigs(Kff_d, Mff_d);
    // if (eigs.info() != Eigen::Success)
    //     throw std::runtime_error("Eigenvalue solve failed");
    // m_evals = eigs.eigenvalues().head(k_eig);
    // Eigen::MatrixXd evecs_free = eigs.eigenvectors().leftCols(k_eig); // nf × k_eig

    // Sparse path: smallest λ of K_ff are 1/μ_max of the operator (K_ff^{-1} for
    // the standard problem, M^{1/2} K_ff^{-1} M^{1/2} for the generalized one).
    // Both are SPD → SymEigsSolver with LARGEST_ALGE.  K_ff^{-1} is applied
    // through the already-built LDLT factor (m_K_fine_ff_solver) — no extra
    // factorisation, no dense matrix.
    //
    // We always run BOTH:
    //   • Generalized  (K φ = λ M φ)  → m_evals    / m_evecs    (M-orthonormal)
    //   • Standard     (K φ = λ φ)    → m_evals_std / m_evecs_std (Euclidean)
    //
    // Spectra with LARGEST_ALGE returns descending μ = 1/λ — which is the same
    // as ascending λ, so no flipping is needed.

    // ── (1) Generalized: K φ = λ M φ ─────────────────────────────────────
    Eigen::MatrixXd evecs_free;     // nf × k_eig, used to fill m_evecs below
    {
        const auto t_eig_start = std::chrono::high_resolution_clock::now();

        InvKM_SymOp op;
        op.solver = &m_K_fine_ff_solver;
        op.n      = nf;
        op.M_sqrt.resize(nf);
        for (int k = 0; k < nf; ++k)
            op.M_sqrt[k] = std::sqrt(m_M_diag[m_free_fine[k]]);

        const int ncv    = std::min(nf, std::max(2 * k_eig + 1, k_eig + 10));
        const int max_it = 2000;
        const double tol = 1e-10;

        Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, InvKM_SymOp>
            eigs(&op, k_eig, ncv);
        eigs.init();
        const int nconv = eigs.compute(max_it, tol);
        if (eigs.info() != Spectra::SUCCESSFUL)
            throw std::runtime_error("Spectra eigenvalue solve failed (generalized)");

        const Eigen::VectorXd evals_inv = eigs.eigenvalues();
        const Eigen::MatrixXd evecs_raw = eigs.eigenvectors();
        const int n_have = std::min((int)evals_inv.size(), k_eig);

        Eigen::VectorXd evals_asc(k_eig);
        evecs_free.setZero(nf, k_eig);
        for (int i = 0; i < n_have; ++i) {
            evals_asc[i] = (evals_inv[i] != 0.0)
                ? 1.0 / evals_inv[i]
                : std::numeric_limits<double>::infinity();
            // ψ → φ = M^{-1/2} ψ
            evecs_free.col(i) = evecs_raw.col(i).cwiseQuotient(op.M_sqrt);
        }
        for (int i = n_have; i < k_eig; ++i)
            evals_asc[i] = std::numeric_limits<double>::infinity();

        m_evals = evals_asc;

        const double t_eig = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_eig_start).count();
        std::cout << "[Spectra/gen] K φ = λ M φ: " << nconv << "/" << k_eig
                  << " converged in " << t_eig << " s (ncv=" << ncv << ")\n";
        if (nconv < k_eig)
            std::cerr << "[Spectra/gen] only " << nconv << "/" << k_eig
                      << " eigenvalues converged\n";
    }

    // Expand m_evecs to all DOFs (zero on fixed DOFs).
    m_evecs = Eigen::MatrixXd::Zero(ndof_f, k_eig);
    for (int k = 0; k < nf; ++k)
        m_evecs.row(m_free_fine[k]) = evecs_free.row(k);

    // ── (2) Standard: K φ = λ φ — kept as side-by-side reference ─────────
    // No eig_tol filtering: smallest k_eig eigenvalues are stored as-is,
    // in ascending order, with eigenvectors directly from Spectra (no M-rescale).
    {
        const auto t_eig_start = std::chrono::high_resolution_clock::now();

        InvK_Op op;
        op.solver = &m_K_fine_ff_solver;
        op.n      = nf;

        const int ncv    = std::min(nf, std::max(2 * k_eig + 1, k_eig + 10));
        const int max_it = 2000;
        const double tol = 1e-10;

        Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, InvK_Op>
            eigs(&op, k_eig, ncv);
        eigs.init();
        const int nconv = eigs.compute(max_it, tol);
        if (eigs.info() != Spectra::SUCCESSFUL)
            throw std::runtime_error("Spectra eigenvalue solve failed (standard)");

        const Eigen::VectorXd evals_inv = eigs.eigenvalues();
        const Eigen::MatrixXd evecs_raw = eigs.eigenvectors();
        const int n_have = std::min((int)evals_inv.size(), k_eig);

        Eigen::VectorXd evals_asc(k_eig);
        Eigen::MatrixXd evecs_free_std = Eigen::MatrixXd::Zero(nf, k_eig);
        for (int i = 0; i < n_have; ++i) {
            evals_asc[i] = (evals_inv[i] != 0.0)
                ? 1.0 / evals_inv[i]
                : std::numeric_limits<double>::infinity();
            evecs_free_std.col(i) = evecs_raw.col(i);
        }
        for (int i = n_have; i < k_eig; ++i)
            evals_asc[i] = std::numeric_limits<double>::infinity();

        m_evals_std = evals_asc;
        m_evecs_std = Eigen::MatrixXd::Zero(ndof_f, k_eig);
        for (int k = 0; k < nf; ++k)
            m_evecs_std.row(m_free_fine[k]) = evecs_free_std.row(k);

        const double t_eig = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_eig_start).count();
        std::cout << "[Spectra/std] K φ = λ φ: " << nconv << "/" << k_eig
                  << " converged in " << t_eig << " s (ncv=" << ncv << ")\n";
        if (nconv < k_eig)
            std::cerr << "[Spectra/std] only " << nconv << "/" << k_eig
                      << " eigenvalues converged\n";
    }

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
//  Current mesh + per-face material E (for visualisation)
// ============================================================

void EigenEdgeCollapse::set_intermediate_bundle(
    const std::string& prefix_base, int base_threshold)
{
    m_bundle_prefix_base = prefix_base;
    m_bundle_base        = std::max(0, base_threshold);
}

void EigenEdgeCollapse::set_cascade_output(const std::string& prefix)
{
    m_cascade_prefix = prefix;
}

void EigenEdgeCollapse::get_current_mesh_with_E(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::VectorXd& face_E) const
{
    extract_current_mesh(V, F);
    face_E.resize(F.rows());
    for (int f = 0; f < (int)F.rows(); ++f)
        face_E[f] = material_E(V, F.row(f));
}

void EigenEdgeCollapse::get_current_mesh_with_material(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    Eigen::VectorXd& face_E,
    Eigen::VectorXd& face_nu) const
{
    extract_current_mesh(V, F);
    face_E.resize(F.rows());
    face_nu.resize(F.rows());
    for (int f = 0; f < (int)F.rows(); ++f) {
        face_E[f]  = material_E (V, F.row(f));
        face_nu[f] = material_nu(V, F.row(f));
    }
}

void EigenEdgeCollapse::save_simulation_bundle(
    const std::string& prefix,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& face_E,
    const Eigen::VectorXd& face_nu) const
{
    // ── Mesh as OBJ (z = 0) ──────────────────────────────────────────────
    {
        Eigen::MatrixXd V3 = Eigen::MatrixXd::Zero(V.rows(), 3);
        V3.leftCols(2) = V;
        igl::write_triangle_mesh(prefix + ".obj", V3, F);
    }

    auto dump_vec = [](const std::string& path, const Eigen::VectorXd& v) {
        std::ofstream ofs(path);
        ofs << std::setprecision(17);
        for (int i = 0; i < v.size(); ++i) ofs << v[i] << "\n";
    };
    dump_vec(prefix + "_E.txt",  face_E);
    dump_vec(prefix + "_nu.txt", face_nu);

    // ── Fixed vertex list (zero-displacement Dirichlet) ──────────────────
    // Matches the BC logic in init_from_obj: left boundary is held only when
    // fixed_left=true AND no springs are active.
    {
        std::vector<int> fixed_verts =
            (m_p.spring_k > 0.0) ? std::vector<int>{}
            : (m_p.fixed_left    ? left_boundary_verts(V) : std::vector<int>{});
        std::ofstream ofs(prefix + "_fixed.txt");
        for (int v : fixed_verts) ofs << v << "\n";
    }

    // ── Spring vertex list (fine-mesh vertex indices) ────────────────────
    // Springs are anchored on the fine mesh — the coarse-side contribution is
    // P^T · spring_k · P at simulation time, derivable from spring_k (params)
    // plus the fine-mesh barycentric prolongation onto the coarse mesh in this
    // bundle.  So we only need the fine-mesh vertex indices here; no triplets.
    {
        std::ofstream ofs(prefix + "_spring.txt");
        for (int vf : m_spring_fine_verts) ofs << vf << "\n";
    }

    // ── Scalar problem parameters ────────────────────────────────────────
    {
        std::ofstream ofs(prefix + "_params.txt");
        ofs << std::setprecision(17)
            << "plane_stress=" << (m_p.plane_stress ? "true" : "false") << "\n"
            << "rho="          << m_p.rho                                << "\n"
            << "fixed_left="   << (m_p.fixed_left ? "true" : "false")    << "\n"
            << "alpha="        << m_p.alpha                              << "\n"
            << "spring_k="     << m_p.spring_k                           << "\n";
    }

    std::cout << "Wrote simulation bundle to '" << prefix << "_{obj,E,nu,fixed,spring,params}.txt'\n";
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
//  Output directory
// ============================================================

void EigenEdgeCollapse::set_output_dir(const std::string& dir)
{
    m_output_dir = dir;
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(dir + "/eigenmodes");
    std::filesystem::create_directories(dir + "/edge_costs");
}



} // namespace app::remesh
