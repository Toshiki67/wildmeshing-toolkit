#pragma once

#include <wmtk/TriMesh.h>
#include <wmtk/AttributeCollection.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace app::remesh {

// Forward declarations for Woodbury / CHOLMOD precomputed state.
// Definitions live in the .cpp so <cholmod.h> stays out of the public header.
struct WoodburyBase;
struct KrrPrecomp;
struct KnewPrecomp;

struct VertexAttributes {
    Eigen::Vector2d pos;
    bool freeze = false;
};

// Parameters matching the Python script's command-line arguments
struct CollapseParams {
    double E_left        = 1.0;
    double E_right       = 10000.0;
    double nu            = 0.30;   // fallback when nu_left / nu_right are unset
    double nu_left       = -1.0;   // < 0 → use nu
    double nu_right      = -1.0;   // < 0 → use nu
    double rho           = 1.0;
    int    num_modes     = 60;
    int    cost_modes    = 20;
    double eig_tol       = 1e-8;
    bool   plane_stress  = true;   // false = plane strain
    bool   fixed_left    = true;   // fix left-boundary DOFs
    double boundary_tol  = 1e-8;
    int    target_vertices = -1;   // -1 → floor(n_fine / 2)
    // weight_mode: 0=none, 1=inv_lambda, 2=inv_lambda2
    int    weight_mode   = 0;
    // Kinetic shift: K_eff = K + alpha*M.  Applied when fixed_left=false to
    // regularize the otherwise singular free-free stiffness matrix.
    double alpha         = 0.0;
    // Spring boundary conditions: add spring_k to K's diagonal at every DOF
    // on the left (x ≈ x_min) and right (x ≈ x_max) boundary.
    // When > 0, overrides fixed_left and alpha (all DOFs stay in the system;
    // the springs provide regularisation and effectively fix both ends).
    double spring_k      = 0.0;
    // When true: boundary detection is topological (any edge shared by only one
    // triangle), and boundary–boundary non-boundary-edge collapses are rejected;
    // boundary–boundary boundary-edge positions use QEM.
    bool   general_mesh  = false;
    // OBJ file defining the "left" material region (optional).
    // Triangles whose centroid is inside this region get E_left / nu_left;
    // those outside get E_right / nu_right.
    // When empty the hardcoded analytical boundary is used instead.
    std::string material_obj;
    // Linear gradient: E and nu vary linearly in x from
    //   (E_left, nu_left) at x_min  to  (E_right, nu_right) at x_max.
    // Overrides material_obj when true.
    bool is_gradient = false;

    // Optional: file containing fine-mesh vertex indices (whitespace-separated)
    // to attach springs to. When set, replaces the default left+right boundary
    // spring placement. Spring stiffness is given by spring_k. The contribution
    // to the coarse stiffness matrix is P^T K_spring^fine P, where P is the
    // barycentric prolongation (fine ← coarse).
    std::string spring_fine_verts_file;

    double get_nu_left()  const { return nu_left  >= 0.0 ? nu_left  : nu; }
    double get_nu_right() const { return nu_right >= 0.0 ? nu_right : nu; }
};

class EigenEdgeCollapse : public wmtk::TriMesh {
public:
    wmtk::AttributeCollection<VertexAttributes> vertex_attrs;

    // ------------------------------------------------------------------ init

    EigenEdgeCollapse();
    ~EigenEdgeCollapse(); // defined in .cpp so unique_ptr<WoodburyBase> can be destroyed

    // Load OBJ, assemble fine FEM, compute eigenmodes, init TriMesh topology.
    void init_from_obj(const std::string& path, const CollapseParams& p);

    // -------------------------------------------------------------- main loop

    // Brute-force: for each step evaluate all edges, collapse cheapest.
    // Returns number of collapses performed.
    int simplify(int target_vertices);

    // Greedy edge collapse with fine-factorization reuse.
    //   * L_f = chol(K_f) (already computed in init_from_obj) is reused for every
    //     per-edge cost evaluation.
    //   * Cumulative scalar restriction R_s (n_curr × n_fine) and prolongation
    //     P_s (n_fine × n_curr) are maintained through the collapse history.
    //     Per collapse only one row of R_s and one column of P_s are touched.
    //   * Edges live in a priority queue keyed by cost; after each collapse only
    //     the closed 1-ring of edges around the merged vertex is re-scored.
    //   * Per-edge cost: 4 fine backsubstitutions + two 4×4 matrix builds + trace.
    // Implementation lives in FactorReuseSimplify.cpp.
    int simplify_factor_reuse(int target_vertices);

    // ----------------------------------------------------------------- output

    void write_obj(const std::string& path) const;

    // Set root output directory; sub-folders are created automatically.
    // Must be called before simplify() to enable per-step edge-cost PNGs.
    void set_output_dir(const std::string& dir);

    // Side-by-side PNG for each used eigenmode:
    //   left  = fine mesh deformed by (1/λ_i) φ_i
    //   right = coarse mesh deformed by K_c^{-1} P^T M_f φ_i
    // Saves <dir>/eigenmodes/mode_NNNN.png.
    void save_eigenmode_pngs() const;

    // Save material distribution of the current mesh as an SVG file.
    // Each triangle is filled with a color interpolated between E_left (blue)
    // and E_right (red) according to its averaged material E value.
    void save_material_svg(const std::string& path) const;

    // Save material distribution for an arbitrary (V, F) pair.
    // Used to visualize the fine mesh without going through the TriMesh.
    void save_material_svg(
        const std::string& path,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F) const;

    // Save fine mesh colored by E with m_spring_fine_verts overlaid as red dots.
    // Useful for verifying where user-specified spring constraints are located.
    // No-op if m_spring_fine_verts is empty.
    void save_springs_svg(const std::string& path) const;

    // Programmatically set the fine vertex IDs that have spring constraints.
    // When non-empty, takes precedence over CollapseParams::spring_fine_verts_file
    // in subsequent init_from_obj() calls.
    void set_spring_fine_verts(const std::vector<int>& verts) {
        m_spring_fine_verts = verts;
    }
    const std::vector<int>& spring_fine_verts() const { return m_spring_fine_verts; }

    // Expose fine mesh for external inspection
    const Eigen::MatrixXd& V_fine() const { return m_V_fine; }
    const Eigen::MatrixXi& F_fine() const { return m_F_fine; }
    const Eigen::VectorXd& fine_E()  const { return m_fine_E;  }
    const Eigen::VectorXd& fine_nu() const { return m_fine_nu; }

    // Extract the current (simplified) mesh as dense V, F plus the per-face
    // Young's modulus (area-weighted average projected from the fine mesh).
    // Convenience wrapper around extract_current_mesh() + material_E() so
    // visualisation code (Polyscope, etc.) does not need to reach into the
    // private API.
    void get_current_mesh_with_E(
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& F,
        Eigen::VectorXd& face_E) const;

    // Same as above but also returns the per-face Poisson's ratio.  Needed by
    // downstream simulation (assembly of the elasticity D matrix).
    void get_current_mesh_with_material(
        Eigen::MatrixXd& V,
        Eigen::MatrixXi& F,
        Eigen::VectorXd& face_E,
        Eigen::VectorXd& face_nu) const;

    // Dump a self-contained "simulation bundle" describing one mesh and its
    // FEM problem instance.  Generates the following files under `prefix`:
    //   <prefix>.obj         — mesh as a 2D OBJ (z = 0)
    //   <prefix>_E.txt       — per-face Young's modulus, one value per line
    //   <prefix>_nu.txt      — per-face Poisson's ratio, one value per line
    //   <prefix>_fixed.txt   — list of vertex indices whose (x, y) DOFs are
    //                          held to zero by the boundary condition
    //                          (left-boundary fixed when fixed_left=true and
    //                           no springs are present; empty otherwise)
    //   <prefix>_spring.txt  — fine-mesh vertex indices where springs are
    //                          attached, one per line (same list regardless of
    //                          which mesh this bundle describes).  Stiffness is
    //                          in <prefix>_params.txt as `spring_k`.
    //   <prefix>_params.txt  — scalar problem parameters: plane_stress, rho,
    //                          fixed_left, alpha, spring_k
    // Mesh-agnostic: callers can pass either the fine or the simplified mesh.
    void save_simulation_bundle(
        const std::string& prefix,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const Eigen::VectorXd& face_E,
        const Eigen::VectorXd& face_nu) const;

    // Configure intermediate simulation-bundle saves during
    // simplify_factor_reuse().  Bundles are written along the geometric
    // sequence  base, 4·base, 16·base, …  (`base` = `base_threshold`), in
    // decreasing order, every time the live coarse vertex count hits one of
    // those values and remains strictly above the final target.  Typical use:
    // base_threshold = 4 · target  → saves at 4·t, 16·t, 64·t, … below n_init.
    // Pass base_threshold = 0 to disable.
    void set_intermediate_bundle(const std::string& prefix_base, int base_threshold);

    // Configure cascadic-MG prolongation output for simplify_factor_reuse().
    // Tracks each C_l vertex's containing face throughout the C_l → C_{l+1}
    // stage via 1-ring-local point-location.  At stage end the barycentric
    // weights are read off from the final coarse rest positions.
    //
    // Output files per stage (l = 0..L-1 with level 0 = finest):
    //   <prefix>_level_<L>.obj         — mesh at level L
    //   <prefix>_P_<l+1>_to_<l>.txt    — triplet "row col val" lines for the
    //                                    scalar prolongation  P̂_{l+1→l}
    //                                    (rows index finer compact vertex IDs,
    //                                     cols index coarser compact vertex IDs).
    //                                    The vector prolongation P = P̂ ⊗ I_2
    //                                    is reconstructible from the scalar form.
    // Empty prefix disables the feature.
    void set_cascade_output(const std::string& prefix);

    // ------------------------------------------- TriMesh callback overrides

    bool collapse_edge_before(const Tuple& t) override;
    bool collapse_edge_after(const Tuple& t) override;

private:
    // ---------------------------------------------------------- fine mesh FEM

    Eigen::MatrixXd m_V_fine;  // n_fine × 2
    Eigen::MatrixXi m_F_fine;  // m_fine × 3

    Eigen::VectorXd m_evals;   // k_eig generalized eigenvalues  (K φ = λ M φ)
    Eigen::MatrixXd m_evecs;   // 2*n_fine × k_eig (extended to all DOFs)

    // Side-by-side standard symmetric eigenpencil (K φ = λ φ) — kept all k_eig
    // modes in ascending order (no eig_tol filtering). Available for alternative
    // cost functions that want the K-orthonormal basis instead of the
    // M-orthonormal one.
    Eigen::VectorXd m_evals_std;
    Eigen::MatrixXd m_evecs_std;
    Eigen::VectorXd m_M_diag;  // 2*n_fine lumped mass diagonal
    Eigen::SparseMatrix<double> m_M_fine;

    // Per-element E and nu values on the fine mesh (computed once, used as ground truth)
    Eigen::VectorXd m_fine_E;          // m_fine length
    Eigen::VectorXd m_fine_nu;         // m_fine length
    // Precomputed bounding boxes of fine triangles for fast overlap queries
    Eigen::MatrixXd m_fine_bb_min;     // m_fine × 2
    Eigen::MatrixXd m_fine_bb_max;     // m_fine × 2

    // Material-region OBJ (loaded when CollapseParams::material_obj is non-empty).
    // Triangles whose centroid falls inside this region get E_left / nu_left.
    Eigen::MatrixXd m_material_verts;  // n × 2
    Eigen::MatrixXi m_material_faces;  // m × 3 (may be empty → polygon vertices)

    std::vector<int> m_modes;  // mode indices used in cost
    std::vector<int> m_free_fine; // free DOF indices on fine mesh

    // Sparse K_fine restricted to free fine DOFs (with kinetic shift / spring BCs
    // already applied — matches the operator used in the eigenproblem) and its
    // LDLT factor.  Built once in init_from_obj and reused by candidate_cost
    // for the K_f^{-1} norm term.
    Eigen::SparseMatrix<double> m_K_fine_ff;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> m_K_fine_ff_solver;

    CollapseParams m_p;
    std::string    m_output_dir; // set via set_output_dir(); "" = disabled

    // Intermediate simulation-bundle config; populated by set_intermediate_bundle().
    // m_bundle_base is the smallest threshold in the geometric sequence
    // {base, 4·base, 16·base, ...}.  Empty prefix or zero base disables saves.
    std::string m_bundle_prefix_base;
    int         m_bundle_base = 0;

    // Cascade-MG prolongation output prefix; empty disables.
    std::string m_cascade_prefix;

    // Fine-mesh vertex IDs to attach springs to (loaded from p.spring_fine_verts_file).
    // Empty → fall back to left/right boundary detection inside apply_spring_bcs.
    std::vector<int> m_spring_fine_verts;

    // ---------------------------- Woodbury / CHOLMOD precomputed base state
    // Holds K_base (full coarse K with BCs applied) + its CHOLMOD factor, plus
    // the V/F snapshot they were built from. Rebuilt every simplify step.
    std::unique_ptr<WoodburyBase> m_wb_base;

    // (Re)build m_wb_base from the current TriMesh state (V_curr, F_curr).
    // Called once per simplify step.
    void rebuild_woodbury_base();

    // ------------------------------------------------ per-collapse cache

    struct CollapseCache {
        Eigen::Vector2d v1pos, v2pos;
        bool v1_frozen = false, v2_frozen = false;
    };
    CollapseCache   m_ccache;      // sequential only
    Eigen::Vector2d m_pending_pos; // set by simplify() before collapse_edge()

    // ---------------------------------------------------------- FEM assembly

    // Returns (K, M_diag) for the given 2D mesh (V: n×2, F: m×3).
    // K is 2n×2n sparse stiffness; M_diag is 2n lumped-mass diagonal.
    void assemble_fem(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        Eigen::SparseMatrix<double>& K,
        Eigen::VectorXd& M_diag) const;

    Eigen::Matrix3d elasticity_D(double E, double nu) const;

    // True if (x, y) lies inside the "left" material region.
    // Uses m_material_verts/faces when the material OBJ was loaded;
    // otherwise falls back to the hardcoded analytical boundary.
    bool point_in_material_left(double x, double y) const;

    // Analytical material E for a fine mesh triangle (centroid sampling).
    // Used only during init to populate m_fine_E.
    double material_E_analytical(const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const;

    // Area-weighted average E for a coarse triangle, projected from m_fine_E
    // via exact triangle-triangle intersection (Sutherland-Hodgman clipping).
    double material_E_from_fine(const Eigen::MatrixXd& V_coarse, const Eigen::Vector3i& tri) const;

    // Dispatch: calls material_E_from_fine when m_fine_E is populated (during
    // candidate_cost), otherwise falls back to analytical (during fine mesh init).
    double material_E(const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const;

    // Analogous to the three E functions above, but for Poisson's ratio nu.
    double material_nu_analytical(const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const;
    double material_nu_from_fine(const Eigen::MatrixXd& V_coarse, const Eigen::Vector3i& tri) const;
    double material_nu(const Eigen::MatrixXd& V, const Eigen::Vector3i& tri) const;

    // ------------------------------------------------- boundary utilities

    // Indices of vertices on left (x ≈ x_min) / right (x ≈ x_max) boundary
    std::vector<int> left_boundary_verts(const Eigen::MatrixXd& V) const;
    std::vector<int> right_boundary_verts(const Eigen::MatrixXd& V) const;

    // Add spring contribution to K.
    //   * If m_spring_fine_verts is non-empty: for each listed fine vertex v_f,
    //     locate the coarse triangle (V_coarse, F_coarse) containing it with
    //     barycentric weights (w0, w1, w2) at coarse vertices (c0, c1, c2), and
    //     add spring_k * w_i * w_j to K(2*c_i+comp, 2*c_j+comp) for comp ∈ {x,y}.
    //     This is exactly P^T K_spring^fine P, the projection of fine springs
    //     onto the coarse DOFs via the barycentric prolongation P.
    //   * Otherwise: fall back to legacy behavior — add spring_k to the diagonal
    //     of K at all DOFs on the left and right boundary of V_coarse.
    // No-op when m_p.spring_k == 0.
    void apply_spring_bcs(Eigen::SparseMatrix<double>& K,
                          const Eigen::MatrixXd& V_coarse,
                          const Eigen::MatrixXi& F_coarse) const;

    // All DOF indices (0-indexed) in the complement of fixed_verts
    std::vector<int> free_dof_indices(int ndof, const std::vector<int>& fixed_verts) const;

    // Boundary / corner classification.
    // general_mesh=false: coordinate-based bounding-box detection (rectangle).
    // general_mesh=true:  topological detection using F.
    // boundary_edges is always populated topologically (edges shared by 1 triangle).
    struct BoundaryInfo {
        std::vector<bool> on_boundary;
        std::vector<bool> on_corner;
        std::set<std::pair<int,int>> boundary_edges;
    };
    BoundaryInfo classify_boundary(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F) const;

    // Constrained collapse position.
    // boundary-boundary boundary edge → enclosure_pos (fine-mesh-enclosing).
    Eigen::Vector2d constrained_pos(
        int vi, int vj,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const BoundaryInfo& binfo) const;

    // Fine-mesh-enclosing position for collapsing boundary edge (vi, vj).
    // Returns the fine boundary vertex farthest outward from the chord vi–vj,
    // or the midpoint when the boundary is straight (all projections ≈ 0).
    // Uses m_coarse_face_fine_bdry_verts which must be current (rebuilt each step).
    Eigen::Vector2d enclosure_pos(
        int vi, int vj,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F) const;

    // Fine mesh boundary vertices (indices into m_V_fine); populated once in init.
    std::vector<int> m_fine_boundary_verts;
    // Per coarse face: fine boundary vertex indices inside that face.
    // Rebuilt once per simplify() step via rebuild_coarse_fine_overlap().
    std::vector<std::vector<int>> m_coarse_face_fine_bdry_verts;

    // (Re)build m_coarse_face_fine_bdry_verts for the given coarse mesh.
    void rebuild_coarse_fine_overlap(
        const Eigen::MatrixXd& V_coarse,
        const Eigen::MatrixXi& F_coarse);

    // -------------------------------------------------- cost computation

    // Builds scalar barycentric prolongation P_s (n_fine × n_coarse sparse).
    // fine vertices projected onto coarse triangulation.
    Eigen::SparseMatrix<double> build_barycentric_P(
        const Eigen::MatrixXd& V_coarse,
        const Eigen::MatrixXi& F_coarse) const;

    // Compute eigen-response cost for a candidate coarse mesh.
    // Returns infinity if infeasible.
    // When verbose=true, prints per-mode cost breakdown + geometric diagnostics.
    double candidate_cost(
        const Eigen::MatrixXd& V_cand,
        const Eigen::MatrixXi& F_cand) const;

    // Woodbury-based variant of candidate_cost (Apple Accelerate version).
    // Reuses the precomputed Apple Cholesky factor of K_base (m_wb_base) and
    // applies:
    //   1) a block-deletion correction for vj's DOFs (2 Apple solves)
    //   2) a Woodbury rank-s update on the 1-star of the merged vertex
    //      (s Apple solves + dense s×s inverse via Eigen)
    // plus spring contribution delta accumulation if applicable.
    //
    // vi_curr, vj_curr are the two endpoints' compact indices in m_wb_base->V.
    // V_cand / F_cand must be the result of collapse_edge_static(V_curr, F_curr, vi, vj).
    // Returns infinity if the Woodbury invariants don't hold (e.g. the V_cand
    // free-DOF set differs from V_curr's minus vj's DOFs because vi changed
    // boundary status), so callers can fall back to candidate_cost.
    double candidate_cost_woodbury_accelerated(
        const Eigen::MatrixXd& V_cand,
        const Eigen::MatrixXi& F_cand,
        int vi_curr, int vj_curr) const;

    // Lightweight approximate cost for edge (vi, vj) collapse.
    //   1) Build the local post-collapse 1-ring mesh (V_curr indexing, vi at
    //      cpos, vj merged into vi, degenerate tris dropped).
    //   2) Compute barycentric expansion of the OLD positions V_curr[vi] and
    //      V_curr[vj] inside this local mesh, decomposing the merged vertex's
    //      weight into V_curr[vi] / V_curr[vj] via ratio r along (vi, vj).
    //      bary_vi / bary_vj are lists of (V_curr index, weight) pairs.
    //   3) Use bary_vi / bary_vj to construct a constraint vector C and solve
    //      x = K_bc^{-1} C, then return x^T M x (or similar).
    // The C construction (step 3) is left in a simple placeholder form;
    // callers can refine it using the bary expansions.
    double cost_approx(
        int vi, int vj,
        int vi_wmtk, int vj_wmtk,
        const Eigen::MatrixXd& V_curr,
        const Eigen::MatrixXi& F_curr,
        const std::vector<std::vector<int>>& v_faces,
        const BoundaryInfo& binfo_curr) const;

    // -------------------------------------------------- mesh extraction

    // Extract current TriMesh topology as dense V (n×2) and F (m×3).
    void extract_current_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const;

    // Perform a single edge collapse on V/F (mirroring
    // collapse_edge_constrained from Python) without touching TriMesh.
    // Returns false if the collapse produces a degenerate mesh.
    bool collapse_edge_static(
        const Eigen::MatrixXd& V_in,
        const Eigen::MatrixXi& F_in,
        int vi, int vj,
        Eigen::MatrixXd& V_out,
        Eigen::MatrixXi& F_out) const;

    // -------------------------------------------------- visualization helpers

    // Called from simplify() each step; saves edge-cost PNG to
    // m_output_dir/edge_costs/step_NNNN.png.
    // edges[k] = {compact_vi, compact_vj}, costs[k] = cost for that edge.
    // best_idx is the index of the selected edge (highlighted in red).
    void save_edge_cost_png(
        int step,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F,
        const std::vector<std::pair<int,int>>& edges,
        const std::vector<double>& costs,
        int best_idx) const;

    // Saves m_output_dir/energy/energy_NNNN.svg showing the energy curve
    // up to and including collapse `step`.
    void save_energy_svg(int step) const;

    // Per-collapse energy log: m_energy_log[k] = cost of the k-th collapse.
    std::vector<double> m_energy_log;

    // Build P_free dense matrix for the given coarse mesh (shared logic
    // between candidate_cost and save_eigenmode_pngs).
    Eigen::MatrixXd build_P_free(
        const Eigen::MatrixXd& V_coarse,
        const Eigen::MatrixXi& F_coarse,
        const std::vector<int>& free_c) const;
};

} // namespace app::remesh
