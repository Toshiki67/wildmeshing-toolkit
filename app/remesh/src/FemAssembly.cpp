// ============================================================
//  FemAssembly.cpp — FEM core
// ============================================================
//  Mesh-level FEM utilities and topology operations:
//    * Polygon clipping (file-local)
//    * Elasticity tensor (plane stress / strain)
//    * Material assignment (analytical + fine-mesh projection)
//    * Stiffness + lumped mass assembly
//    * Boundary detection, spring BCs, free-DOF maps
//    * Barycentric prolongation, coarse-fine overlap
//    * Static edge collapse on (V, F) arrays

#include "EigenEdgeCollapse.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace app::remesh {

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

    // ── Initial candidate set ──────────────────────────────────────────────
    // The original set {mid, vi_pos, vj_pos} all sit on the chord vi–vj. When
    // the boundary curves outward (convex), neither endpoint can be enclosed
    // from a chord-resident cpos: vi falls outside any post-tri whose chord is
    // (cpos, vk_right), and likewise for vj.  OR-POCS then oscillates between
    // the two endpoints and converges to 3/4 coverage.
    //
    // We extend the candidate set with chord-perpendicular OUTWARD offsets so
    // that the search can escape the chord into the region outside the mesh,
    // where post-tris widen and can wrap both endpoints simultaneously.
    Eigen::Vector2d outward(0, 0);
    const Eigen::Vector2d chord = vj_pos - vi_pos;
    const double chord_len = chord.norm();
    if (chord_len > 1e-12 && adj_face >= 0) {
        // Find the adj-face vertex that is neither vi nor vj — it sits on the
        // interior side of the chord.
        int k_other = -1;
        for (int k = 0; k < 3; ++k) {
            const int v = F(adj_face, k);
            if (v != vi && v != vj) { k_other = k; break; }
        }
        if (k_other >= 0) {
            Eigen::Vector2d perp(-chord[1] / chord_len, chord[0] / chord_len);
            const Eigen::Vector2d w_adj = V.row(F(adj_face, k_other));
            // Flip so perp points AWAY from the interior (= outward).
            if (perp.dot(w_adj - mid) > 0) perp = -perp;
            outward = perp;
        }
    }

    std::vector<Eigen::Vector2d> candidates = {mid, vi_pos, vj_pos};
    if (outward.squaredNorm() > 0.5 && chord_len > 1e-12) {
        for (double f : {0.1, 0.3, 1.0, 3.0})
            candidates.push_back(mid + outward * (chord_len * f));
    }

    // ── OR-POCS from each candidate; keep the best result ─────────────────
    const int n_fbvs = (int)seen_fbvs.size();
    Eigen::Vector2d best_p = mid;
    int best_ever = -1;

    auto run_or_pocs = [&](Eigen::Vector2d p) {
        Eigen::Vector2d cand_best_p = p;
        int cand_best_ever = coverage_count(p);
        for (int iter = 0; iter < 2000 && cand_best_ever < n_fbvs; ++iter) {
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
            if (cov > cand_best_ever) { cand_best_ever = cov; cand_best_p = p; }
        }
        if (cand_best_ever > best_ever) {
            best_ever = cand_best_ever;
            best_p    = cand_best_p;
        }
    };

    for (const auto& p_init : candidates) {
        run_or_pocs(p_init);
        if (best_ever >= n_fbvs) break;   // fully covered; no need to try more
    }

    // Non-convergence diagnostic: report whenever OR-POCS terminates with at
    // least one uncovered fbv.  These are the cases where the returned cpos
    // can leave fine boundary verts outside the post-collapse 1-ring.
    if (best_ever < (int)seen_fbvs.size()) {
        std::cout << "[enclosure_pos] non-converged: edge (" << vi << "," << vj
                  << ")  covered " << best_ever << "/" << seen_fbvs.size()
                  << " fbvs  best_p=(" << best_p[0] << "," << best_p[1] << ")\n";
        std::cout << "  uncovered fbvs:";
        for (int fbv : seen_fbvs) {
            if (!is_covered(best_p, fbv)) {
                const Eigen::Vector2d q = m_V_fine.row(fbv);
                std::cout << " " << fbv << "(" << q[0] << "," << q[1] << ")";
            }
        }
        std::cout << "\n";
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


} // namespace app::remesh
