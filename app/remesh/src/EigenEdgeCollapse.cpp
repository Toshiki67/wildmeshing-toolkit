#include "EigenEdgeCollapse.hpp"
#include "VisUtils.hpp"

#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>

namespace app::remesh {

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
    Eigen::SparseMatrix<double>& K, const Eigen::MatrixXd& V) const
{
    if (m_p.spring_k <= 0.0) return;
    for (int v : left_boundary_verts(V)) {
        K.coeffRef(2 * v,     2 * v)     += m_p.spring_k;
        K.coeffRef(2 * v + 1, 2 * v + 1) += m_p.spring_k;
    }
    for (int v : right_boundary_verts(V)) {
        K.coeffRef(2 * v,     2 * v)     += m_p.spring_k;
        K.coeffRef(2 * v + 1, 2 * v + 1) += m_p.spring_k;
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
    apply_spring_bcs(Kc, V_cand);

    const int ndof_c = (int)Kc.rows();

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

    // Kinetic shift on coarse K_eff = K_c + alpha*M_c (matches fine-mesh shift).
    // Skipped when spring_k > 0 (springs already regularise the system).
    if (m_p.spring_k <= 0.0 && !m_p.fixed_left && m_p.alpha > 0.0)
        for (int k = 0; k < nf_c; ++k)
            Kc_ff.coeffRef(k, k) += m_p.alpha * Mc_diag[free_c[k]];

    // Factorize
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(Kc_ff);
    // if (ldlt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();

    // Barycentric prolongation P_s (n_fine × n_coarse scalar)
    Eigen::SparseMatrix<double> Ps = build_barycentric_P(V_cand, F_cand);

    // Expand to vector DOF: P (2*n_fine × 2*n_coarse)
    // P_free restricts coarse to free DOFs: (2*n_fine × nf_c)
    // P_free[2r, k]   = Ps[r, free_c[k]/2] if free_c[k] is even
    // P_free[2r+1, k] = Ps[r, free_c[k]/2] if free_c[k] is odd
    const int n_f = (int)m_V_fine.rows();
    const int ndof_f = 2 * n_f;

    // Build P_free as a dense matrix for simplicity (sizes are moderate)
    Eigen::MatrixXd P_free = Eigen::MatrixXd::Zero(ndof_f, nf_c);
    for (int k = 0; k < nf_c; ++k) {
        int gc = free_c[k];
        int vc = gc / 2;
        int comp = gc % 2; // 0=x, 1=y
        // Column k of P_free: set rows 2*pf+comp = Ps(pf, vc)
        for (Eigen::SparseMatrix<double>::InnerIterator it(Ps, vc); it; ++it) {
            int pf = (int)it.row();
            P_free(2 * pf + comp, k) = it.value();
        }
    }

    // M_f as diagonal vector (already stored)
    const Eigen::VectorXd& Mfd = m_M_diag;

    double cost = 0.0;
    int used = 0;

    Eigen::MatrixXd P_Mfd_inv = P_free.transpose() * Mfd.cwiseInverse().asDiagonal();
    Eigen::MatrixXd Mc_diag_inv_Kcff_P = Mc_diag.cwiseInverse().asDiagonal() * Kc_ff * P_free.transpose();


    for (int mode_id : m_modes) {
        const double lam = m_evals[mode_id];
        if (!std::isfinite(lam) || std::abs(lam) <= m_p.eig_tol) continue;

        Eigen::VectorXd phi = m_evecs.col(mode_id); // 2*n_fine

        // // b_f = M_f phi
        // Eigen::VectorXd bf = Mfd.cwiseProduct(phi);

        // // b_c = P_free^T b_f (nf_c vector)
        // Eigen::VectorXd bc = P_free.transpose() * bf;

        // // Solve K_c_ff y = b_c
        // Eigen::VectorXd yc = ldlt.solve(bc);
        // if (ldlt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();

        // // Coarse response projected back: P_free * yc
        // Eigen::VectorXd response = P_free * yc;

        // // Target: (1/lambda) phi
        // Eigen::VectorXd diff = (1.0 / lam) * phi - response;


    

        


        // // Weight
        // double w = 1.0;
        // if (m_p.weight_mode == 1)      w = 1.0 / std::abs(lam);
        // else if (m_p.weight_mode == 2) w = 1.0 / (lam * lam);

        // // ||diff||^2_{M_f}
        // double norm2 = (Mfd.cwiseProduct(diff)).dot(diff);
        // cost += w * norm2;


        Eigen::VectorXd weighted_phi = lam * P_Mfd_inv * phi;

        Eigen::VectorXd weighted_bc = Mc_diag_inv_Kcff_P * phi;
        Eigen::VectorXd diff = weighted_phi - weighted_bc;
        double norm2 = (Mc_diag.cwiseProduct(diff)).dot(diff);
        // std::cout << "Mode " << mode_id << ": lambda=" << lam << ", cost contribution=" << norm2 << "\n";
        cost += norm2;
        ++used;
    }

    if (used == 0) return std::numeric_limits<double>::infinity();
    return cost;
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
    apply_spring_bcs(K_fine, m_V_fine);

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
        // Extract current mesh arrays
        Eigen::MatrixXd V_curr;
        Eigen::MatrixXi F_curr;
        extract_current_mesh(V_curr, F_curr);

        // Precompute which fine boundary vertices lie in each coarse face.
        // Used by enclosure_pos() for boundary edge collapse positioning.
        rebuild_coarse_fine_overlap(V_curr, F_curr);

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
        };
        std::vector<Candidate> candidates;
        candidates.reserve(edges.size());

        for (const auto& et : edges) {
            const int vi_wmtk = (int)et.vid(*this);
            const int vj_wmtk = (int)et.switch_vertex(*this).vid(*this);

            // Reject only corner–corner edges; one-corner edges are allowed
            if (vertex_attrs[vi_wmtk].freeze && vertex_attrs[vj_wmtk].freeze) continue;

            // Skip edges that fail wmtk's link condition (topological constraint)
            if (!check_link_condition(et)) continue;

            const int vi = wmtk_to_compact[vi_wmtk];
            const int vj = wmtk_to_compact[vj_wmtk];

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

            double cost = candidate_cost(V_cand, F_cand);
            if (!std::isfinite(cost)) continue;

            Eigen::Vector2d cpos = constrained_pos(
                std::min(vi, vj), std::max(vi, vj), V_curr, F_curr, binfo_curr);

            candidates.push_back({cost, et, vi, vj, cpos});
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

    // Collect finite costs for range computation
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

    // Draw edges coloured by log10(cost)
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
    apply_spring_bcs(Kc, V_c);

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
    }
    std::cout << "Saved " << m_modes.size() << " eigenmode PNGs to "
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

} // namespace app::remesh
