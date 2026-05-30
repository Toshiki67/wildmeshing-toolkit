// ============================================================
//  Visualization.cpp — image / SVG output and small viz helpers
// ============================================================
//  * Colormap (matplotlib-like blue→white→red)
//  * Bitmap font + scientific-notation formatter for colorbar labels
//  * build_P_free helper (used by save_eigenmode_pngs)
//  * save_edge_cost_png + matching SVG
//  * save_eigenmode_pngs + matching SVG
//  * save_energy_svg
//  * save_material_svg (V/F overload + current-TriMesh overload)
//  * save_springs_svg

#include "EigenEdgeCollapse.hpp"
#include "VisUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace app::remesh {

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
