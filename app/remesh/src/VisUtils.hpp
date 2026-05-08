#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace app::remesh::vis {

// ── RGBA framebuffer ────────────────────────────────────────────────────────

struct Framebuffer {
    int W, H;
    std::vector<uint8_t> rgba; // row-major, 4 bytes per pixel

    Framebuffer(int w, int h, uint8_t bg = 255);

    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

    // Save to PNG via stb_image_write
    bool save_png(const std::string& path) const;
};

// ── Colormaps ────────────────────────────────────────────────────────────────

// t ∈ [0,1]  →  (r,g,b)
void cm_viridis(float t, uint8_t& r, uint8_t& g, uint8_t& b);
void cm_coolwarm(float t, uint8_t& r, uint8_t& g, uint8_t& b); // blue-white-red

// ── Rasterisation helpers ────────────────────────────────────────────────────

// Filled triangle (screen coords, integer centred pixels)
void fill_tri(Framebuffer& fb,
              float x0, float y0, float x1, float y1, float x2, float y2,
              uint8_t r, uint8_t g, uint8_t b);

// Anti-aliased line (1-px or thick)
void draw_line(Framebuffer& fb,
               float x0, float y0, float x1, float y1,
               uint8_t r, uint8_t g, uint8_t b, int thickness = 1);

// Filled circle
void draw_circle(Framebuffer& fb, float cx, float cy, int radius,
                 uint8_t r, uint8_t g, uint8_t b);

// ── Coordinate mapper ────────────────────────────────────────────────────────

// Maps 2-D world coordinates into screen (pixel) coordinates.
// fit_mesh() scales V to fill [pad, W-pad] x [pad, H-pad] preserving aspect.
struct CoordMap {
    float ox, oy;   // world origin in screen coords
    float scale;    // world units per pixel
    int   H;        // canvas height (for Y flip)
    int   pad;

    CoordMap() = default;
    CoordMap(const Eigen::MatrixXd& V, int W, int H, int pad = 30);

    float sx(double wx) const { return ox + (float)wx * scale; }
    float sy(double wy) const { return H - pad - ((float)wy - oy / scale) * scale; }

    // Convenience: sy that flips y correctly given the stored offset
    float screen_x(double wx) const;
    float screen_y(double wy) const;
};

// ── Mesh drawing ─────────────────────────────────────────────────────────────

// Draw mesh edges in a solid colour
void draw_mesh_edges(Framebuffer& fb, const CoordMap& cm,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     uint8_t r = 100, uint8_t g = 100, uint8_t b = 100,
                     int thickness = 1);

// Fill each triangle with a colour looked up from a per-face value in [0,1]
void draw_mesh_faces(Framebuffer& fb, const CoordMap& cm,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     const std::vector<float>& face_val, // same length as F.rows()
                     bool use_coolwarm = false);

// Draw displaced mesh: Vd = V + u  where u is a 2*n vector (u_x, u_y per vertex)
void draw_displaced_mesh(Framebuffer& fb, const CoordMap& cm,
                         const Eigen::MatrixXd& V,
                         const Eigen::MatrixXi& F,
                         const Eigen::VectorXd& u, // length 2*n
                         const std::vector<float>& face_mag); // colour per face

// ── Colorbar ─────────────────────────────────────────────────────────────────

void draw_colorbar(Framebuffer& fb, int x, int y, int w, int h,
                   float vmin, float vmax, bool use_coolwarm = false,
                   bool log_scale = false);

} // namespace app::remesh::vis
