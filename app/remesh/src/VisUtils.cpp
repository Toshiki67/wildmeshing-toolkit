#include "VisUtils.hpp"
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace app::remesh::vis {

// ============================================================
//  Framebuffer
// ============================================================

Framebuffer::Framebuffer(int w, int h, uint8_t bg)
    : W(w), H(h), rgba(w * h * 4)
{
    std::fill(rgba.begin(), rgba.end(), bg);
    // Alpha = 255
    for (int i = 3; i < w * h * 4; i += 4) rgba[i] = 255;
}

void Framebuffer::set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    int i = (y * W + x) * 4;
    rgba[i]   = r;
    rgba[i+1] = g;
    rgba[i+2] = b;
    rgba[i+3] = a;
}

bool Framebuffer::save_png(const std::string& path) const
{
    return stbi_write_png(path.c_str(), W, H, 4, rgba.data(), W * 4) != 0;
}

// ============================================================
//  Colormaps
// ============================================================

// Viridis (5 key-point approximation)
static const float viridis_keys[5][3] = {
    {0.267f, 0.005f, 0.329f},
    {0.231f, 0.322f, 0.545f},
    {0.129f, 0.569f, 0.549f},
    {0.369f, 0.788f, 0.384f},
    {0.993f, 0.906f, 0.145f},
};

void cm_viridis(float t, uint8_t& r, uint8_t& g, uint8_t& b)
{
    t = std::max(0.0f, std::min(1.0f, t));
    float s = t * 4.0f;
    int   k = std::min((int)s, 3);
    float f = s - k;
    auto lerp = [&](int ch) {
        return viridis_keys[k][ch] * (1.0f - f) + viridis_keys[k+1][ch] * f;
    };
    r = (uint8_t)(std::max(0.0f, std::min(1.0f, lerp(0))) * 255);
    g = (uint8_t)(std::max(0.0f, std::min(1.0f, lerp(1))) * 255);
    b = (uint8_t)(std::max(0.0f, std::min(1.0f, lerp(2))) * 255);
}

// Cool-warm: blue(0) → white(0.5) → red(1)
void cm_coolwarm(float t, uint8_t& r, uint8_t& g, uint8_t& b)
{
    t = std::max(0.0f, std::min(1.0f, t));
    if (t < 0.5f) {
        float s = t * 2.0f;
        r = (uint8_t)(s * 255);
        g = (uint8_t)(s * 255);
        b = 255;
    } else {
        float s = (t - 0.5f) * 2.0f;
        r = 255;
        g = (uint8_t)((1.0f - s) * 255);
        b = (uint8_t)((1.0f - s) * 255);
    }
}

// ============================================================
//  Rasterisation
// ============================================================

void fill_tri(Framebuffer& fb,
              float x0, float y0, float x1, float y1, float x2, float y2,
              uint8_t r, uint8_t g, uint8_t b)
{
    int xmin = std::max(0, (int)std::floor(std::min({x0, x1, x2})));
    int xmax = std::min(fb.W - 1, (int)std::ceil(std::max({x0, x1, x2})));
    int ymin = std::max(0, (int)std::floor(std::min({y0, y1, y2})));
    int ymax = std::min(fb.H - 1, (int)std::ceil(std::max({y0, y1, y2})));

    const float denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (std::abs(denom) < 1e-5f) return;

    for (int py = ymin; py <= ymax; ++py) {
        for (int px = xmin; px <= xmax; ++px) {
            float bx = px + 0.5f, by = py + 0.5f;
            float w0 = ((x1 - bx) * (y2 - by) - (x2 - bx) * (y1 - by)) / denom;
            float w1 = ((x2 - bx) * (y0 - by) - (x0 - bx) * (y2 - by)) / denom;
            float w2 = 1.0f - w0 - w1;
            if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                fb.set_pixel(px, py, r, g, b);
        }
    }
}

void draw_line(Framebuffer& fb,
               float x0, float y0, float x1, float y1,
               uint8_t r, uint8_t g, uint8_t b, int thickness)
{
    int ix0 = (int)x0, iy0 = (int)y0;
    int ix1 = (int)x1, iy1 = (int)y1;
    int dx = std::abs(ix1 - ix0), dy = std::abs(iy1 - iy0);
    int sx = ix0 < ix1 ? 1 : -1, sy = iy0 < iy1 ? 1 : -1;
    int err = dx - dy;
    int half = thickness / 2;
    while (true) {
        for (int tx = -half; tx <= half; ++tx)
            for (int ty = -half; ty <= half; ++ty)
                fb.set_pixel(ix0 + tx, iy0 + ty, r, g, b);
        if (ix0 == ix1 && iy0 == iy1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; ix0 += sx; }
        if (e2 <  dx) { err += dx; iy0 += sy; }
    }
}

void draw_circle(Framebuffer& fb, float cx, float cy, int radius,
                 uint8_t r, uint8_t g, uint8_t b)
{
    int icx = (int)cx, icy = (int)cy;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            if (dx * dx + dy * dy <= radius * radius)
                fb.set_pixel(icx + dx, icy + dy, r, g, b);
}

// ============================================================
//  CoordMap
// ============================================================

CoordMap::CoordMap(const Eigen::MatrixXd& V, int W, int H, int pad_)
    : H(H), pad(pad_)
{
    double xmin = V.col(0).minCoeff(), xmax = V.col(0).maxCoeff();
    double ymin = V.col(1).minCoeff(), ymax = V.col(1).maxCoeff();
    double sx = (W - 2.0 * pad_) / std::max(xmax - xmin, 1e-9);
    double sy = (H - 2.0 * pad_) / std::max(ymax - ymin, 1e-9);
    scale = (float)std::min(sx, sy);
    // Centre the mesh horizontally and vertically
    ox = pad_ + (float)(0.5 * (W - 2 * pad_) - 0.5 * (xmax - xmin) * scale - xmin * scale);
    oy = (float)(ymin - 0.5 * ((H - 2.0 * pad_) / scale - (ymax - ymin)));
}

float CoordMap::screen_x(double wx) const
{
    return ox + (float)wx * scale;
}

float CoordMap::screen_y(double wy) const
{
    // Flip Y: world ymin → bottom of canvas
    return (float)(H - pad) - ((float)wy - oy) * scale;
}

// ============================================================
//  Mesh drawing
// ============================================================

void draw_mesh_edges(Framebuffer& fb, const CoordMap& cm,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     uint8_t r, uint8_t g, uint8_t b, int thickness)
{
    for (int f = 0; f < (int)F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int a = F(f, e), b_ = F(f, (e + 1) % 3);
            draw_line(fb,
                      cm.screen_x(V(a, 0)), cm.screen_y(V(a, 1)),
                      cm.screen_x(V(b_, 0)), cm.screen_y(V(b_, 1)),
                      r, g, b, thickness);
        }
    }
}

void draw_mesh_faces(Framebuffer& fb, const CoordMap& cm,
                     const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     const std::vector<float>& face_val,
                     bool use_coolwarm)
{
    for (int f = 0; f < (int)F.rows(); ++f) {
        uint8_t r, g, b;
        if (use_coolwarm)
            cm_coolwarm(face_val[f], r, g, b);
        else
            cm_viridis(face_val[f], r, g, b);

        fill_tri(fb,
                 cm.screen_x(V(F(f,0), 0)), cm.screen_y(V(F(f,0), 1)),
                 cm.screen_x(V(F(f,1), 0)), cm.screen_y(V(F(f,1), 1)),
                 cm.screen_x(V(F(f,2), 0)), cm.screen_y(V(F(f,2), 1)),
                 r, g, b);
    }
}

void draw_displaced_mesh(Framebuffer& fb, const CoordMap& cm,
                         const Eigen::MatrixXd& V,
                         const Eigen::MatrixXi& F,
                         const Eigen::VectorXd& u,
                         const std::vector<float>& face_mag)
{
    // Build displaced vertex positions (still 2-D)
    Eigen::MatrixXd Vd = V;
    for (int i = 0; i < (int)V.rows(); ++i) {
        Vd(i, 0) += u[2 * i];
        Vd(i, 1) += u[2 * i + 1];
    }
    draw_mesh_faces(fb, cm, Vd, F, face_mag, false);
    draw_mesh_edges(fb, cm, Vd, F, 60, 60, 60, 1);
}

// ============================================================
//  Colorbar
// ============================================================

void draw_colorbar(Framebuffer& fb, int x, int y, int w, int h,
                   float vmin, float vmax,
                   bool use_coolwarm, bool log_scale)
{
    // Filled bar
    const int steps = h;
    for (int s = 0; s < steps; ++s) {
        float t = 1.0f - (float)s / steps; // top = high
        uint8_t r, g, b;
        if (use_coolwarm) cm_coolwarm(t, r, g, b);
        else              cm_viridis (t, r, g, b);
        for (int dx = 0; dx < w; ++dx)
            fb.set_pixel(x + dx, y + s, r, g, b);
    }
    // Border
    for (int s = 0; s <= h; ++s) {
        fb.set_pixel(x,     y + s, 50, 50, 50);
        fb.set_pixel(x + w, y + s, 50, 50, 50);
    }
    for (int dx = 0; dx <= w; ++dx) {
        fb.set_pixel(x + dx, y,     50, 50, 50);
        fb.set_pixel(x + dx, y + h, 50, 50, 50);
    }
}

} // namespace app::remesh::vis
