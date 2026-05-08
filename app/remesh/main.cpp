#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <string>

#include "EigenEdgeCollapse.hpp"

namespace fs = std::filesystem;

// Replace the file extension of `path` with `new_ext` (e.g. ".svg")
static std::string replace_ext(const std::string& path, const std::string& new_ext)
{
    return fs::path(path).replace_extension(new_ext).string();
}

int main(int argc, char** argv)
{
    CLI::App app{"Eigen-response edge-collapse simplification for 2D triangle OBJ meshes"};

    std::string input, output = "simplified.obj", output_dir;
    app::remesh::CollapseParams p;

    app.add_option("input",             input,              "Input 2D OBJ mesh")->required();
    app.add_option("-o,--output",       output,             "Output OBJ path");
    app.add_option("--output_dir",      output_dir,         "Directory for edge-cost / eigenmode PNGs");
    app.add_option("--E_left",          p.E_left,           "Young's modulus (left material)");
    app.add_option("--E_right",         p.E_right,          "Young's modulus (right material)");
    app.add_option("--nu",              p.nu,               "Poisson's ratio (both materials, overridden by --nu_left/--nu_right)");
    app.add_option("--nu_left",         p.nu_left,          "Poisson's ratio (left material)");
    app.add_option("--nu_right",        p.nu_right,         "Poisson's ratio (right material)");
    app.add_option("--material_obj",    p.material_obj,     "OBJ file defining the left-material region (centroid-in-polygon test)");
    app.add_flag  ("--is_gradient",     p.is_gradient,      "Linear gradient: E/nu vary from (E_left,nu_left) at x_min to (E_right,nu_right) at x_max");
    app.add_option("--rho",             p.rho,              "Density");
    app.add_option("--num_modes",       p.num_modes,        "Number of eigenmodes to compute");
    app.add_option("--cost_modes",      p.cost_modes,       "Number of modes used in cost");
    app.add_option("--eig_tol",         p.eig_tol,          "Eigenvalue tolerance");
    app.add_flag  ("!--plane_strain",   p.plane_stress,     "Use plane strain (default: plane stress)");
    app.add_flag  ("!--no_fixed_left",  p.fixed_left,       "Do not fix left boundary");
    app.add_option("--weight_mode",     p.weight_mode,      "Weight mode: 0=none 1=inv_lambda 2=inv_lambda2");
    app.add_option("--alpha",           p.alpha,            "Kinetic shift alpha for K_eff=K+alpha*M (used when --no_fixed_left)");
    app.add_option("--spring_k",        p.spring_k,         "Spring BC: add spring_k to K diagonal at left+right boundary DOFs (overrides --no_fixed_left and --alpha)");
    app.add_flag  ("--general_mesh",    p.general_mesh,     "Non-rectangle input: use topological boundary detection and QEM for boundary edge collapses");
    app.add_option("--boundary_tol",    p.boundary_tol,     "Boundary classification tolerance");
    app.add_option("--target_vertices", p.target_vertices,  "Target vertex count (-1 = half)");

    CLI11_PARSE(app, argc, argv);

    // If output_dir is given, redirect all file outputs into it.
    // The bare filename part of --output is kept; the directory is replaced.
    if (!output_dir.empty()) {
        fs::create_directories(output_dir);
        output = (fs::path(output_dir) / fs::path(output).filename()).string();
    }

    app::remesh::EigenEdgeCollapse mesh;
    mesh.init_from_obj(input, p);

    if (!output_dir.empty())
        mesh.set_output_dir(output_dir);

    const int n_init = (int)mesh.get_vertices().size();
    const int target = (p.target_vertices > 0)
                           ? p.target_vertices
                           : std::max(4, n_init / 2);

    if (target >= n_init) {
        std::cerr << "target_vertices must be smaller than initial vertex count\n";
        return 1;
    }

    // ── material visualisation: fine mesh (before) ────────────────────
    const std::string svg_before = replace_ext(output, "_material_before.svg");
    mesh.save_material_svg(svg_before, mesh.V_fine(), mesh.F_fine());

    // ── simplification ────────────────────────────────────────────────
    int collapses = mesh.simplify(target);
    mesh.consolidate_mesh();
    mesh.write_obj(output);

    // ── material visualisation: simplified mesh (after) ───────────────
    const std::string svg_after = replace_ext(output, "_material_after.svg");
    mesh.save_material_svg(svg_after);

    // ── eigenmode PNGs ────────────────────────────────────────────────
    if (!output_dir.empty())
        mesh.save_eigenmode_pngs();

    std::cout << "Finished: " << n_init << " -> "
              << mesh.get_vertices().size() << " vertices ("
              << collapses << " collapses)\n";
    return 0;
}
