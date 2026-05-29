#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <igl/read_triangle_mesh.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/pick.h>
#include "imgui.h"

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
    app.add_option("--spring_fine_verts", p.spring_fine_verts_file, "File with fine-mesh vertex indices (whitespace-separated) to attach springs (uses P^T k P projection)");
    app.add_flag  ("--general_mesh",    p.general_mesh,     "Non-rectangle input: use topological boundary detection and QEM for boundary edge collapses");
    app.add_option("--boundary_tol",    p.boundary_tol,     "Boundary classification tolerance");
    app.add_option("--target_vertices", p.target_vertices,  "Target vertex count (-1 = half)");

    CLI11_PARSE(app, argc, argv);

    // ── Interactive parameter tuning + spring vertex picking ──────────────
    // Load the fine mesh just for visualization. The user adjusts `p` in place
    // via ImGui and picks spring vertices on the mesh. Pressing "Run" exits
    // the Polyscope window and proceeds with the existing pipeline below.
    std::set<int> picked_spring_verts;
    {
        Eigen::MatrixXd V_load;
        Eigen::MatrixXi F_load;
        if (!igl::read_triangle_mesh(input, V_load, F_load)) {
            std::cerr << "Could not read mesh: " << input << "\n";
            return 1;
        }
        // Promote to 3D (z = 0) for Polyscope.
        Eigen::MatrixXd V3(V_load.rows(), 3);
        V3.leftCols(2) = V_load.leftCols(2);
        V3.col(2).setZero();

        polyscope::init();
        auto* psFine = polyscope::registerSurfaceMesh("Fine mesh", V3, F_load);
        psFine->setSurfaceColor({0.85f, 0.85f, 0.85f});

        // Optional material region overlay
        polyscope::SurfaceMesh* psMaterial = nullptr;
        if (!p.material_obj.empty()) {
            Eigen::MatrixXd Vm3;
            Eigen::MatrixXi Fm;
            if (igl::read_triangle_mesh(p.material_obj, Vm3, Fm) && Vm3.rows() > 0) {
                if (Vm3.cols() < 3) {
                    Eigen::MatrixXd Vm3p(Vm3.rows(), 3);
                    Vm3p.leftCols(Vm3.cols()) = Vm3;
                    Vm3p.col(2).setZero();
                    Vm3 = Vm3p;
                } else {
                    Vm3.col(2).setZero();
                }
                psMaterial = polyscope::registerSurfaceMesh("Material region", Vm3, Fm);
                psMaterial->setSurfaceColor({0.2f, 0.4f, 1.0f});
                psMaterial->setTransparency(0.4f);
            }
        }

        // Seed initial selection from CLI file (if any).
        if (!p.spring_fine_verts_file.empty()) {
            std::ifstream ifs(p.spring_fine_verts_file);
            int idx;
            while (ifs >> idx)
                if (idx >= 0 && idx < (int)V_load.rows())
                    picked_spring_verts.insert(idx);
        }

        auto update_selection_viz = [&]() {
            const int n = (int)picked_spring_verts.size();
            Eigen::MatrixXd P(n, 3);
            int i = 0;
            for (int v : picked_spring_verts) {
                P(i, 0) = V3(v, 0);
                P(i, 1) = V3(v, 1);
                P(i, 2) = 0.0;
                ++i;
            }
            auto* pc = polyscope::registerPointCloud("Spring vertices", P);
            pc->setPointColor({1.0f, 0.1f, 0.1f});
            pc->setPointRadius(0.008);
        };
        if (!picked_spring_verts.empty()) update_selection_viz();

        bool pick_mode = false;
        bool run_now   = false;

        polyscope::state::userCallback = [&]() {
            // Vertex pick on mouse click (when in pick mode and not over ImGui)
            if (pick_mode && ImGui::IsMouseClicked(ImGuiMouseButton_Left)
                && !ImGui::GetIO().WantCaptureMouse) {
                ImVec2 mp = ImGui::GetMousePos();
                polyscope::PickResult pr =
                    polyscope::pickAtScreenCoords({mp.x, mp.y});
                if (pr.isHit && pr.structure == psFine) {
                    auto sm = psFine->interpretPickResult(pr);
                    if (sm.elementType == polyscope::MeshElement::VERTEX) {
                        int vid = (int)sm.index;
                        if (picked_spring_verts.count(vid))
                            picked_spring_verts.erase(vid);
                        else
                            picked_spring_verts.insert(vid);
                        update_selection_viz();
                    }
                }
            }

            ImGui::SetNextWindowSize(ImVec2(360, 760), ImGuiCond_FirstUseEver);
            ImGui::Begin("Simplification controls");

            if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputDouble("E_left",   &p.E_left,  0.0, 0.0, "%.4g");
                ImGui::InputDouble("E_right",  &p.E_right, 0.0, 0.0, "%.4g");
                ImGui::InputDouble("nu_left",  &p.nu_left, 0.01, 0.1);
                ImGui::InputDouble("nu_right", &p.nu_right,0.01, 0.1);
                ImGui::InputDouble("nu",       &p.nu,      0.01, 0.1);
                ImGui::InputDouble("rho",      &p.rho,     0.1, 1.0);
                ImGui::Checkbox("plane_stress", &p.plane_stress);
                ImGui::Checkbox("is_gradient",  &p.is_gradient);
            }

            if (ImGui::CollapsingHeader("Boundary conditions", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("fixed_left",  &p.fixed_left);
                ImGui::InputDouble("alpha",    &p.alpha);
                ImGui::InputDouble("spring_k", &p.spring_k, 0.0, 0.0, "%.4g");
                ImGui::InputDouble("boundary_tol", &p.boundary_tol, 0.0, 0.0, "%.4g");
                ImGui::Checkbox("general_mesh", &p.general_mesh);
            }

            if (ImGui::CollapsingHeader("Spring vertices", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Selected: %zu", picked_spring_verts.size());
                ImGui::Checkbox("Pick mode (click vertex to toggle)", &pick_mode);
                ImGui::SameLine();
                if (ImGui::Button("Clear")) {
                    picked_spring_verts.clear();
                    update_selection_viz();
                }
            }

            if (ImGui::CollapsingHeader("Modes", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputInt("num_modes",   &p.num_modes);
                ImGui::InputInt("cost_modes",  &p.cost_modes);
                ImGui::InputDouble("eig_tol",  &p.eig_tol, 0.0, 0.0, "%.4g");
                ImGui::InputInt("weight_mode", &p.weight_mode);
            }

            if (ImGui::CollapsingHeader("Target", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::InputInt("target_vertices", &p.target_vertices);
            }

            ImGui::Separator();
            if (ImGui::Button("Run simplification", ImVec2(-1, 30))) {
                run_now = true;
                polyscope::unshow();
            }

            ImGui::End();
        };

        polyscope::show();

        if (!run_now) return 0;   // user closed the window without running
    }
    // ── End interactive block; the rest is the original pipeline ─────────

    // If output_dir is given, redirect all file outputs into it.
    // The bare filename part of --output is kept; the directory is replaced.
    if (!output_dir.empty()) {
        fs::create_directories(output_dir);
        output = (fs::path(output_dir) / fs::path(output).filename()).string();
    }

    app::remesh::EigenEdgeCollapse mesh;

    // Seed picked spring vertices BEFORE init_from_obj — they take precedence
    // over CollapseParams::spring_fine_verts_file.
    if (!picked_spring_verts.empty()) {
        std::vector<int> picked(picked_spring_verts.begin(), picked_spring_verts.end());
        mesh.set_spring_fine_verts(picked);
    }
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

    // ── spring constraint locations overlay ───────────────────────────
    if (!mesh.spring_fine_verts().empty()) {
        const std::string svg_springs = replace_ext(output, "_springs.svg");
        mesh.save_springs_svg(svg_springs);
    }

    // ── simplification ────────────────────────────────────────────────
    int collapses = mesh.simplify(target);
    // int collapses = mesh.simplify_incremental(target);
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
