#include <Eigen/Core>
#include <igl/cotmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/project.h>
#include <igl/unproject.h>

Eigen::SparseMatrix<double, Eigen::RowMajor>  constructLaplace(const Eigen::SparseMatrix<double, Eigen::RowMajor> & w, std::vector<int> & map,int free){
    int n = w.cols();

    Eigen::SparseMatrix<double, Eigen::RowMajor> L(free,free);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n);

    for (int i = 0; i < n; ++i) {
        bool is_constrained = map[i] == -1;
        if(is_constrained){
            continue;
        }

         double sum = 0.0;
         for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(w, i); it; ++it){
            int j = it.col();
            double w_ij = it.value();
            if(i == j)
                continue;
            sum+=w_ij;
            if(map[j] != -1)
                triplets.emplace_back(map[i],map[j],-w_ij);
         }
             triplets.emplace_back(map[i], map[i], sum);
    }

    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
    
}

double cotan(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    double denom = a.cross(b).norm();
    if (denom < 1e-12) return 0.0;
    return a.dot(b) / denom;
}
Eigen::SparseMatrix<double, Eigen::RowMajor> constructWeights(const Eigen::MatrixXd & V,const Eigen::MatrixXi & F ){

    int n = V.rows();
    int f = F.rows();
    Eigen::SparseMatrix<double, Eigen::RowMajor>  W(n,n);
    std::vector<Eigen::Triplet<double>> triplets;
    for(int i = 0; i<f; i++){

        int i1 =F(i,0);
        int i2 =F(i,1);
        int i3 =F(i,2);

        Eigen::Vector3d  v1 = V.row(i1).transpose();
        Eigen::Vector3d  v2 = V.row(i2).transpose();
        Eigen::Vector3d  v3 = V.row(i3).transpose();


       double w1 = 0.5 * cotan(v2 - v1, v3 - v1);
       double w2 = 0.5 * cotan(v1 - v2, v3 - v2);
       double w3 = 0.5 * cotan(v1 - v3, v2 - v3);

        triplets.emplace_back(i2,i3,w1);
        triplets.emplace_back(i3,i2,w1);
        
        triplets.emplace_back(i1,i3,w2);
        triplets.emplace_back(i3,i1,w2);

        triplets.emplace_back(i1,i2,w3);
        triplets.emplace_back(i2,i1,w3);
    }
    W.setFromTriplets(triplets.begin(), triplets.end()); 
    return W;
}

Eigen::Matrix3d computeRotation(int i, const Eigen::MatrixXd & V_1, const Eigen::MatrixXd & V_2, const Eigen::SparseMatrix<double, Eigen::RowMajor> & w){
    
    Eigen::Matrix3d Si = Eigen::Matrix3d::Zero();
    Eigen::Vector3d vi1 = V_1.row(i).transpose();
    Eigen::Vector3d vi2 = V_2.row(i).transpose();

    for (Eigen::SparseMatrix<double, Eigen::RowMajor> ::InnerIterator it(w, i); it; ++it) {
        int j = it.col();

        Eigen::Vector3d e1 = V_1.row(j).transpose() - vi1;
        Eigen::Vector3d e2 = V_2.row(j).transpose() - vi2;

        Si += it.value()*e1*e2.transpose();
        
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Si, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V*U.transpose();

    if (R.determinant() < 0.0) {
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        D(2,2) = -1.0;
        R = V * D * U.transpose();
    }

    return R;
}

std::vector<Eigen::Vector3d> computeB(const Eigen::MatrixXd & V_1, const Eigen::SparseMatrix<double, Eigen::RowMajor> & W, const std::vector<Eigen::Matrix3d>& R){
   
    int n = V_1.rows();
    std::vector<Eigen::Vector3d> b(n, Eigen::Vector3d::Zero());
    
    for(int i = 0; i < n; i++){
            const Eigen::Matrix3d & R_i = R[i];
            Eigen::Vector3d p_i = V_1.row(i).transpose();
         for (Eigen::SparseMatrix<double, Eigen::RowMajor> ::InnerIterator it(W, i); it; ++it) {
            int j = it.col();
            Eigen::Vector3d p_j = V_1.row(j).transpose();
            b[i] += 0.5*it.value()*(R_i+R[j])*(p_i-p_j);
         }
    }
    return b;
}

int main(){

    Eigen::MatrixXd V_1;
    Eigen::MatrixXd V_2;
    Eigen::MatrixXi F;
    int numberOfPasses = 10;

    igl::readOFF("dino.off",V_1,F);
    V_2 = V_1;
    int n = V_1.rows();

    igl::opengl::glfw::Viewer viewer;

    std::unordered_map<int, Eigen::RowVector3d> constrained_pos;

    viewer.data().set_mesh(V_1, F);
    int active_handle = -1;
    bool dragging = false;

    Eigen::RowVector3d drag_start_pos(0.0, 0.0, 0.0);
    double drag_depth = 0.0;

    auto redraw_constraints = [&]() {
    viewer.data().clear_points();

    Eigen::MatrixXd P(constrained_pos.size(), 3);
    Eigen::MatrixXd C(constrained_pos.size(), 3);

    int k = 0;
    for (const auto& [vid, pos] : constrained_pos) {
        P.row(k) = pos;
        C.row(k) = Eigen::RowVector3d(1, 0, 0);
        ++k;
    }

    viewer.data().add_points(P, C);
};
    
    viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) -> bool
{
    int fid;
    Eigen::Vector3f bc;

    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;

    if (igl::unproject_onto_mesh(
            Eigen::Vector2f(x, y),
            viewer.core().view,
            viewer.core().proj,
            viewer.core().viewport,
            V_1,
            F,
            fid,
            bc))
    {
        int max_idx;
        bc.maxCoeff(&max_idx);

        int vid = F(fid, max_idx);

        if (modifier & GLFW_MOD_SHIFT) {
            constrained_pos[vid] = V_1.row(vid);
            redraw_constraints();
            std::cout << "Anchor vertex: " << vid << std::endl;
            return true;
        } else {
            auto it = constrained_pos.find(vid);
            if (it != constrained_pos.end()) {
                active_handle = vid;
                dragging = true;
                drag_start_pos = it->second;
         Eigen::Vector3f p = drag_start_pos.transpose().cast<float>();

Eigen::Matrix4f view = viewer.core().view.cast<float>();
Eigen::Matrix4f projm = viewer.core().proj.cast<float>();
Eigen::Vector4f viewport = viewer.core().viewport.cast<float>();

Eigen::Vector3f proj = igl::project(p, view, projm, viewport);
drag_depth = proj(2);
                std::cout << "Dragging handle: " << vid << std::endl;
                return true;
            }
        }
    }

    return false;
};


    viewer.callback_mouse_move =
    [&](igl::opengl::glfw::Viewer& viewer, int mouse_x, int mouse_y) -> bool
{
    if (!dragging || active_handle == -1)
        return false;

    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;

Eigen::Matrix4f view = viewer.core().view.cast<float>();
Eigen::Matrix4f projm = viewer.core().proj.cast<float>();
Eigen::Vector4f viewport = viewer.core().viewport.cast<float>();

Eigen::Vector3f win;
win << static_cast<float>(x),
       static_cast<float>(y),
       static_cast<float>(drag_depth);

Eigen::Vector3f obj = igl::unproject(win, view, projm, viewport);

constrained_pos[active_handle] = obj.cast<double>().transpose();

    redraw_constraints();

    return true;
};

viewer.callback_mouse_up =
    [&](igl::opengl::glfw::Viewer& viewer, int button, int modifier) -> bool
{
    dragging = false;
    active_handle = -1;
    return false;
};
    viewer.launch();

    Eigen::SparseMatrix<double, Eigen::RowMajor> W = constructWeights(V_1,F);

    std::vector<int> map(n, -1); 
    int indx = 0;
    for(int i = 0; i < n; i++){
        bool is_constrained =constrained_pos.find(i) != constrained_pos.end() ;
        if(!is_constrained){
            map[i] = indx;
            indx++;
        }
    }

    int free = 0;
    for (int i = 0; i < n; ++i)
        if (map[i] != -1) free++;

    auto L = constructLaplace(W,map,free);

    for(const auto& [i, pos] : constrained_pos){
        V_2.row(i) = pos;
    }

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Lcol = L;
    solver.analyzePattern(Lcol);
    solver.factorize(Lcol);

    if (solver.info() != Eigen::Success) {
        std::cout << "Factorization failed\n";
    }

    Eigen::VectorXd bx(free),by(free),bz(free);

    for(int iter = 0; iter < numberOfPasses; ++iter){

        std::vector<Eigen::Matrix3d> R;
        R.resize(n);

        for(int i=0; i<n; i++){
            R[i] = computeRotation(i,V_1,V_2,W);
        }

        auto b = computeB(V_1,W,R);
    
        
        for (int i = 0; i < n; i++) {
            if (map[i] == -1) continue; 

            int ii = map[i];

            Eigen::Vector3d bi = b[i];

            // Add contributions from constrained vertices
            for (Eigen::SparseMatrix<double, Eigen::RowMajor> ::InnerIterator  it(W, i); it; ++it) {
                int j = it.col();
                double w_ij = it.value();

                if (map[j] == -1) {
                    Eigen::Vector3d cj;
                   

                    bi += w_ij * constrained_pos[j].transpose();
                }
            }

            bx(ii) = bi(0);
            by(ii) = bi(1);
            bz(ii) = bi(2);
        }
    
    
        Eigen::VectorXd x = solver.solve(bx);
        Eigen::VectorXd y = solver.solve(by);
        Eigen::VectorXd z = solver.solve(bz);

       for(int i = 0; i < n; i++) {
       if (map[i] != -1) {
         int ii = map[i];
        V_2(i,0) = x(ii);
        V_2(i,1) = y(ii);
        V_2(i,2) = z(ii);
       }
    }
      
}    
       

    igl::opengl::glfw::Viewer viewer2;
    viewer2.data().set_mesh(V_2, F);
    viewer2.data().compute_normals();
    viewer2.launch();
    return 0;
}