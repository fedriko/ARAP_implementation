#include <Eigen/Core>
#include <igl/cotmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>

Eigen::SparseMatrix<double, Eigen::RowMajor>  constructLaplace(const Eigen::SparseMatrix<double, Eigen::RowMajor> & w, std::vector<int> & map){
    int n = w.cols();

    //Compute once
    int free = 0;
    for (int i = 0; i < n; ++i)
        if (map[i] != -1) free++;
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

    igl::readOFF("dino.off",V_1,F);
    V_2 = V_1;
    igl::opengl::glfw::Viewer viewer;
    std::set<int> anchors;
    std::set<int> moved;

    viewer.data().set_mesh(V_1, F);
    
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

            if (modifier == GLFW_MOD_SHIFT) {
                anchors.insert(vid);
                std::cout << "Anchor vertex: " << vid << std::endl;
            } else if (modifier == GLFW_MOD_CONTROL) {
                moved.insert(vid);
                std::cout << "moved vertex: " << vid << std::endl;
            } else {
                std::cout << "Picked vertex: " << vid << std::endl;
            }

            Eigen::RowVector3d p = V_1.row(vid);
            viewer.data().add_points(p, Eigen::RowVector3d(1,0,0));
            return true;
        }

        return false;
    };

   
    viewer.launch();



    int n = V_1.rows();
    Eigen::SparseMatrix<double, Eigen::RowMajor> W = constructWeights(V_1,F);

    std::vector<int> map(n, -1); 
    int indx = 0;
    for(int i = 0; i < n; i++){
        bool is_constrained =
        anchors.find(i) != anchors.end() || moved.find(i) != moved.end(); 
        if(!is_constrained){
            map[i] = indx;
            indx++;
        }
    }

    auto L = constructLaplace(W,map);

    for(int a : moved){
        V_2.row(a) += Eigen::RowVector3d(0,0,0.5);
    }

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseMatrix<double> Lcol = L;
    solver.analyzePattern(Lcol);
    solver.factorize(Lcol);
    if (solver.info() != Eigen::Success) {
    std::cout << "Factorization failed\n";
}
    int free = 0;
    for (int i = 0; i < n; ++i)
        if (map[i] != -1) free++;

    Eigen::VectorXd bx(free),by(free),bz(free);

    for(int iter = 0; iter < 15; ++iter){

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

            for (Eigen::SparseMatrix<double, Eigen::RowMajor> ::InnerIterator  it(W, i); it; ++it) {
                int j = it.col();
                double w_ij = it.value();

                if (map[j] == -1) {
                    Eigen::Vector3d cj;
                    if (anchors.find(j) != anchors.end())
                        cj = V_1.row(j).transpose();
                    else if (moved.find(j) != moved.end())
                        cj = V_1.row(j).transpose() + Eigen::Vector3d(0,0,0.5);

                    bi += w_ij * cj;
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
       if (map[i] == -1) {
                    Eigen::Vector3d ci;
                    if (anchors.find(i) != anchors.end())
                         ci = V_1.row(i).transpose();
                    else if (moved.find(i) != moved.end())
                        ci = V_1.row(i).transpose() + Eigen::Vector3d(0,0,0.5);
        V_2.row(i) = ci.transpose();
       }else{
         int ii = map[i];
        V_2(i,0) = x(ii);
        V_2(i,1) = y(ii);
        V_2(i,2) = z(ii);
       }
    }
        std::cout << "solve info x: " << (solver.info() == Eigen::Success) << std::endl;
        std::cout << "solve info y: " << (solver.info() == Eigen::Success) << std::endl;
        std::cout << "solve info z: " << (solver.info() == Eigen::Success) << std::endl;

        std::cout << "x finite: " << x.allFinite() << std::endl;
        std::cout << "y finite: " << y.allFinite() << std::endl;
        std::cout << "z finite: " << z.allFinite() << std::endl;

        std::cout << "max |x| = " << x.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "max |y| = " << y.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "max |z| = " << z.cwiseAbs().maxCoeff() << std::endl;
     
}    
       
    std::cout << "V_2 finite: " << V_2.allFinite() << std::endl;
    std::cout << "max |V_2| = " << V_2.cwiseAbs().maxCoeff() << std::endl;
    igl::opengl::glfw::Viewer viewer2;
    viewer2.data().set_mesh(V_2, F);
    viewer2.data().compute_normals();
    viewer2.launch();
    Eigen::SparseMatrix<double, Eigen::RowMajor> LT = L.transpose();
    Eigen::MatrixXd diff = Eigen::MatrixXd(L - LT);
    std::cout << "symmetry error = " << diff.norm() << std::endl;
    return 0;
}