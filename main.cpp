#include <Eigen/Eigen> 
#include <igl/cotmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <Eigen/Core>
Eigen::SparseMatrix<double, Eigen::RowMajor>  constructLaplace(const Eigen::SparseMatrix<double, Eigen::RowMajor> & w){
    Eigen::VectorXd sums = w * Eigen::VectorXd::Ones(w.cols());

    Eigen::SparseMatrix<double, Eigen::RowMajor> D_sparse(sums.size(), sums.size());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(sums.size());

    for (int i = 0; i < sums.size(); ++i) {
        triplets.emplace_back(i, i, sums[i]);
    }

    D_sparse.setFromTriplets(triplets.begin(), triplets.end());
   return D_sparse - w;
    
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
    // check determinant 
    return V*U.transpose();
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
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.launch();
    int n = V_1.rows();
    Eigen::SparseMatrix<double, Eigen::RowMajor> W = constructWeights(V_1,F);

    auto L = constructLaplace(W);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
    solver.compute(L);




    for(int j = 0; j < 3; j++){

        std::vector<Eigen::Matrix3d> R;
        R.resize(n);

        for(int i=0; i<n; i++){
            R[i] = computeRotation(i,V_1,V_2,W);
        }
        auto b = computeB(V_1,W,R);
    
        Eigen::VectorXd bx(n),by(n),bz(n);
        for(int i = 0; i < n; i++){
            bx(i) = b[i](0);
            by(i) = b[i](1);
            bz(i) = b[i](2);

        }
        Eigen::VectorXd x = solver.solve(bx);
        Eigen::VectorXd y = solver.solve(by);
        Eigen::VectorXd z = solver.solve(bz);

        for(int i = 0; i < n; i++) {
            V_2(i, 0) = x(i);
            V_2(i, 1) = y(i);
            V_2(i, 2) = z(i);
        }
    }
   
    return 0;
}