#include <Eigen/Eigen> 
Eigen::SparseMatrix<double>  constructLaplace(Eigen::SparseMatrix<double> w){
   Eigen::VectorXd sums = W_sparse * Eigen::VectorXd::Ones(W_sparse.cols());
   Eigen::SparseMatrix<double> D_sparse = sums.asDiagonal();
   return D_sparse - w;
    
}

Eigen::SparseMatrix<double> constructWeights(const Eigen::MatrixXd & V,const Eigen::MatrixXi & F ){

    int n = V.rows();
    int f = F.rows();
    Eigen::SparseMatrix<double> W(n,n);
    std::vector<Eigen::Triplet<double>> triplets;
    for(int i = 0; i<f; i++){

        int i1 =F(i,0);
        int i2 =F(i,1);
        int i3 =F(i,2);

        Eigen::Vector3d  v1 = V.row(i1);
        Eigen::Vector3d  v2 = V.row(i2);
        Eigen::Vector3d  v3 = V.row(i3);

        double a1 = std::atan2((v2-v1).cross(v3-v1).norm() ,(v2-v1).dot(v3-v1)); 
        double a2 = std::atan2((v1-v2).cross(v3-v2).norm() ,(v1-v2).dot(v3-v2)); 
        double a3 = std::atan2((v1-v3).cross(v2-v3).norm() ,(v1-v3).dot(v2-v3)); 

        double w1 = (1.0/2)*(std::cos(a1)/std::sin(a1));
        double w2 = (1.0/2)*(std::cos(a2)/std::sin(a2));
        double w3 = (1.0/2)*(std::cos(a3)/std::sin(a3));

        triplets.push_back({i2,i3,w1});
        triplets.push_back({i3,i2,w1});
        
        triplets.push_back({i1,i3,w2});
        triplets.push_back({i3,i1,w2});

        triplets.push_back({i1,i2,w3});
        triplets.push_back({i2,i1,w3});
    }
    W.setFromTriplets(triplets.begin(), triplets.end()); 
    return W;
}

void main(){
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::SparseMatrix<double> weights = ; constructWeights(V,F);
    
    
}