#include <torch/torch.h>
#include <iostream>

/*

rm -rf build;mkdir build;cd build;cmake \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_PREFIX_PATH="$PWD/libtorch" ..;make VERBOSE=1;cd ..

./build/app

*/


int main() {
torch::Tensor tensor = torch::eye(3);
std::cout << tensor << std::endl;

//   double  embedding[512];
double arr1[3] = { 1.0, 1.0, 0.0 }; 
double arr2[3] = { 0.0, 1.0, 0.0 }; 

std::cout << "\n\nDistance Start------------------------------------------------: " << std::endl;
auto options = torch::TensorOptions().dtype(torch::kFloat64);
// at::Tensor arr1Tensor = torch::from_blob(arr1, {1, 3}, options); 
// at::Tensor arr2Tensor = torch::from_blob(arr2, {1, 3}, options); 
// std::cout << "Tensor 1: " << arr1Tensor << '\n';
// std::cout << "Tensor 2: " << arr2Tensor << '\n';



double arrAvg[4] = { 1.0, 2.0, 3.0, 4.0 }; 
at::Tensor arrAvgTensor = torch::from_blob(arrAvg, {1, 4}, options); 
std::cout << "arrAvgTensor: " << arrAvgTensor << '\n';

// double* floatBuffer = arrAvgTensor.data_ptr<double>();
// double summ = 0.0;
// for( unsigned int a = 0; a < 4; a = a + 1 )
// {
//     summ =+ &floatBuffer[0][a];
// }
// std::cout << "floatBuffer : " << &floatBuffer << '\n';

/*
    ###### Based on `scipy`
    uv = np.average(embeddings1 * embeddings2)
    uu = np.average(np.square(embeddings1))
    vv = np.average(np.square(embeddings2))
    dist = 1.0 - uv / np.sqrt(uu * vv)

*/

std::cout << "Mean : " << arrAvgTensor.mean() << '\n'; // 2.5
std::cout << "Square : " << arrAvgTensor.square() << '\n'; // [ 1  4  9 16]


}
