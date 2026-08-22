#include "gpuforge/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <mma.h>
using namespace nvcuda;
namespace gpuforge {
__global__ void gemm_tiled(const float* A,const float* B,float* C,int M,int N,int K){
  __shared__ float As[32][32], Bs[32][32]; int tx=threadIdx.x,ty=threadIdx.y; int row=blockIdx.y*32+ty,col=blockIdx.x*32+tx; float acc=0;
  for(int t=0;t<K;t+=32){As[ty][tx]=(row<M&&t+tx<K)?A[row*K+t+tx]:0;Bs[ty][tx]=(t+ty<K&&col<N)?B[(t+ty)*N+col]:0;__syncthreads();for(int i=0;i<32;++i)acc+=As[ty][i]*Bs[i][tx];__syncthreads();}
  if(row<M&&col<N)C[row*N+col]=acc;
}
__global__ void qkv_attention(const float* Q,const float* K,const float* V,float* O,int T,int D){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=T)return;float score[256];float mx=-1e30f;
  for(int j=0;j<T;++j){float s=0;for(int d=0;d<D;++d)s+=Q[i*D+d]*K[j*D+d];score[j]=s/sqrtf((float)D);if(j>i)score[j]=-1e9f;mx=fmaxf(mx,score[j]);}
  float den=0;for(int j=0;j<T;++j){score[j]=expf(score[j]-mx);den+=score[j];}for(int d=0;d<D;++d){float x=0;for(int j=0;j<T;++j)x+=score[j]/den*V[j*D+d];O[i*D+d]=x;}
}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
bool cuda_available(){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}
Tensor gemm_cuda(const Tensor&a,const Tensor&b,const GemmConfig&){if(a.ndim()!=2||b.ndim()!=2||a.shape()[1]!=b.shape()[0])throw std::invalid_argument("gemm_cuda shape");int M=(int)a.shape()[0],K=(int)a.shape()[1],N=(int)b.shape()[1];Tensor o({(size_t)M,(size_t)N});float *da,*db,*dc;ck(cudaMalloc(&da,a.numel()*4));ck(cudaMalloc(&db,b.numel()*4));ck(cudaMalloc(&dc,o.numel()*4));ck(cudaMemcpy(da,a.data(),a.numel()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.numel()*4,cudaMemcpyHostToDevice));dim3 block(32,32),grid((N+31)/32,(M+31)/32);gemm_tiled<<<grid,block>>>(da,db,dc,M,N,K);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(o.data(),dc,o.numel()*4,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(db);cudaFree(dc);return o;}
Tensor attention_cuda(const Tensor&q,const Tensor&k,const Tensor&v,const AttentionConfig&){if(q.ndim()!=2||k.shape()!=q.shape()||v.shape()!=q.shape()||q.shape()[0]>256)throw std::invalid_argument("attention_cuda shape");Tensor o(q.shape());float *dq,*dk,*dv,*do_;size_t bytes=q.numel()*4;ck(cudaMalloc(&dq,bytes));ck(cudaMalloc(&dk,bytes));ck(cudaMalloc(&dv,bytes));ck(cudaMalloc(&do_,bytes));ck(cudaMemcpy(dq,q.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(dk,k.data(),bytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,v.data(),bytes,cudaMemcpyHostToDevice));qkv_attention<<<(q.shape()[0]+127)/128,128>>>(dq,dk,dv,do_,(int)q.shape()[0],(int)q.shape()[1]);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(o.data(),do_,bytes,cudaMemcpyDeviceToHost));cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(do_);return o;}
__global__ void wmma_gemm(const half* a,const half* b,float* c,int M,int N,int K){int bx=blockIdx.x,by=blockIdx.y;wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb;wmma::fragment<wmma::accumulator,16,16,16,float> fc;wmma::fill_fragment(fc,0.0f);for(int t=0;t<K;t+=16){wmma::load_matrix_sync(fa,a+by*16*K+t,K);wmma::load_matrix_sync(fb,b+t*N+bx*16,N);wmma::mma_sync(fc,fa,fb,fc);}wmma::store_matrix_sync(c+by*16*N+bx*16,fc,N,wmma::mem_row_major);}
Tensor gemm_tensorcore(const Tensor&a,const Tensor&b){if(a.ndim()!=2||b.ndim()!=2||a.shape()[1]!=b.shape()[0]||a.shape()[0]%16||b.shape()[1]%16||a.shape()[1]%16)throw std::invalid_argument("tensorcore requires multiples of 16");int M=a.shape()[0],K=a.shape()[1],N=b.shape()[1];Tensor o({(size_t)M,(size_t)N});std::vector<half> ha(a.numel()),hb(b.numel());for(size_t i=0;i<a.numel();++i)ha[i]=__float2half(a.at(i));for(size_t i=0;i<b.numel();++i)hb[i]=__float2half(b.at(i));half*da;half*db;float*dc;ck(cudaMalloc(&da,ha.size()*2));ck(cudaMalloc(&db,hb.size()*2));ck(cudaMalloc(&dc,o.numel()*4));ck(cudaMemcpy(da,ha.data(),ha.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));wmma_gemm<<<dim3(N/16,M/16),32>>>(da,db,dc,M,N,K);ck(cudaDeviceSynchronize());ck(cudaMemcpy(o.data(),dc,o.numel()*4,cudaMemcpyDeviceToHost));cudaFree(da);cudaFree(db);cudaFree(dc);return o;}
}
