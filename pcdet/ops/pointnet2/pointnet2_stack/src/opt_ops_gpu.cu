#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cmath>
namespace cg = cooperative_groups;

__global__ void bq_kernel_a(int B, int M, float radius, int nsample,
    const float *__restrict__ new_xyz, const int *__restrict__ new_xyz_batch_cnt,
    const float *__restrict__ xyz, const int *__restrict__ xyz_batch_cnt, int *__restrict__ idx) {
    int pt_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++) {
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }
    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int base = 0; base < n; base += 32) {
        int k = base + lane;
        bool hit = false;
        if (k < n) {
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            hit = d2 < radius2;
        }
        unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (mask) {
            if (cnt == 0) {
                int first = base + __ffs(mask) - 1;
                for (int l = lane; l < nsample; l += 32) idx[l] = first;
                __syncwarp();
            }
            int rank = cnt + __popc(mask & ((1u << lane) - 1u));
            if (hit && rank < nsample) idx[rank] = k;
            cnt += __popc(mask);
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0 && lane == 0) idx[0] = -1;
}

void ball_query_wrapper(int B, int M, double radius, int nsample, at::Tensor new_xyz, at::Tensor new_xyz_batch_cnt,
                        at::Tensor xyz, at::Tensor xyz_batch_cnt, at::Tensor idx) {
    if (M <= 0) return;
    const int threads = 256;
    long long total = (long long)M * 32;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bq_kernel_a<<<blocks, threads, 0, stream>>>(B, M, (float)radius, nsample,
        new_xyz.data_ptr<float>(), new_xyz_batch_cnt.data_ptr<int>(),
        xyz.data_ptr<float>(), xyz_batch_cnt.data_ptr<int>(), idx.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct Cand { float v; int k; };

__device__ __forceinline__ bool better(float v, int k, float bv, int bk, int S) {
    if (v != bv) return v > bv;
    int r = k % S, br = bk % S;
    if (r != br) return r < br;
    return k < bk;
}

__global__ void fps_kernel_a(int n, int m, int chunk, int S,
    const float *__restrict__ dataset, int *__restrict__ idxs,
    float *__restrict__ g_v, int *__restrict__ g_k) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sh[];
    float *sx = sh, *sy = sh + chunk, *sz = sh + 2 * chunk, *st = sh + 3 * chunk;
    __shared__ float wv[32];
    __shared__ int wk[32];
    __shared__ int s_old;

    const int G = gridDim.x;
    const int start = blockIdx.x * chunk;
    const int cnt = max(0, min(chunk, n - start));
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        sx[i] = dataset[(start + i) * 3 + 0];
        sy[i] = dataset[(start + i) * 3 + 1];
        sz[i] = dataset[(start + i) * 3 + 2];
        st[i] = 1e10f;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) idxs[0] = 0;
    int old = 0;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();

    for (int j = 1; j < m; j++) {
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        float best = -1.f; int besti = 0x7fffffff;
        for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
            float x2 = sx[i], y2 = sy[i], z2 = sz[i];
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, st[i]);
            st[i] = d2;
            int k = start + i;
            if (better(d2, k, best, besti, S)) { best = d2; besti = k; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best, off);
            int ok = __shfl_down_sync(0xffffffffu, besti, off);
            if (better(ov, ok, best, besti, S)) { best = ov; besti = ok; }
        }
        if (lane == 0) { wv[wid] = best; wk[wid] = besti; }
        __syncthreads();
        if (wid == 0) {
            best = lane < nw ? wv[lane] : -1.f;
            besti = lane < nw ? wk[lane] : 0x7fffffff;
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, best, off);
                int ok = __shfl_down_sync(0xffffffffu, besti, off);
                if (better(ov, ok, best, besti, S)) { best = ov; besti = ok; }
            }
            if (lane == 0) {
                g_v[(j & 1) * G + blockIdx.x] = best;
                g_k[(j & 1) * G + blockIdx.x] = besti;
            }
        }
        grid.sync();
        if (wid == 0) {
            best = -1.f; besti = 0x7fffffff;
            for (int g = lane; g < G; g += 32) {
                float ov = g_v[(j & 1) * G + g];
                int ok = g_k[(j & 1) * G + g];
                if (better(ov, ok, best, besti, S)) { best = ov; besti = ok; }
            }
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffffu, best, off);
                int ok = __shfl_down_sync(0xffffffffu, besti, off);
                if (better(ov, ok, best, besti, S)) { best = ov; besti = ok; }
            }
            if (lane == 0) s_old = besti;
        }
        __syncthreads();
        old = s_old;
        if (blockIdx.x == 0 && threadIdx.x == 0) idxs[j] = old;

    }
}

static int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return std::max(std::min(1 << pow_2, 1024), 1);
}

void farthest_point_sampling_wrapper(int b, int n, int m, at::Tensor points, at::Tensor temp, at::Tensor idx) {
    if (m <= 0) return;
    int dev; cudaGetDevice(&dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int maxsh = 0; cudaDeviceGetAttribute(&maxsh, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int G = sms;
    const int threads = 256;
    size_t budget = (size_t)maxsh - 1024;
    int chunk = (n + G - 1) / G;
    if ((size_t)chunk * 4 * sizeof(float) > budget) {
        chunk = (int)(budget / (4 * sizeof(float)));
        G = (n + chunk - 1) / chunk;
    }
    size_t shmem = (size_t)chunk * 4 * sizeof(float);
    cudaFuncSetAttribute(fps_kernel_a, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)budget);
    int maxb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fps_kernel_a, threads, shmem);
    TORCH_CHECK((long long)maxb * sms >= G, "fps: launch configuration does not fit");
    const int S = opt_n_threads(n);
    auto scratch_v = torch::empty({2 * G}, points.options());
    auto scratch_k = torch::empty({2 * G}, idx.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int bi = 0; bi < b; bi++) {
        const float *ds = points.data_ptr<float>() + (size_t)bi * n * 3;
        int *out = idx.data_ptr<int>() + (size_t)bi * m;
        float *gv = scratch_v.data_ptr<float>();
        int *gk = scratch_k.data_ptr<int>();
        void *args[] = {(void *)&n, (void *)&m, (void *)&chunk, (void *)&S, (void *)&ds, (void *)&out,
                        (void *)&gv, (void *)&gk};
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)fps_kernel_a, dim3(G), dim3(threads), args, shmem, stream));
    }
}

__global__ void group_points_kernel_stack_s(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);
    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;
    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++) {
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }
    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
    features += features_batch_start_idx * C;
    idx += pt_idx * nsample + sample_idx;
    int in_idx = idx[0] * C + C_idx;
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;
    out[out_idx] = features[in_idx];
}

void group_points_wrapper(int B, int M, int C, int nsample, at::Tensor features, at::Tensor features_batch_cnt,
                          at::Tensor idx, at::Tensor idx_batch_cnt, at::Tensor out) {
    long long total = (long long)M * C * nsample;
    if (total <= 0) return;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    group_points_kernel_stack_s<<<blocks, threads, 0, stream>>>(B, M, C, nsample, features.data_ptr<float>(),
        features_batch_cnt.data_ptr<int>(), idx.data_ptr<int>(), idx_batch_cnt.data_ptr<int>(), out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void group_kernel_b(int B, int M, int C, int nsample,
    const float *__restrict__ xyz, const float *__restrict__ features, const int *__restrict__ xyz_batch_cnt,
    const float *__restrict__ new_xyz, const int *__restrict__ idx, const int *__restrict__ idx_batch_cnt,
    const bool *__restrict__ empty, float *__restrict__ out) {
    const int C3 = C + 3;
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)C3 * M * nsample;
    if (index >= total) return;
    int s = index % nsample;
    int m = (index / nsample) % M;
    int c = index / ((long long)nsample * M);
    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++) {
        if (m < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }
    int start = 0;
    for (int k = 0; k < bs_idx; k++) start += xyz_batch_cnt[k];
    float v;
    if (empty[m]) {
        v = 0.f;
    } else {
        int j = start + idx[m * nsample + s];
        if (c < 3) {
            float g = xyz[j * 3 + c];
            g -= new_xyz[m * 3 + c];
            v = g;
        } else {
            v = features[(long long)j * C + (c - 3)];
        }
    }
    out[index] = v;
}

void fused_group_wrapper(int B, int M, int C, int nsample, at::Tensor xyz, at::Tensor features, at::Tensor xyz_batch_cnt,
                         at::Tensor new_xyz, at::Tensor idx, at::Tensor idx_batch_cnt, at::Tensor empty, at::Tensor out) {
    long long total = (long long)(C + 3) * M * nsample;
    if (total <= 0) return;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    group_kernel_b<<<blocks, threads, 0, stream>>>(B, M, C, nsample, xyz.data_ptr<float>(),
        C > 0 ? features.data_ptr<float>() : xyz.data_ptr<float>(), xyz_batch_cnt.data_ptr<int>(),
        new_xyz.data_ptr<float>(), idx.data_ptr<int>(), idx_batch_cnt.data_ptr<int>(), empty.data_ptr<bool>(),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper, "ball query");
    m.def("farthest_point_sampling_wrapper", &farthest_point_sampling_wrapper, "farthest point sampling");
    m.def("group_points_wrapper", &group_points_wrapper, "grouping");
    m.def("fused_group_wrapper", &fused_group_wrapper, "QueryAndGroup");
}
