
typedef struct
{
    float x, y, w, h, s;
} box;

__device__ iou(box b1, box b2)
{
    float ai = (float)(b1.w) * (b1.h);
    float aj = (float)(b2.w) * (b2.h);

    float x_inter = max(b1.x, b2.x);
    float y_inter = max(b1.y, b2.y);

    float x2_inter = min((b1.x + b1.w), (b2.x + b2.w));
    float y2_inter = min((b1.y + b1.h), (b2.y + b2.h));

    float w = (float)max((float)0, x2_inter - x_inter);
    float h = (float)max((float)0, y2_inter - y_inter);

    float inter = ((w * h) / (ai + aj - w * h));
    return inter;
}
__global__ void NMS_GPU(box *d_b, bool *d_res)
{
    int abs_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int abs_x = (blockIdx.x * blockDim.x) + threadIdx.x;

    float theta = 0.6;

    if (d_b[abs_x].s < d_b[abs_y].s)
    {
        if (iou(d_b[abs_y], d_b[abs_x]) > theta)
        {
            d_res[abs_x] = false;
        }
    }
}