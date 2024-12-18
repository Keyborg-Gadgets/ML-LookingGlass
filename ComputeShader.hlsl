RWStructuredBuffer<uint> xBuffer : register(u0);
RWStructuredBuffer<uint> yBuffer : register(u1);
RWStructuredBuffer<uint> regionX : register(u2);
RWStructuredBuffer<uint> regionY : register(u3);
RWTexture2D<float4> cudaTexture : register(u4);
RWTexture2D<float4> outputTexture : register(u5);
RWStructuredBuffer<uint> debug : register(u6);

Texture2D<float4> screenTexture : register(t0);
SamplerState samplerState : register(s0);

static const float4 pattern[16] = {
    float4(1.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 1.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 1.0f, 0.0f),
    float4(1.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 1.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 1.0f, 0.0f),
    float4(1.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 1.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 1.0f, 0.0f),
    float4(1.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 1.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 1.0f, 0.0f),
    float4(1.0f, 0.0f, 0.0f, 0.0f),
    float4(0.0f, 1.0f, 0.0f, 0.0f),
    float4(0.0f, 0.0f, 1.0f, 0.0f),
    float4(1.0f, 0.0f, 0.0f, 0.0f)
};

[numthreads(32, 32, 1)]
void ScanTexture(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint dx = dispatchThreadID.x;
    uint dy = dispatchThreadID.y;
    uint width, height;
    screenTexture.GetDimensions(width, height);
    outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);


    if (dx >= width || dy >= height)
        return;

    if (dx + 15 >= width || dy + 15 >= height)
        return;

    bool isMatch = true;
    float epsilon = 0.1f;
    [loop]
    for (uint row = 0; row < 16 && isMatch; row++)
    {
        float4 expectedColor = pattern[row];
        for (uint col = 0; col < 16; col++)
        {
            float2 texCoord = (float2(dx + col + 0.5f, dy + row + 0.5f)) / float2(width, height);
            float4 color = screenTexture.SampleLevel(samplerState, texCoord, 0);

            if (abs(color.r - expectedColor.r) > epsilon ||
                abs(color.g - expectedColor.g) > epsilon ||
                abs(color.b - expectedColor.b) > epsilon)
            {
                isMatch = false;
                break;
            }
        }
    }
    if (isMatch)
    {
        xBuffer[0] = dx - 9;
        yBuffer[0] = dy + 25;
    }
}

[numthreads(32, 32, 1)]
void CopyTexture(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint dx = dispatchThreadID.x;
    uint dy = dispatchThreadID.y;
    uint width, height;
    screenTexture.GetDimensions(width, height);

    uint leftOfClient = xBuffer[0];
    uint topOfClient = yBuffer[0];
    uint subregionWidth = regionX[0];
    uint subregionHeight = regionY[0];
    uint cudaWidth, cudaHeight;
    cudaTexture.GetDimensions(cudaWidth, cudaHeight);

    bool inHorizontalBounds = (dx >= leftOfClient && dx < (leftOfClient + subregionWidth - 2));
    bool inVerticalBounds = (dy >= topOfClient && dy < (topOfClient + subregionWidth - 1));
    bool inCudaHorizontalBounds = dx <= cudaWidth;
    bool inCudaVerticalBounds = dy <= cudaHeight;

    if (inCudaHorizontalBounds && inCudaVerticalBounds)
    {
        float x_ratio = float(subregionWidth) / float(cudaWidth);
        float y_ratio = float(subregionWidth) / float(cudaHeight);

        float src_x = x_ratio * dx + leftOfClient - 2;
        float src_y = y_ratio * dy + topOfClient - 1;

        uint srcX = (uint)(src_x + 0.5f);
        uint srcY = (uint)(src_y + 0.5f);

        srcX = clamp(srcX, leftOfClient, leftOfClient + subregionWidth - 1);
        srcY = clamp(srcY, topOfClient, topOfClient + subregionWidth - 1);

        float2 texCoord = float2(srcX + 0.5f, srcY + 0.5f) / float2(width, height);

        float4 pixel = screenTexture.SampleLevel(samplerState, texCoord, 0);

        cudaTexture[uint2(dx, dy)] = pixel;
    }

    if (inHorizontalBounds && inVerticalBounds)
    {
        outputTexture[uint2(dx, dy)] = screenTexture[uint2(dx, dy)];
    }
    else
    {
        if (debug[0] == 1) {
            if (inHorizontalBounds)
                outputTexture[uint2(dx, dy)] = float4(1.0, 0.0, 0.0, 0.1);
            else if (!inVerticalBounds)
                outputTexture[uint2(dx, dy)] = float4(0.0, 1.0, 0.0, 0.1);
            else
                outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 1.0, 0.1);
        }
        else {
            if (inHorizontalBounds)
                outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
            else if (!inVerticalBounds)
                outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
            else
                outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
        }
    }
}