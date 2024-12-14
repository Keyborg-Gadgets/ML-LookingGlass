// Control buffer layout:
// [0] = triggerScan (uint)
// [1] = scanComplete (uint)
RWStructuredBuffer<uint> controlBuffer : register(u0);

// Output buffer layout:
// [0] = outputPositionX (uint)
// [1] = outputPositionY (uint)
RWStructuredBuffer<uint> outputBuffer : register(u1);

// Region buffer layout:
// [0] = subregionWidth (uint)
// [1] = subregionHeight (uint)
RWStructuredBuffer<uint> regionBuffer : register(u2);

RWStructuredBuffer<uint> xBuffer : register(u5);
RWStructuredBuffer<uint> yBuffer : register(u6);

// Readable/writable textures
RWTexture2D<float4> cudaTexture : register(u3);
RWTexture2D<float4> outputTexture : register(u4);

// Read-only texture
Texture2D<float4> screenTexture : register(t0);
SamplerState samplerState : register(s0);

// Pattern array
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
    uint cb = controlBuffer[0];
    if (cb == 1)
    {
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
            outputBuffer[0] = dx - 10;
            outputBuffer[1] = dy + 25;
        }
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
    uint subregionWidth = regionBuffer[0];
    uint subregionHeight = regionBuffer[1];
    uint cudaWidth, cudaHeight;
    cudaTexture.GetDimensions(cudaWidth, cudaHeight);

    // Debug: Ensure bounds of dx and dy are correctly handled
    bool inHorizontalBounds = (dx >= leftOfClient && dx < (leftOfClient + subregionWidth));
    bool inVerticalBounds = (dy >= topOfClient && dy < (topOfClient + subregionWidth));

    if (inHorizontalBounds && inVerticalBounds)
    {
        outputTexture[uint2(dx, dy)] = screenTexture[uint2(dx, dy)];
    }
    else
    {
        if (inHorizontalBounds)
            outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
        else if (!inVerticalBounds)
            outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
        else
            outputTexture[uint2(dx, dy)] = float4(0.0, 0.0, 0.0, 0.0);
    }

    //uint i = dispatchThreadID.y;
    //uint j = dispatchThreadID.x;

    //if (j >= cudaWidth || i >= cudaHeight)
    //    return; 

    //float u = float(j) / float(cudaWidth - 1);
    //float v = float(i) / float(cudaHeight - 1);

    //float srcX = dx + u * (subregionWidth - 1);
    //float srcY = dy + v * (subregionHeight - 1);
    //float2 texCoord = float2(srcX + 0.5f, srcY + 0.5f) / float2(width, height);

    //if (srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
    //    return; 

    //float4 color = screenTexture.SampleLevel(samplerState, texCoord, 0);

    //cudaTexture[uint2(j, i)] = color;
}



