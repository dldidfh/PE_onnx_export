import tensorrt as trt

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

preprocess = T.Compose([
    # (C,H,W) 텐서 전제. uint8이면 0..255 → float32 0..1 로 변환
    # T.Resize((336, 336), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

def trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }[dt]

def run_v3(context):
    if hasattr(context, "execute_v3"):
        return context.execute_v3()  # 동기
    elif hasattr(context, "enqueue_v3"):
        return context.enqueue_v3(0)  # 기본 CUDA stream(0)
    elif hasattr(context, "execute_async_v3"):
        return context.execute_async_v3(stream_handle=0)  # 기본 스트림
    else:
        raise RuntimeError("TensorRT v3 execution API not found in this build.")

# mean/std는 필요에 맞게 바꾸세요 (예: CLIP 계열은 다른 값 사용)
DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD  = [0.5, 0.5, 0.5]

@torch.inference_mode()
def preprocess_image(
    image: torch.Tensor,
    size: int = 336,
    device: str = "cuda",
    mean = DEFAULT_MEAN,
    std  = DEFAULT_STD,
) -> torch.Tensor:
    """
    입력: torch.Tensor
      - (H, W, C) 또는 (C, H, W) 또는 (B, C, H, W) 또는 (B, H, W, C)
    출력: (B, 3, size, size), float32, normalize 완료, device로 이동
    """

    x = image

    # ---- 레이아웃 정규화: BCHW로 맞추기 ----
    if x.ndim == 3:
        # 3D → CHW 또는 HWC 판별
        if x.shape[0] in (1, 3, 4):     # CHW
            x = x
        elif x.shape[-1] in (1, 3, 4):  # HWC
            x = x.permute(2, 0, 1).contiguous()
        else:
            raise ValueError("3D tensor must be CHW or HWC with channels in {1,3,4}.")
        x = x.unsqueeze(0)  # 배치 차원 추가 → (1,C,H,W)

    elif x.ndim == 4:
        # 4D → BCHW 또는 BHWC
        if x.shape[1] in (1, 3, 4):     # BCHW
            x = x
        elif x.shape[-1] in (1, 3, 4):  # BHWC
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("4D tensor must be BCHW or BHWC with channels in {1,3,4}.")
    else:
        raise ValueError("Tensor must be 3D or 4D.")


    # ---- grayscale → RGB ----
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    # ---- 리사이즈 (BCHW 유지) ----
    x = TF.resize(x, [size, size], interpolation=T.InterpolationMode.BILINEAR, antialias=True)

    # ---- dtype/스케일: float32 [0,1] ----
    # 다양한 정수/실수 dtype을 안전하게 [0,1]로
    x = TF.convert_image_dtype(x, torch.float32)

    # ---- 정규화 ----
    x = TF.normalize(x, mean=mean, std=std)

    # ---- 디바이스로 이동 ----
    x = x.to(device, non_blocking=True).contiguous()
    return x
