import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import os
import time
import torch
import tensorrt as trt

# ---- helpers ----
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

logger = trt.Logger(trt.Logger.ERROR)
engine_path = "trt_export/vision_x64.engine"

with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 모델/전처리 (입력 크기 확인용)
cfg_name = os.path.basename("PE-Core-L14-336")
model = pe.CLIP.from_config(cfg_name, pretrained=True).cuda().eval()
preprocess = transforms.get_image_transform(model.image_size)

# I/O 메타데이터 준비
num_io = engine.num_io_tensors
tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
inp_names  = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
out_names  = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
assert len(inp_names) == 1, "this sample assumes a single input"
input_name = inp_names[0]

trt_in_dtype   = engine.get_tensor_dtype(input_name)
torch_in_dtype = trt_to_torch_dtype(trt_in_dtype)

def prepare_bindings(context, image_cuda, out_names):
    """입력 shape 설정 후 출력 텐서들을 1회 할당하고 바인딩 주소 등록."""
    # 동적 입력 shape 설정
    context.set_input_shape(input_name, tuple(image_cuda.shape))

    # 출력 텐서 준비
    out_tensors = {}
    for name in out_names:
        dims = context.get_tensor_shape(name)  # tensorrt.Dims -> 확정된 shape
        out_shape = tuple(int(d) for d in dims)
        trt_out_dtype   = engine.get_tensor_dtype(name)
        torch_out_dtype = trt_to_torch_dtype(trt_out_dtype)
        out_tensors[name] = torch.empty(out_shape, dtype=torch_out_dtype, device="cuda")

    # 바인딩 주소 설정
    bindings = [0] * num_io
    bindings[tensor_names.index(input_name)] = int(image_cuda.data_ptr())
    for name in out_names:
        bindings[tensor_names.index(name)] = int(out_tensors[name].data_ptr())

    for i in range(num_io):
        name = engine.get_tensor_name(i)
        context.set_tensor_address(name, bindings[i])

    return out_tensors

@torch.inference_mode()
def trt_infer_once(context):
    ok = run_v3(context)
    if ok is False:
        raise RuntimeError("TensorRT v3 execution failed")

def ensure_dtype_contig(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return x

def load_or_dummy_image():
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        img = transforms.load_image(image_path)
        return preprocess(img).cuda()  # (C,H,W)
    else:
        # test_image.jpg가 없으면 더미 텐서 생성
        if isinstance(model.image_size, (tuple, list)):
            size = model.image_size
            if len(size) == 2:
                H, W = size
            else:
                H = W = size[0]
        else:
            H = W = int(model.image_size)
        return torch.randn(3, H, W, device="cuda")

def benchmark(batch_sizes=range(1, 33), iters=100, warmup=10):
    base = load_or_dummy_image()  # (C,H,W)

    results = []
    torch.cuda.synchronize()
    for b in batch_sizes:
        # 입력 배치 구성
        image_b = base.unsqueeze(0).repeat(b, 1, 1, 1)  # (B,C,H,W)
        image_b = ensure_dtype_contig(image_b, torch_in_dtype)

        # 컨텍스트 생성 및 1회 출력 버퍼/바인딩 설정
        with engine.create_execution_context() as context:
            out_tensors = prepare_bindings(context, image_b, out_names)

            # 메모리 통계 초기화
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # 워밍업
            for _ in range(warmup):
                trt_infer_once(context)
            torch.cuda.synchronize()

            # 타이밍
            start = time.perf_counter()
            for _ in range(iters):
                trt_infer_once(context)
            torch.cuda.synchronize()
            end = time.perf_counter()

            avg_latency_ms = (end - start) * 1000.0 / iters
            throughput_img_s = (b * iters) / (end - start)

            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)   # MiB
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2) # MiB

            results.append({
                "batch": b,
                "avg_latency_ms": avg_latency_ms,
                "throughput_img_s": throughput_img_s,
                "peak_alloc_mib": peak_alloc,
                "peak_reserved_mib": peak_reserved,
            })

            # 바인딩/버퍼는 컨텍스트 종료와 함께 해제됨

        # 배치 간 캐시 정리
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    # 배치 1~32, 100회 평균
    bench = benchmark(batch_sizes=range(1, 33), iters=100, warmup=10)

    # 결과 출력
    print(f"{'B':>3} | {'avg(ms)':>10} | {'imgs/s':>10} | {'peak_alloc(MiB)':>16} | {'peak_reserved(MiB)':>18}")
    print("-" * 65)
    for r in bench:
        print(f"{r['batch']:>3} | {r['avg_latency_ms']:>10.3f} | {r['throughput_img_s']:>10.1f} | "
              f"{r['peak_alloc_mib']:>16.1f} | {r['peak_reserved_mib']:>18.1f}")
