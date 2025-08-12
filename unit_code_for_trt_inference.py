from func import * 
from torchvision.io import read_image, ImageReadMode
import torch
import tensorrt as trt

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # I/O 메타데이터
        self.io_tensor_num = self.engine.num_io_tensors
        self.tensor_names  = [self.engine.get_tensor_name(i) for i in range(self.io_tensor_num)]
        self.inp_names     = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out_names     = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(self.inp_names) == 1, "This example assumes a single input."

        # I/O dtype
        self.trt_input_dtypes   = [self.engine.get_tensor_dtype(n) for n in self.inp_names]
        self.torch_input_dtypes = [trt_to_torch_dtype(dt) for dt in self.trt_input_dtypes]
        self.trt_output_dtypes   = [self.engine.get_tensor_dtype(n) for n in self.out_names]
        self.torch_output_dtypes = [trt_to_torch_dtype(dt) for dt in self.trt_output_dtypes]

    def _resolve_neg1(self, shape, batch):
        # TensorRT가 -1(동적)을 줄 경우 배치로 치환
        return tuple(batch if int(d) == -1 else int(d) for d in shape)

    def infer(self, image: torch.Tensor):
        # 1) 전처리 (B,3,H,W) float on CUDA
        x = preprocess_image(image)  # (B,3,336,336), float32, cuda, contiguous
        # 2) 엔진 입력 dtype에 맞춰 캐스팅/연속화
        want_dtype = self.torch_input_dtypes[0]
        if x.dtype != want_dtype:
            x = x.to(want_dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        # 3) 입력 shape 설정
        input_name = self.inp_names[0]
        self.context.set_input_shape(input_name, tuple(x.shape))

        # 4) 출력 텐서들 shape 계산 & 할당 (각 출력별 개별 shape)
        out_tensors = {}
        B = x.shape[0]
        for name, dt in zip(self.out_names, self.torch_output_dtypes):
            dims = self.context.get_tensor_shape(name)  # tensorrt.Dims
            out_shape = self._resolve_neg1(dims, batch=B)
            out_tensors[name] = torch.empty(out_shape, dtype=dt, device="cuda")

        # 5) 바인딩 주소 등록
        bindings = [0] * self.io_tensor_num
        bindings[self.tensor_names.index(input_name)] = int(x.data_ptr())
        for name in self.out_names:
            bindings[self.tensor_names.index(name)] = int(out_tensors[name].data_ptr())

        for i in range(self.io_tensor_num):
            name = self.tensor_names[i]
            self.context.set_tensor_address(name, bindings[i])

        # 6) 실행
        ok = run_v3(self.context)
        if not ok:
            raise RuntimeError("TensorRT inference failed.")

        # 7) 반환
        if len(self.out_names) == 1:
            return out_tensors[self.out_names[0]]
        return out_tensors

if __name__ == "__main__":
    engine_path = "trt_export/vision_x64.engine"
    image_path = "assets/cat.jpg"

    image = read_image(image_path, mode=ImageReadMode.RGB)  # (C,H,W), uint8, CPU
    trt_infer = TRTInference(engine_path)
    ret = trt_infer.infer(image)

    if isinstance(ret, dict):
        print({k: v.shape for k, v in ret.items()})
    else:
        print("TRT Inference Result:", ret.shape)
        print(ret)
