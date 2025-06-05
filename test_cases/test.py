import os
import numpy as np
import time
from libpyCauchyKesaiS100FeaturemapsTools import CauchyKesai, __version__


TOTAL_N = 100
TASK_N = 28


def main():
    result_str = ""
    act_vision_path = "hbm_models/BPU_ACTPolicy_VisionEncoder.hbm"
    m = CauchyKesai(act_vision_path, TASK_N)
    for i in range(TOTAL_N):
        input1 = np.random.rand(1, 3, 480, 640).astype(np.float32)
        input1.tofile( "input1.bin")
        begin_time = time.time()
        output = m.inference([input1], i%TASK_N )[0]
        print("CauchyKesai Inference time =  %.2f ms"%(1000*(time.time() - begin_time)))
        cmd = f"hrt_model_exec infer --model_file {act_vision_path} --input_file input1.bin --enable_dump true --dump_path outputs"
        os.system(cmd)
        output_dump = np.fromfile("outputs/model_infer_output_0_Vision_Features.bin", dtype=np.float32).reshape(1, 512, 15, 20)
        os.system("rm -rf outputs")
        os.system("rm *bin")
        cos_sim = cosine_similarity(output, output_dump)
        diff = output - output_dump
        result_str += f"\n[ACT Vision Encoder] Cosine Similarity: {cos_sim:.4f}  diff: {np.max(diff)=}  {np.min(diff)=}  {np.mean(diff)=}"
    print(result_str)
    m.summary()


    act_transformers_path = "hbm_models/BPU_ACTPolicy_TransformerLayers.hbm"
    m = CauchyKesai(act_transformers_path, TASK_N)
    for i in range(TOTAL_N):
        input1 = np.random.rand(1, 6).astype(np.float32)           # shape: (1, 6)
        input2 = np.random.rand(1, 512, 15, 20).astype(np.float32)   # shape: (1, 512, 15, 20)
        input3 = np.random.rand(1, 512, 15, 20).astype(np.float32)   # shape: (1, 512, 15, 20)

        input1.tofile( "input1.bin")
        input2.tofile( "input2.bin")
        input3.tofile( "input3.bin")
        begin_time = time.time()
        output = m.inference([input1, input2, input3])[0]
        print("CauchyKesai Inference time =  %.2f ms"%(1000*(time.time() - begin_time)))
        cmd = f"hrt_model_exec infer --model_file {act_transformers_path} --input_file input1.bin,input2.bin,input3.bin --enable_dump true --dump_path outputs"
        os.system(cmd)
        output_dump = np.fromfile("outputs/model_infer_output_0_Actions.bin", dtype=np.float32).reshape(1,50,6)
        os.system("rm -rf outputs")
        os.system("rm *bin")
        cos_sim = cosine_similarity(output, output_dump)
        diff = output - output_dump
        result_str += f"\n[ACT Transformer Layers] Cosine Similarity: {cos_sim:.4f}  diff: {np.max(diff)=}  {np.min(diff)=}  {np.mean(diff)=}"
    print(result_str)
    m.summary()

def cosine_similarity(A, B):
    # 将张量展平为一维向量
    A_flat = A.flatten()
    B_flat = B.flatten()
    # 计算点积和范数
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    # 避免除以零
    if norm_A == 0 or norm_B == 0:
        return 0
    return dot_product / (norm_A * norm_B)


if __name__ == "__main__":
    main()
