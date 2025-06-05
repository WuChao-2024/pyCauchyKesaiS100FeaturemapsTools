# pyCauchyKesaiS100FeaturemapsTools

# 声明

# 1. 所有源代码均开源, 使用前请务必保证您对程序有足够的认识. 本接口仅供社区开发者个人临时调试使用, 不保证其功能正确性, 作者不对任何错误和后果负责. 作者不对任何错误和后果负责. 作者不对任何错误和后果负责. 
# 2. 商业量产需求请使用BSP标准交付的Python接口模块. BSP都是专业的研发人员所开发的交付物, 提供的接口更加有保障. 如果你恰好发现BSP的接口不支持某些您使用OpenExplore编译的hbm模型, 请直接联系您的FAE或者销售哦, 去疯狂push他们交付一套好用的Python接口, 反正我是不会给你开发的.
# 3. 本仓库采用 GPL3.0 协议, 请遵守相关协议使用.


TODO: 
1. 模型初始化的打印注释掉
2. 新增dirty_run, 并带性能打印
3. release模型时输出模型的名称

## Abstract

```bash
OE 3.0.31
```


rm -rf ./* && cmake .. && make && rm ../../libpyCauchyKesaiS100FeaturemapsTools.so && cp libpyCauchyKesaiS100FeaturemapsTools.so ../../

## Make

```bash
mkdir build
cd build
cmake ..
make
```

## install

```bash
cp libpyCauchyKesaiS100FeaturemapsTools.so /usr/lib/python3.10/
cp libpyCauchyKesaiS100FeaturemapsTools.so ~/miniconda3/envs/< env name >/lib/python3.10/

python3 -c "from libpyCauchyKesaiS100FeaturemapsTools import __version__ ;print(__version__)"
```


## 模型编译说明

全部使用featuremap, 不要使用其他, 我知道这样不高效, 数据来自DDR, 不来自Primary, 无法和JPU, VPU等其他IP的数据衔接. 但是管他呢, 这样hbm模型的前后处理行为和ONNX是一致的, 工程的同学请不要约束算法的同学, 谢谢.

```yaml
model_parameters:
  onnx_model: 'onnx_name_BPU_ACTPolicy_TransformerLayers'
  march: "nash-e"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: 'BPU_TransformerLayers'
input_parameters:
  input_name: "states;laptop_features;phone_features;"
  input_type_rt: 'featuremap;featuremap;featuremap;'
  input_layout_rt: 'NCHW;NCHW;NCHW;'
  input_type_train: 'featuremap;featuremap;featuremap;'
  input_layout_train: 'NCHW;NCHW;NCHW;'
  norm_type: 'no_preprocess;no_preprocess;no_preprocess;'
calibration_parameters:
  cal_data_dir: '{os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, "state")};{os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, "laptop")};{os.path.join(calbrate_data_name_BPU_ACTPolicy_TransformerLayers, "phone")};'
  cal_data_type: 'float32;float32;float32;'
  calibration_type: 'default'
  optimization: set_all_nodes_int16
compiler_parameters:
  extra_params: {'input_no_padding': True, 'output_no_padding': True}
  jobs: 16
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O2'
```  

## 接口说明

懒得写的, 反正是自己用的. 


### 模型初始化


### 模型信息


### 同步推理接口



其中 [0,253] 为普通低优任务，254 为high抢占任务，255 为urgent抢占任务, 任务的优先级越高，在优先级队列中就越靠前，任务就会越早运行。

UCP提供嵌套抢占能力，抢占顺序：urgent抢占任务 > high抢占任务 > 普通低优任务。若需要实现抢占功能，请参考[模型优先级控制](http://j6.doc.oe.hobot.cc/3.0.31/guide/ucp/runtime/runtime_dev.html#preemption).


Nash BPU计算单元硬件本身没有任务抢占功能，对于每一个推理任务，一旦它进到BPU模型计算之后，在该任务执行完成之前都会一直占用BPU，其他任务只能排队等待。 此时很容易出现BPU计算资源被一个大模型推理任务所独占，进而影响其他高优先级模型的推理任务执行。 针对这种问题，Runtime SDK基于模型的优先级通过软件的方式实现了BPU资源抢占的功能。

其中有以下点需要被关注：

编译后的数据指令模型在BPU上进行推理计算时，它将表现为1个或者多个function-call的调用，其中function-call是BPU的执行粒度，多个function-call调用任务将在BPU的硬件队列上按序进行调度，当一个模型所有的function-call都执行完成，那么一个模型推理任务也就执行完成了。
基于上述描述，BPU模型任务抢占粒度设计为function-call更为简单，即BPU执行完一个function-call之后，暂时挂起当前模型，然后切入执行另外一个模型，当新模型执行完成之后，再恢复原来模型的状态继续运行。 但是这里存在两个问题，第一是经过编译器编译出来的模型function-call都是merge在一起，此时模型只有一个大的function-call，它无法被抢占；第二是每个function-call的执行时间比较长或者不固定，也会造成抢占时机不固定，影响抢占效果。
为了解决上述的两个问题，地平线在模型编译和系统软件层面都给予了支持，下面分别介绍其实现原理和操作方法：

首先，如果您选择使用QAT方案处理模型，则在 模型编译 阶段，您需要在编译接口中的额外参数配置中添加 max_time_per_fc 选项，用于设置每个function call的执行时间（以微秒为单位），其默认取值为 0 （即不做限制）。 您可以自行设置这个选项控制上板运行阶段个别大function-call的执行时间。假设某function-call执行时间为10ms，当模型编译时 将 max_time_per_fc 设置为 500，则这个function-call将会被拆分成20个。 而如果您使用PTQ方案处理模型，则在 模型转换 阶段，可以在模型的YAML配置文件中的编译器相关参数( compiler_parameters )中，添加 max_time_per_fc 参数。
其次，需要在推理任务提交时设置 hbUCPSchedParam.priority 参数。按照优先级还可以支持高优抢占嵌套能力。 如：配置 infer 任务优先级小于 254，则为普通任务，不可抢占其他任务。 配置 infer 任务优先级等于 254，则为high抢占任务，可支持抢占普通任务。 配置 infer 任务优先级等于 HB_DNN_PRIORITY_PREEMP(255)，则为urgent抢占任务，可抢占普通任务和high抢占任务。



### 异步推理接口


开启推理
```bash
start()
```

不做形状检查, 只最最终的size检查.
不做数据类型检查, 只memcpy.



## 测试用例

测试ACT的Vision Encoder 和 Transformer Layers, 和hrt_model_exec infer命令的结果一致.

```bash
Cosine Similarity: 1.0000  diff: np.max(diff)=0.0  np.min(diff)=0.0  np.mean(diff)=0.0
```