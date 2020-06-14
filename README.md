# Tip for Multiple NCS2 with OpenVINO - Throughput mode
(日本語の説明はちょっと下にあります)  
Intel(r) Neural Computing Stick 2 is widly used in many hobby projects. It's low cost, high performance and easy to get.  
If you want a bit more performance in your project and considering to add some more NCS2s, this is a MUST KNOW tip to achieve your goal.  
OpenVINO creates an `IENetwork` object from the IR model by `ie.read_network()` API and create one or more `ExecutableNetwork` object(s) from the `IENetwork` object.  
```
 IR model -> IENetwork (device independent) -> ExecutableNetwork (device dependent)
```
The `ExecutableNetwork` object contains `InferRequest` objects which is the queue to send the actual inferencing request. If you create multiple `InferRequest` objects, you can submit multiple inferencing requests to a device at a time. This inferencing mode is so called **'throughput mode'** and it will increase the inferencing throughput (, not latency).    
You can specify how many `InferRequests` to create in the `ExecutableNetwork` by `num_requests` parameter in the `ie.load_network()` API.  
```Python
E.g. exec_net = ie.load_network(net, 'MYRIAD', num_requests=4)
```
To saturate a NCS2 device to draw the maximum performance form the device, Intel recommends to send 4 inference requests simultaneously. This means, you should create 4 `InferRequest`(s) per device. If you have 2 NCS2s, you sould create 8 `InferRequest`s.  
Also, to bind multiple NCS2 devices and treat them as a single device, you should use `'MULTI:'` to specify the device in the `load_network()` API. When you have 2 NCS2 devices, individual device names are something like `MYRIAD.x.y-ma2480`. In this case, the actual device name should be `'MULTI:MYRIAD.x.y1-ma2480,MYRIAD.x.y2-ma2480`.  
```Python
E.g. exec_net = ie.load_network(net, 'MULTI:MYRIAD.1.1-ma2480,MYRIAD.1.2-ma2480', num_requests=4*2)
```

Here's a simple throughput performance test result. If you don't use throughput mode (multiple infer requests with async API), the performance is the same regardless the number of NCS2s attached to the system.

|#NCS|SYNC|ASYNC(Throughput mode)|
|:--:|--:|--:|
|x2|40.26|177.62|
|x1|41.02|92.68|

(FPS, googlenet-v1)
  
------
  
Intel(r) Neural Computing Stick 2は幅広いホビープロジェクトに利用されています。安価でパフォーマンスが高く、入手性もよいためです。  
ここではあなたがもう少し性能を上げるためにプロジェクトにさらなるNCS2を追加しようと考えているなら**知っておかなければならないテクニック**を紹介しています。  
OpenVINOはIRモデルを`ie.read_network()` APIで読み込み、`IENetwork`オブジェクトを生成し、そこから１つまたは複数の`ExecutableNetwork`オブジェクトを生成します。  
```
 IR model -> IENetwork (device independent) -> ExecutableNetwork (device dependent)
```
`ExecutableNetwork`オブジェクトは`InferRequest`オブジェクトを含んでおり、これは実際の推論要求を送るためのキューとして機能します。複数の`InferRequests`オブジェクトを生成することで１つのデバイスに複数の推論要求を送信することが可能になります。この推論モードのことを**Throught mode**と呼び、これにより推論スループットを向上することが可能になります(レイテンシーは向上しません)。  
いくつ`InferRequest`オブジェクトを生成するかは`ie.load_network()` APIの`num_requests`パラメータで指定します。
```Python
E.g. exec_net = ie.load_network(net, 'MYRIAD', num_requests=4)
```
NCS2を飽和させ、最大のパフォーマンスを引き出すためには同時に4つの推論要求を送信することが推奨されています。つまり、1つのデバイス当たり4つの`InferRequest`オブジェクトを生成するとよいということです。もしあなたがNCS2を2つ持っているなら`InferRequest`を8個生成してください。  
また、複数のNCS2デバイスをまとめて1つのデバイスとして取り扱うためには`'MULTI:'`を`load_network()` APIでのデバイス指定に指定します。もしNCS2が2つあるなら、それぞれのデバイス名は`MYRIAD.x.y-ma2480`のようになるでしょう。この場合、デバイス名としては`'MULTI:MYRIAD.x.y1-ma2480,MYRIAD.x.y2-ma2480`のように指定します。   
```Python
E.g. exec_net = ie.load_network(net, 'MULTI:MYRIAD.1.1-ma2480,MYRIAD.1.2-ma2480', num_requests=4*2)
```

以下にに簡単なスループット性能テストの結果を示します。Throughput mode (複数推論要求をAsync APIで送信)を使用しない場合、NCS2が複数あっても性能が伸びないことが分かります。  

|#NCS|SYNC|ASYNC(Throughput mode)|
|:--:|--:|--:|
|x2|40.26|177.62|
|x1|41.02|92.68|

(FPS, googlenet-v1)  
  
------
  
### Required DL Models to Run This Demo

The demo expects the following models in the Intermediate Representation (IR) format:

  * `googlenet-v1`

You can download those models from OpenVINO [Open Model Zoo](https://github.com/opencv/open_model_zoo).
In the `models.lst` is the list of appropriate models for this demo that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

## How to Run

(Assuming you have successfully installed and setup OpenVINO 2020.2 or 2020.3. If you haven't, go to the OpenVINO web page and follow the [*Get Started*](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) guide to do it.)  

### 1. Install dependencies  
The demo depends on:
- `numpy`

To install all the required Python modules you can use:

``` sh
(Linux) pip3 install numpy
(Win10) pip install numpy
```

### 2. Download DL models from OMZ
Use `Model Downloader` to download the required models.
``` sh
(Linux) python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
(Win10) python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst
```

### 3. Run the demo app

Plug 1 or multiple NCS/NCS2 devices, then run the program.  

``` sh
(Linux) python3 multi-ncs.py [--sync]
(Win10) python multi-ncs.py [--sync]
```
If you specify `--sync` option, the program doesn't use throughput mode (and use synchronous inferencing API)

### Test log (reference)
```sh
>python multi-ncs.py
[E:] [BSL] found 0 ioexpander device
2 MYRIAD devices found. ['MYRIAD.5.1-ma2480', 'MYRIAD.5.3-ma2480']
Device name : MULTI:MYRIAD.5.1-ma2480,MYRIAD.5.3-ma2480
Start inferencing (100 times, ASYNC)
Performance = 177.61989342803182 FPS

>python multi-ncs.py --sync
[E:] [BSL] found 0 ioexpander device
2 MYRIAD devices found. ['MYRIAD.5.1-ma2480', 'MYRIAD.5.3-ma2480']
Device name : MULTI:MYRIAD.5.1-ma2480,MYRIAD.5.3-ma2480
Start inferencing (100 times, SYNC)
Performance = 40.257648953302365 FPS

>python multi-ncs.py
[E:] [BSL] found 0 ioexpander device
1 MYRIAD devices found. ['MYRIAD']
Device name : MYRIAD
Start inferencing (100 times, ASYNC)
Performance = 92.6784059314222 FPS

>python multi-ncs.py --sync
[E:] [BSL] found 0 ioexpander device
1 MYRIAD devices found. ['MYRIAD']
Device name : MYRIAD
Start inferencing (100 times, SYNC)
Performance = 41.01722723543717 FPS
```
## Tested Environment  
- Windows 10 x64 1909 and Ubuntu 18.04 LTS  
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2 and 2020.3  
- Python 3.6.5 x64  

## See Also  
* [Using Open Model Zoo demos](../../README.md)  
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)  
* [Model Downloader](../../../tools/downloader/README.md)  
