import sys
import time
import argparse

import numpy as np
from openvino.inference_engine import IECore

def main(args):
    # Search available NCS2 devices on the system
    MYRIADs = []
    ie = IECore()
    for device in ie.available_devices:
        if 'MYRIAD' in device:
            MYRIADs.append(device)
    num_devices = len(MYRIADs)
    print('{} MYRIAD devices found. {}'.format(len(MYRIADs), MYRIADs))
    if num_devices==0:
        return

    model = 'public/googlenet-v1/FP16/googlenet-v1'
    net = ie.read_network(model+'.xml', model+'.bin')

    # Build up the device descriptor
    if num_devices==1:
        device = 'MYRIAD'
    else:
        device='MULTI'
        for i, MYRIAD in enumerate(MYRIADs):
            device += ',' if i!=0 else ':'
            device += MYRIAD
    print('Device name : {}'.format(device))

    inblob   = list(net.input_info.keys())[0]
    inshape  = net.input_info[inblob].tensor_desc.dims
    outblob  = list(net.outputs.keys())[0]
    outshape = net.outputs[outblob].shape 

    config = {'VPU_HW_STAGES_OPTIMIZATION': 'YES'}    # default = 'YES'
    num_requests = 4 * num_devices
    execnet = ie.load_network(net, device, config=config, num_requests=num_requests)

    dummy = np.random.rand(1,3,224,224)

    niter = 100
    print('Start inferencing ({} times, {})'.format(niter, 'SYNC' if args.sync else 'ASYNC'))
    start = time.monotonic()
    for i in range(niter):
        if args.sync==True:
            execnet.infer(inputs={inblob:dummy})                        # Synchronous inference
        else:
            reqId = -1
            while reqId == -1:
                reqId = execnet.get_idle_request_id()
            execnet.requests[reqId].async_infer(inputs={inblob:dummy})  # Asynchronous inference

    if args.sync==False:
        # Wait for all requests to complete    
        for i in range(num_requests):
            execnet.requests[i].wait()

    end = time.monotonic()

    print('Performance = {} FPS'.format(niter/(end-start)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync', action='store_true', required=False, default=False, help='Use synchronous API for inferencing')
    args = parser.parse_args()

    sys.exit(main(args) or 0)
