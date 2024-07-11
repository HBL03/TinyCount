import torch
import time
from datetime import datetime
import argparse

from models.SCC_Model.TinyCount import TinyCount
from models.SCC_Model.MCNN import MCNN
from models.SCC_Model.CSRNet import CSRNet
from models.M2TCC_Model.SANet import SANet

device = 'cuda'
input_data = torch.randn(1000, 1, 3, 768, 1024).to(device)
device_name = torch.cuda.get_device_name()

def show_inference_time(model_name, model, file):
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for i in range(1000):
            output = model(input_data[i])
            torch.cuda.synchronize()

    inference_time = (time.time() - start_time) / 1000
    fps = 1 / inference_time
    millisecond = inference_time * 1000
    print(f'{model_name}\tInference time: {millisecond:.2f} milliseconds')
    print(f'{model_name}\tFPS: {fps:.2f}')
    file.write(f'{model_name}\tInference time: {millisecond:.2f} milliseconds\n')
    file.write(f'{model_name}\tFPS: {fps:.2f}\n')
    file.flush() 

def measure_model_speed(model_name, model_class):
    if torch.cuda.is_available():
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open('inference_results.txt', 'w') as file:

                file.write(f'Results recorded at: {current_time}\n')
                file.write(f'Device: {device_name}\n')

                model = model_class().to(device)
                show_inference_time(model_name, model, file)
                print()

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print('Cuda is not available...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Speed Measurement")
    parser.add_argument('--net', type=str, required=True, help='Name of the network to test')
    args = parser.parse_args()

    model_dict = {
        'TinyCount': TinyCount,
        'MCNN': MCNN,
        'CSRNet': CSRNet,
        'SANet': SANet
    }

    if args.net in model_dict:
        print('Model speed measurement in GPU...')
        print('This task may take anywhere from a few minutes to several tens of minutes...')
        print('Device: ', device_name)
        measure_model_speed(args.net, model_dict[args.net])
    else:
        print(f"Model {args.net} not recognized. Available models are: {', '.join(model_dict.keys())}")
