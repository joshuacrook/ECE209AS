import numpy as np
import time
import tflite_runtime.interpreter as tflite

if __name__ == '__main__':
  IN_STEPS=32
  OUT_STEPS=8
  data_name = 'out_61_quiet_outside'

  test_inputs = np.load(data_name + '_test_inputs.npy')
  test_labels = np.load(data_name + '_test_labels.npy')

  model_path = str(IN_STEPS) + '_' + str(OUT_STEPS) + '/' 
  #model_name = 'model_linear.tflite'
  #model_name = 'model_dense.tflite'
  model_name = 'model_conv.tflite'
  #model_name = 'model_lstm.tflite'
  num_threads = 1

  interpreter = tflite.Interpreter(model_path = model_path + model_name, num_threads=num_threads)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  trials = int(test_labels.shape[0]/OUT_STEPS)

  inference_time = 0
  for i in range(trials):
    start_time = time.time()
    input_data = np.float32(np.expand_dims(test_inputs[i*OUT_STEPS], axis=0))
    #start_time = time.time() # faster if formatting input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    stop_time = time.time()
    inference_time += stop_time-start_time
  print('Inference frequency:', (trials*OUT_STEPS)/inference_time, 'Hz')
