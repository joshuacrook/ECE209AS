import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable GPU, CPU trains better on some tasks
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile

# inspired by https://www.tensorflow.org/tutorials/structured_data/time_series
class WindowGenerator():
  def __init__(self, input_width, label_width,
               label_offset, train_df, val_df, test_df):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.label_offset = label_offset

    self.total_window_size = input_width + label_offset

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, self.label_start + self.label_width)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
    inputs = features[self.input_slice]
    labels = features[self.labels_slice]
    return inputs, labels

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}'])


def chunk_data(window, dataset):
  all_inputs = []
  all_labels = []
  for i in range(len(dataset)-window.total_window_size):
    inputs = dataset[window.input_indices+i]
    labels = -dataset[window.label_indices+i] # labels are simply inverted
    all_inputs.append(inputs)
    all_labels.append(labels)
  return np.array(all_inputs), np.array(all_labels)


def define_model(arch, in_steps, out_steps):
  model = None
  if arch == 'linear':
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=out_steps)
    ])
  elif arch == 'dense':
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=out_steps)
    ])
  elif arch == 'conv':
    model = tf.keras.Sequential([
      tf.keras.layers.Reshape((in_steps, 1), input_shape=(in_steps, )),
      tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=out_steps),
    ])
  elif arch == 'lstm':
    model = tf.keras.Sequential([
      tf.keras.layers.Reshape((in_steps, 1), input_shape=(in_steps, )), # in_steps timesteps, 1 feature
      tf.keras.layers.LSTM(units=32, return_sequences=False),
      tf.keras.layers.Dense(units=out_steps)
    ])
  else:
    print('Unknown architecture:', arch)
    sys.exit()

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[
                         tf.metrics.MeanAbsoluteError(),
                         tf.metrics.MeanSquaredError()
                         ])
  return model


if __name__ == "__main__":
  path = './'
  audio_file = path + 'out_61_quiet_outside.wav'

  anc = 'nn' # naive or nn anc (nn also tests naive)
  nn_task = 'both' # train or test neural network model, or both
  model_arch = 'conv' # linear, dense, conv, lstm

  # define window parameters
  # examples:
    # IN_STEPS = 32, OFFSET = 0, OUT_STEPS = 32
    #   given 32 samples, output the corresponding anti-noise
    # IN_STEPS = 32, OFFSET = 8, OUT_STEPS = 32
    #   given 32 samples, output the corresponding anti-noise for the last 24 samples,
    #   plus predict 8 anti-noise samples into the future
    # IN_STEPS = 32, OFFSET = 32, OUT_STEPS = 32
    #   given 32 samples, predict the next 32 samples of anti-noise
  IN_STEPS=32
  OFFSET=64
  OUT_STEPS=32

  TEST_DELAY=64 # frame delay due to driver/hardware

  # prepare audio data
  print("Loading", audio_file)
  sample_rate, data = scipy.io.wavfile.read(audio_file)
  print("Sample rate:", sample_rate)
  data = np.interp(data, (-32768, 32767), (0, 1))
  n = len(data)
  train_df = data[0:int(n*0.7)]
  val_df = data[int(n*0.7):int(n*0.9)]
  test_df = data[int(n*0.9):]
  w1 = WindowGenerator(input_width=IN_STEPS, label_width=OUT_STEPS,label_offset=OFFSET,
                       train_df=train_df, val_df=val_df, test_df=test_df)
  print(w1)

  # prepare workspace
  run_folder = path + str(IN_STEPS) + '-' + str(OFFSET) + '-' + str(OUT_STEPS) \
                 + '-' + str(TEST_DELAY) + '-' + str(sample_rate) + '/'
  if not os.path.exists(run_folder):
    os.makedirs(run_folder)

  # define and train neural network model
  if anc == 'nn':
    model = define_model(model_arch, IN_STEPS, OUT_STEPS)
    print("Architecture:", model_arch)

    if nn_task == 'info':
      model.build((None, IN_STEPS))
      print(model.summary())
      sys.exit()

    if nn_task == 'train' or nn_task == 'both':
      print("Training model")
      train_inputs, train_labels = chunk_data(w1, w1.train_df)
      history = model.fit(train_inputs, train_labels, batch_size=32, epochs=1)

      print("Evaluating model")
      val_inputs, val_labels = chunk_data(w1, w1.val_df)
      results = model.evaluate(np.array(val_inputs), np.array(val_labels))

      print("Converting model")
      converter = tf.lite.TFLiteConverter.from_keras_model(model)
      tflite_model = converter.convert()
      with open(run_folder + 'model_' + model_arch + '.tflite', 'wb') as f:
        f.write(tflite_model)

  # test ANC (naive or NN)
  if anc == 'naive' or nn_task != 'train':
    print("Testing ANC")
    test_inputs, test_labels = chunk_data(w1, w1.test_df)

    # save data for testing on the Raspberry Pi
    #np.save(path + 'out_61_quiet_outside_test_inputs.npy', test_inputs)
    #np.save(path + 'out_61_quiet_outside_test_labels.npy', test_labels)
    #test_inputs = np.load(path + 'out_61_quiet_outside_test_inputs.npy')
    #test_labels = np.load(path + 'out_61_quiet_outside_test_labels.npy')

    # prepare TFLite model for testing
    if anc == 'nn':
      model_file = run_folder + 'model_' + model_arch + '.tflite'
      num_threads = 1
      interpreter = tf.lite.Interpreter(model_path=model_file, num_threads=num_threads)
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      out_pred = np.asarray([], dtype=np.float32)

    in_original = np.asarray([], dtype=np.float32)
    out_truth = np.asarray([], dtype=np.float32)
    trials = int(test_labels.shape[0]/OUT_STEPS - 2)

    # run ANC algorithm (NN and naive)
    for i in range(trials):
      input_data = np.float32(np.expand_dims(test_inputs[i*OUT_STEPS], axis=0))
      if anc == 'nn':
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        out_pred = np.concatenate((out_pred, out[0]))
      in_original = np.concatenate((in_original, test_inputs[OFFSET+i*OUT_STEPS]))
      out_truth = np.concatenate((out_truth, test_labels[i*OUT_STEPS]))
      if i%1000 == 0:
        print(str(i) + '\t/ ' + str(trials))

    # save resulting waveforms

    scipy.io.wavfile.write(run_folder + 'out_' + 'truth.wav', sample_rate, out_truth)
    scipy.io.wavfile.write(run_folder + 'out_' + 'anc_naive.wav', sample_rate, in_original+out_truth)
    #scipy.io.wavfile.write(run_folder + 'out_' + 'sub.wav', sample_rate, out_truth-out_pred)

    # model naive system delay
    in_original_shift = np.pad(in_original, (0, TEST_DELAY), 'constant')
    out_truth_shift = np.pad(out_truth, (TEST_DELAY, 0), 'constant')
    scipy.io.wavfile.write(run_folder + 'out_' + 'anc_naive_delay.wav', sample_rate, in_original_shift+out_truth_shift)

    if anc == 'nn':
      mse = np.square(np.subtract(out_truth, out_pred)).mean()
      mae = np.absolute(np.subtract(out_truth, out_pred)).mean()
      print("MSE:", mse)
      print("MAE:", mae)
      scipy.io.wavfile.write(run_folder + 'out_' + 'nn_pred.wav', sample_rate, out_pred)
      scipy.io.wavfile.write(run_folder + 'out_' + 'nn_anc.wav', sample_rate, in_original+out_pred)
      # model nn system delay only if the network can't predict that far
      if TEST_DELAY > OFFSET:
        in_original_shift = np.pad(in_original, (0, TEST_DELAY-OFFSET), 'constant')
        out_pred_shift = np.pad(out_pred, (TEST_DELAY-OFFSET, 0), 'constant')
        scipy.io.wavfile.write(run_folder + 'out_' + 'nn_anc_delay.wav', sample_rate, in_original_shift+out_pred_shift)
