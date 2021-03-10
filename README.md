# Active Noise Cancellation for Windows using Audio Exciters

Project for ECE 209AS - Special Topics in Circuits and Embedded Systems: Artificial Intelligence and Machine Learning for IoT and HCPS.

## Background

Loud environments created by busy roads, airports, or otherwise heavily trafficked locations can negatively affect the health and productivity of individuals in nearby buildings. Out of all the surfaces in a building, windows generally transmit the most noise from outside and are difficult to insulate against sound. This project proposes an affordable active noise cancellation system that uses audio exciters mounted on windows to reduce noise levels transmitted into the building.

## Goals and Specific Aims

In short, the goal of this project is to transmit vibrations that are 180 degrees out of phase with the outside noise directly into windowpanes, cancelling outside noises before they can be transmitted into the building.

The first major component of this project is to develop an effective, low latency noise cancellation algorithm. More specifically, this project will explore using time series forecasting with neural networks to predict the cancellation waveform. The idea behind this approach is that a neural network will be able to learn and adapt to the different dynamics at play in the system, as well as respond to predictable noises with higher accuracy.

The second component is integrating the processing power, amplifiers, microphones, and audio exciters together in a way that actually works. The system will need to receive, process, and transmit audio with low latency. System response will not be linear, and the algorithms will need to compensate accordingly. Dealing with latency and the predictive aspect of the project are especially important since affordable, general purpose components are to be used.

## Technical Approach

### Hardware List

Compute
* [Raspberry Pi 4B (4GB)](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)

Audio
* [8W Shure Electronics amplifier board](https://www.parts-express.com/Sure-AA-AB32231-2x8W-at-4-Ohm-TPA3110-Class-D-Audio-Amplifier-Board-Only-320-329)
* [5W Dayton Audio audio exciter pair](https://www.parts-express.com/Dayton-Audio-DAEX25-Sound-Exciter-Pair-300-375)
* Approach 1: [Audio injector sound card](http://www.audioinjector.net/rpi-hat) + [electret microphone](https://www.adafruit.com/product/1063) x2
* Approach 2: [I2S MEMS microphone](https://www.adafruit.com/product/3421) x2

### Software Libraries

The noise cancellation algorithm uses the Keras/Tensorflow machine learning framework for training, and TFLite for execution. The audio processing is done through asoundlib, an application interface library for the Linux ALSA driver.

## Implementation and Experiments

### Data Collection

The main microphone is placed outside the window and measures the incoming noise. A second microphone used during training is placed inside to measure the noise that passes through the window, that the ANC algorithm will predict and cancel.

#### Approach 1

The first approach is to use a [Audio Injector sound card](http://www.audioinjector.net/rpi-hat) and [electret microphones](https://www.adafruit.com/product/1063). The exact brand of microphone does not matter, it is just an electret microphone with a MAX4466 amplifier and adjustable gain. The adjustable gain puts the output voltages in an appropriate range for line level applications. Instructions for setting up the Raspberry Pi to use the Audio Injector sound card can be found on their [GitHub](https://github.com/Audio-Injector/stereo-and-zero).

In practice, especially when recording in quiet environments, the noise can overpower the target sound. As seen in the [datasheet](https://cdn-shop.adafruit.com/datasheets/CMA-4544PF-W.pdf), the electret microphone has a signal-to-noise ratio (SNR) of 60 dBA. 60 dBA by itself is not terrible, but the noise is then amplified by the MAX4466. Additionally, the MAX4466 introduces some noise by itself ([datasheet](https://cdn-shop.adafruit.com/datasheets/MAX4465-MAX4469.pdf)), compounding the issue.

Tests showed the microphone worked fine to detect loud, outdoor sounds, but the noise overpowered the quieter indoor sounds that are required for training.

#### Approach 2

The second approach takes advantage of the Raspberry Pi's I2S support with [I2S MEMS microphone](https://www.adafruit.com/product/3421). As seen in the [datasheet](https://cdn-shop.adafruit.com/product-files/3421/i2S+Datasheet.PDF), the MEMS microphone has a slightly better signal-to-noise ratio (SNR) of 64 dBA, but the real improvement is that it is digital, so the noise is not amplified or compounded with noise from an amplifier. There is still noise in quiet environments, but it is an improvement over the previous approach. Microphone setup and installation instructions can be found on [Adafruit's guide](https://learn.adafruit.com/adafruit-i2s-mems-microphone-breakout/raspberry-pi-wiring-test).

### ANC Algorithm

As mentioned previously, the ANC algorithm is a neural network trained with Keras/Tensorflow and executed with TFLite on the Raspberry Pi. In order to facilitate training and testing, a Python script was created that allows for generating data for models with different numbers of inputs, output predictions and offsets. The script will then train the model for a given architecture (which can also be easily changed), test accuracy, save residual noise for listening, and save the TFLite model for execution and testing latency on the Raspberry Pi.

### Linux Audio Processing

Audio processing on the Raspberry Pi is done using the ALSA APIs. ALSA (Advanced Linux Sound Architecture) is a low-level interface for sound devices on Linux systems.

The following resources give a good overview of how to work with ALSA:
* [Introduction to Sound Programming with ALSA](https://www.linuxjournal.com/article/6735) by Jeff Tranter
* [A Tutorial on Using the ALSA Audio API](http://equalarea.com/paul/alsa-audio.html) by Paul Davis
* [Programming and Using Linux Sound](https://jan.newmarch.name/LinuxSound/) ([ALSA chapter](https://jan.newmarch.name/LinuxSound/Sampled/Alsa/)) by Jan Newmarch

The key points that affect low latency systems are the potential limitations to settings such as channel number, sample rate, format, and buffer sizes, all of which depend on hardware support. In some cases, ALSA can automatically resample or otherwise modify the stream in real-time to meet certain setting requirements (at the cost of CPU overhead), but there are still limitations. Most importantly, minimum period size is limited, meaning the buffers can only be read or updated after a certain number of samples.

### System Latency

#### ALSA Latency

As mentioned previously, ALSA limits the minimum period size and therefore there is a minimum latency. To understand this better, the way ALSA structures its data needs to be understood. ALSA structures its input and output data in samples, frames, periods, and buffers. A sample is one unit of data for one channel, and may range between 8 and 32 bits, depending on hardware support. A frame includes samples for both the left and right channels (assuming a two channels system in interleaved mode). A period contains a certain number of frames, and only after each period can a hardware interrupt be generated to refresh the data. Finally, the buffer contains multiple periods.

ALSA includes a [latency test](https://www.alsa-project.org/main/index.php/Test_latency.c) that finds the read latency of capture and playback devices. Latency cannot be measured between the I2S capture device and the 3.5mm playback device on the Raspberry Pi, but it can run using I2S as both the capture and playback device. For unknown reasons, latency.c assigns the minimum period size for the I2S devices as 64 frames, even though they support down to 32 frames as returned by `snd_pcm_hw_params_get_period_size_min`. Output from latency.c shows the maximum read latency is exactly 1/frequency * samples, which supports the statement earlier that captured data can only be read once per period.

For this project, a working period size of 32 frames was achieved for the I2S capture device, but only 256 frames for the playback device. It is unclear why the minimum assignable frame size is 256 for the playback device when `snd_pcm_hw_params_get_period_size_min` returns 80. Regardless, operating the capture device at 32 frames and the playback device at 256 frames does not create any adverse effects. Perhaps the playback device is interrupted when `snd_pcm_writei` is called so it does not attempt to play all 256 frames.

The table below shows the read latencies and frequencies at different sampling rates and frame counts.

| Frames | Sampling rate | Latency (us) | Latency (ms) | Frequency   |
| ------ | ------------- | ------------ | ------------ | ----------- |
| 32     | 44100         | 725.624      | 0.725624     | 1378.125 Hz |
|        | 22050         | 1451.247     | 1.451247     | 689.063 Hz  |
|        | 16000         | 2000.000     | 2.000000     | 500.000 Hz  |
| 64     | 44100         | 1451.247     | 1.451247     | 689.063 Hz  |
|        | 22050         | 2902.494     | 2.902494     | 344.531 Hz  |
|        | 16000         | 4000.000     | 4.000000     | 250.000 Hz  |

#### Processing Latency

TODO

#### Hardware Latency

TODO

## Prior Work

[DeNoize](https://denoize.com/)

[Sono noise cancelling system](https://www.ippinka.com/blog/sono-peace-quiet-home/)

[Active control of broadband sound through the open aperture of a full-sized domestic window](https://www.researchgate.net/publication/342821305_Active_control_of_broadband_sound_through_the_open_aperture_of_a_full-sized_domestic_window) by Bhan Lam, Dongyuan Shi, Woon-Seng Gan, Stephen J. Elliott & Masaharu Nishimura.

[Directional cancellation of acoustic noise for home window applications](https://www.sciencedirect.com/science/article/abs/pii/S0003682X12002599) by S. Hu, R. Rajamani, X. Yu.

[Long Short-Term Memory and Convolutional Neural Networks for Active Noise Control](https://ieeexplore.ieee.org/document/8938042) by Samuel Park, Eric Patterson, Carl Baum.

## Future Work

TODO

## References

TODO
