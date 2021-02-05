# Active Noise Cancellation for Windows using Audio Exciters

Project for ECE 209AS - Special Topics in Circuits and Embedded Systems: Artificial Intelligence and Machine Learning for IoT and HCPS.

## Background

Loud environments created by busy roads, airports, or otherwise heavily trafficked locations can negatively affect the health and productivity of individuals in nearby buildings. Out of all the surfaces in a building, windows generally transmit the most noise from outside and are difficult to insulate against sound. This project proposes an affordable active noise cancellation system that uses audio exciters mounted on windows to reduce noise levels transmitted into the building.

## Goals and Specific Aims

In short, the goal of this project is to transmit vibrations that are 180 degrees out of phase with the outside noise directly into windowpanes, cancelling outside noises before they can be transmitted into the building.

The first major component of this project is to develop an effective, low latency noise cancellation algorithm. More specifically, this project will explore using time series forecasting with neural networks to predict the cancellation waveform. The idea behind this approach is that a neural network will be able to learn and adapt to the different dynamics at play in the system, as well as respond to predictable noises with higher accuracy.

The second component is integrating the processing power, amplifiers, microphones, and audio exciters together in a way that actually works. The system will need to receive, process, and transmit audio with low latency. System response will not be linear, and the algorithms will need to compensate accordingly. Dealing with latency and the predictive aspect of the project are especially important since affordable, general purpose components are to be used.