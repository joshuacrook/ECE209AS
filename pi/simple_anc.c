#include <alsa/asoundlib.h>

#define N_CHANNELS 2
//#define SAMPLE_RATE 48000
#define SAMPLE_RATE 44100
//#define SAMPLE_RATE 22050
//#define SAMPLE_RATE 16000
//#define SAMPLE_RATE 8000

//#define PERIOD_SIZE 1024
//#define PERIOD_SIZE 256
#define PERIOD_SIZE 32              // number of frames
#define FRAME_SIZE (N_CHANNELS * 4) // 2 channels * 4-byte (32-bit) samples
#define BUF_SIZE (PERIOD_SIZE * FRAME_SIZE)

//#define SAVE_CAPTURE // output raw audio capture to file

static char *device_out = "plughw:0,0"; /* playback device */
static char *device_in = "dmic_sv";     /* capture device */

#ifdef SAVE_CAPTURE
FILE *fout = NULL;
static char *fout_name = "capture.raw";
#endif

/*
 * Inspired by the following:
 * https://www.alsa-project.org/alsa-doc/alsa-lib/examples.html
 * https://jan.newmarch.name/LinuxSound/Sampled/Alsa/
 * http://equalarea.com/paul/alsa-audio.html
 * https://www.linuxjournal.com/article/6735
 */

static int set_hwparams(snd_pcm_t *handle, snd_pcm_hw_params_t *params)
{
    int err;
    /* Fill it in with default values. */
    if ((err = snd_pcm_hw_params_any (handle, params)) < 0) {
        fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n",
                snd_strerror (err));
        exit (1);
    }

    /* Interleaved mode */
    if ((err = snd_pcm_hw_params_set_access (handle, params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf (stderr, "cannot set access type (%s)\n",
                 snd_strerror (err));
        exit (1);
    }

    /* Signed 32-bit little-endian format */
    if ((err = snd_pcm_hw_params_set_format (handle, params, SND_PCM_FORMAT_S32_LE)) < 0) {
        fprintf (stderr, "cannot set sample format (%s)\n",
                 snd_strerror (err));
        exit (1);
    }

    /* Channels */
    if ((err = snd_pcm_hw_params_set_channels (handle, params, N_CHANNELS)) < 0) {
        fprintf (stderr, "cannot set channel count (%s)\n",
                 snd_strerror (err));
        exit (1);
    }

    unsigned int rate = SAMPLE_RATE;
    if ((err = snd_pcm_hw_params_set_rate_near (handle, params, &rate, 0)) < 0) {
        fprintf (stderr, "cannot set sample rate (%s)\n",
                 snd_strerror (err));
        exit (1);
    }
    fprintf (stderr, "Rate is %d\n", rate);

    /* Set period size */
    snd_pcm_uframes_t periodsize = PERIOD_SIZE;
    if ((err = snd_pcm_hw_params_set_period_size_near(handle, params, &periodsize, 0)) < 0) {
        fprintf (stderr, "Unable to set period size %li: %s\n", periodsize, snd_strerror(err));
        exit (1);
    }
    // fprintf (stderr, "Actual period size is %d\n", periodsize);

    /* Write the parameters to the driver */
    if ((err = snd_pcm_hw_params (handle, params)) < 0) {
        fprintf (stderr, "cannot set parameters (%s)\n",
                 snd_strerror (err));
        exit (1);
    }

    snd_pcm_uframes_t buffersize = BUF_SIZE;
    if ((err = snd_pcm_hw_params_set_buffer_size_near(handle, params, &buffersize)) < 0) {
        fprintf (stderr, "Unable to set buffer size %li: %s\n", BUF_SIZE, snd_strerror(err));
        exit (1);
    }

    snd_pcm_uframes_t p_psize;
    snd_pcm_hw_params_get_period_size(params, &p_psize, NULL);
    fprintf(stderr, "period size %d\n", p_psize);
    snd_pcm_hw_params_get_buffer_size(params, &p_psize);
    fprintf(stderr, "buffer size %d\n", p_psize);

    return 0;
}

static int set_swparams(snd_pcm_t *handle, snd_pcm_sw_params_t *swparams)
{
    int err;

    /* get the current swparams */
    err = snd_pcm_sw_params_current(handle, swparams);
    if (err < 0) {
        fprintf (stderr, "Unable to determine current swparams for playback: %s\n", snd_strerror(err));
        exit (1);
    }

    /* start the transfer when the buffer is almost full: */
    err = snd_pcm_sw_params_set_start_threshold(handle, swparams, PERIOD_SIZE);
    if (err < 0) {
        fprintf (stderr, "Unable to set start threshold mode for playback: %s\n", snd_strerror(err));
        exit (1);
    }

    /* allow the transfer when at least period_size samples can be processed */
    err = snd_pcm_sw_params_set_avail_min(handle, swparams, PERIOD_SIZE);
    if (err < 0) {
        fprintf (stderr, "Unable to set avail min for playback: %s\n", snd_strerror(err));
        exit (1);
    }

    /* write the parameters to the playback device */
    err = snd_pcm_sw_params(handle, swparams);
    if (err < 0) {
        fprintf (stderr, "Unable to set sw params for playback: %s\n", snd_strerror(err));
        exit (1);
    }
    return 0;
}

int main(void)
{
    int *buf = (int *) malloc(BUF_SIZE);

    int err;
    unsigned int i;
    snd_pcm_t *handle_out;
    snd_pcm_t *handle_in;
    snd_pcm_sframes_t frames;

    /* Open devices */
    if ((err = snd_pcm_open(&handle_out, device_out, SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
    //if ((err = snd_pcm_open(&handle_out, device_out, SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK)) < 0) {
        fprintf (stderr, "Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    if ((err = snd_pcm_open(&handle_in, device_in, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf (stderr, "Playback open error: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }

    /* Allocate hardware parameter objects */
    snd_pcm_hw_params_t *params_out;
    snd_pcm_hw_params_t *params_in;

    if ((err = snd_pcm_hw_params_malloc(&params_out)) < 0) {
        fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
                snd_strerror (err));
        exit (EXIT_FAILURE);
    }
    if ((err = snd_pcm_hw_params_malloc(&params_in)) < 0) {
        fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
                snd_strerror (err));
        exit (EXIT_FAILURE);
    }

    /* Set hardware parameters */
    if ((err = set_hwparams(handle_out, params_out)) < 0) {
        fprintf (stderr, "Setting of hwparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    if ((err = set_hwparams(handle_in, params_in)) < 0) {
        fprintf (stderr, "Setting of hwparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    snd_pcm_hw_params_free (params_out);
    snd_pcm_hw_params_free (params_in);

    /* Allocate softwareparameter objects */
    snd_pcm_sw_params_t *sw_params_out;
    snd_pcm_sw_params_t *sw_params_in;

    if ((err = snd_pcm_sw_params_malloc(&sw_params_out)) < 0) {
        fprintf (stderr, "cannot allocate software parameter structure (%s)\n",
                snd_strerror (err));
        exit (EXIT_FAILURE);
    }
    if ((err = snd_pcm_sw_params_malloc(&sw_params_in)) < 0) {
        fprintf (stderr, "cannot allocate software parameter structure (%s)\n",
                snd_strerror (err));
        exit (EXIT_FAILURE);
    }

    /* Set hardware parameters */
    if ((err = set_swparams(handle_out, sw_params_out)) < 0) {
        printf ("Setting of swparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    if ((err = set_swparams(handle_in, sw_params_in)) < 0) {
        printf ("Setting of swparams failed: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }
    snd_pcm_sw_params_free (sw_params_out);
    snd_pcm_sw_params_free (sw_params_in);

    /* Play two silent buffers first to avoid errors/noise */
    if (snd_pcm_format_set_silence(SND_PCM_FORMAT_S32_LE, buf, BUF_SIZE) < 0) {
        fprintf (stderr, "silence error\n");
        exit(1);
    }
    int n = 0;
    while (n++ < 2) {
        if (snd_pcm_writei (handle_out, buf, BUF_SIZE) < 0) {
            fprintf (stderr, "write error\n");
            exit(1);
        }
    }

#ifdef SAVE_CAPTURE
    if ((fout = fopen(fout_name, "w")) == NULL) {
        fprintf (stderr, "Can't open file for writing\n");
        exit(1);
    }
#endif

    int nread;

    while(1) {

        if ((nread = snd_pcm_readi (handle_in, buf, PERIOD_SIZE)) != PERIOD_SIZE) {
            if (nread < 0) {
                fprintf (stderr, "read from audio interface failed (%s)\n",
                snd_strerror (nread));
            } else {
                fprintf (stderr, "short read, read %d frames\n", nread);
            }
        }
#ifdef SAVE_CAPTURE
        else {
            fwrite(buf, FRAME_SIZE, nread, fout);
        }
#endif

        for (int i = 0; i < BUF_SIZE/sizeof(buf[0]); i++) {
            buf[i] = ~buf[i]; // naive nosie cancellation (invert samples)
        }

        frames = snd_pcm_writei(handle_out, buf, PERIOD_SIZE);
        if (frames < 0) {
            printf("Recovering\n");
            frames = snd_pcm_recover(handle_out, frames, 0);
        }
        if (frames < 0) {
            printf("snd_pcm_writei failed: %s\n", snd_strerror(frames));
            break;
        }
        /*if (frames > 0 && frames < (long)buf_size)
            printf("Short write (expected %li, wrote %li)\n", (long)buf_size, frames);*/
    }

    /* pass the remaining samples, otherwise they're dropped in close */
    err = snd_pcm_drain(handle_out);
    if (err < 0)
        printf("snd_pcm_drain failed: %s\n", snd_strerror(err));
    snd_pcm_close(handle_out);
    snd_pcm_close(handle_in);

    free(buf);

#ifdef SAVE_CAPTURE
    fclose(fout);
#endif

    return 0;
}
